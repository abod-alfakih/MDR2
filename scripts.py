import os
import time
import warnings
from copy import deepcopy
from os.path import join
warnings.filterwarnings("ignore")
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import csv
import numpy as np
import torch
import torchopt
import functorch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

import torch.nn as nn
from monai.inferers import sliding_window_inference
from torch.cuda.amp import GradScaler, autocast
import torchopt
import utils.metrics as metrics
from configs import parse_seg_args
from dataset import brats2021
from models import get_unet
from utils.loss import clean_SoftDiceBCEWithLogitsLoss,SoftDiceBCEWithLogitsLoss
from utils.misc import (AverageMeter, CaseSegMetricsMeterBraTS, ProgressMeter, LeaderboardBraTS,
                        brats_post_processing, initialization, load_cases_split, save_brats_nifti)
from utils.optim import get_optimizer, get_clean_optimizer
from utils.scheduler import get_scheduler,get_clean_scheduler

def print_model_summary(model):
    """Print the model parameter names, shapes, and total parameter count.

    This helper is useful when inspecting a network architecture or verifying
    that all expected layers are present before training or inference.

    Parameters:
    - model: a PyTorch nn.Module whose named parameters should be displayed.
    """
    print("Model Summary:")
    total_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        print(f"Layer: {name}, Size: {param.size()}")
    print(f"Total Parameters: {total_params}")




def infer(args, epoch, model:nn.Module, infer_loader, writer, logger, mode:str, save_pred:bool=False):
    """Run inference and compute evaluation metrics for a dataset loader.

    Parameters:
    - args: parsed command-line arguments containing experiment settings.
    - epoch: current epoch number used for logging and output folder naming.
    - model: PyTorch model used for prediction.
    - infer_loader: data loader for validation or test cases.
    - writer: tensorboard writer used for logging scalar metrics.
    - logger: logger used for printing training progress.
    - mode: a string such as 'val' or 'test' used in logging and directories.
    - save_pred: if True, predicted segmentations are saved to disk.

    Returns:
    - infer_metrics: dictionary of averaged metrics for the loader.
    """
    model.eval()

    batch_time = AverageMeter('Time', ':6.3f')
    case_metrics_meter = CaseSegMetricsMeterBraTS()

    folder_dir = mode if epoch is None else f"{mode}_epoch_{epoch:02d}"
    save_path = join(args.exp_dir, folder_dir)
    if not os.path.exists(save_path):
        os.system(f"mkdir -p {save_path}")

    with torch.no_grad():
        end = time.time()
        for i, (image, label, _, brats_names) in enumerate(infer_loader):
            image, label = image.cuda(), label.bool().cuda()
            bsz = image.size(0)

            seg_map = sliding_window_inference(
                inputs=image,
                predictor=model,
                roi_size=args.patch_size,
                sw_batch_size=args.sw_batch_size,
                overlap=args.patch_overlap,
                mode=args.sliding_window_mode
            )

            seg_map = torch.where(seg_map > 0.5, True, False)

            seg_map = brats_post_processing(seg_map)

            dice = metrics.dice(seg_map, label)
            hd95 = metrics.hd95(seg_map, label)

            if save_pred:
                save_brats_nifti(seg_map, brats_names, mode, args.data_root, save_path)

            torch.cuda.synchronize()
            batch_time.update(time.time() - end)
            case_metrics_meter.update(dice, hd95, brats_names, bsz)

            if (i == 0) or (i + 1) % args.print_freq == 0:
                mean_metrics = case_metrics_meter.mean()
                logger.info("\t".join([
                    f'{mode.capitalize()}: [{epoch}][{i + 1}/{len(infer_loader)}]', str(batch_time),
                    f"Dice_WT {dice[:, 1].mean():.3f} ({mean_metrics['Dice_WT']:.3f})",
                    f"Dice_TC {dice[:, 0].mean():.3f} ({mean_metrics['Dice_TC']:.3f})",
                    f"Dice_ET {dice[:, 2].mean():.3f} ({mean_metrics['Dice_ET']:.3f})",
                    f"HD95_WT {hd95[:, 1].mean():7.3f} ({mean_metrics['HD95_WT']:7.3f})",
                    f"HD95_TC {hd95[:, 0].mean():7.3f} ({mean_metrics['HD95_TC']:7.3f})",
                    f"HD95_ET {hd95[:, 2].mean():7.3f} ({mean_metrics['HD95_ET']:7.3f})",
                ]))

            end = time.time()

        case_metrics_meter.output(save_path)

    infer_metrics = case_metrics_meter.mean()
    for key, value in infer_metrics.items():
        writer.add_scalar(f"{mode}/{key}", value, epoch)

    return infer_metrics


def save_loss_history(loss_dict, exp_dir):
    """
    Save the loss history to separate text files in the specified directory.

    Parameters:
    - loss_dict: A dictionary where keys are loss names and values are lists of loss values.
    - exp_dir: The directory where the loss history files will be saved.
    """
    os.makedirs(exp_dir, exist_ok=True)

    for loss_name, loss_values in loss_dict.items():
        file_path = os.path.join(exp_dir, f"{loss_name}_history.txt")
        with open(file_path, 'w') as f:
            for value in loss_values:
                f.write(f"{value}\n")
def calc_loss_gradient(model, loss_fn, sample, target):
    """Compute gradients of the loss with respect to model parameters.

    This function converts the input sample and target to float32, enables
    gradient tracking for the sample, runs a forward pass, and returns a list
    of gradients for each model parameter.

    Parameters:
    - model: PyTorch model used for prediction.
    - loss_fn: loss function returning a tuple of losses.
    - sample: input tensor for the model.
    - target: ground-truth tensor for the loss.

    Returns:
    - list of gradient tensors for every parameter in the model.
    """
    sample = sample.to(dtype=torch.float32)
    target = target.to(dtype=torch.float32)

    sample.requires_grad_(True)
    model.train()

    prediction = model(sample)



    bce_loss, dsc_loss = loss_fn(prediction, target)


    loss = (bce_loss + dsc_loss).mean()


    return torch.autograd.grad(loss, list(model.parameters()))
def log_meta_case_selection(meta_cases, epoch, log_path="meta_case_log.csv"):
    """Append selected meta-case names to a CSV log file for the current epoch.

    Parameters:
    - meta_cases: iterable of meta-case tuples containing case metadata.
    - epoch: epoch number associated with the selection.
    - log_path: path to the CSV log file.
    """
    with open(log_path, "a", newline='') as csvfile:
        writer = csv.writer(csvfile)
        for _, _, _, case_name in meta_cases:
            writer.writerow([epoch, case_name])

def compute_sample_grads( model, loss_fn, data, targets):
    """Compute gradients for each sample in a batch independently.

    This helper calls calc_loss_gradient for each example in the batch and
    returns a list of gradient tuples, where each tuple corresponds to one
    sample's parameter gradients.

    Parameters:
    - model: PyTorch model used for prediction.
    - loss_fn: loss function applied per sample.
    - data: batch tensor of input samples.
    - targets: batch tensor of corresponding labels.

    Returns:
    - list of gradient tuples, one per sample in the batch.
    """
    batch_size = len(data)
    sample_grads = [
        calc_loss_gradient(model, loss_fn, data[i:i+1], targets[i:i+1])
        for i in range(batch_size)
    ]
    return sample_grads

def save_learning_rate_history(learning_rate_history, exp_dir):
    """Append learning rate values to a text file in the experiment directory.

    Parameters:
    - learning_rate_history: iterable of scalar learning rate values.
    - exp_dir: directory where the history file will be stored.
    """
    lr_file_path = os.path.join(exp_dir, "learning_rate_history.txt")
    with open(lr_file_path, "a") as f:
        for lr in learning_rate_history:
            f.write(f"{lr}\n")



def compute_reference_gradient(model, loss_fn, pediatric_loader):
    """
    Compute the average gradient over all pediatric samples in the loader.
    Returns a list of tensors (one per parameter).
    """
    grads = []
    model.eval()

    for images, labels, *_ in pediatric_loader:  # ❌ no torch.no_grad() here
        images = images.cuda()
        labels = labels.cuda().float()
        grad = calc_loss_gradient(model, loss_fn, images, labels)

        grads.append([g.detach().cpu() for g in grad])

    avg_grad = [torch.mean(torch.stack([g[i] for g in grads]), dim=0) for i in range(len(grads[0]))]


    return avg_grad
def select_topk_meta_cases(model, loss_fn, pediatric_dataset, reference_grad, k=10):
    """
    Select top-k pediatric samples whose gradients are most similar to the reference gradient.
    Returns a list of (image, label, ...) tuples.
    """
    similarities = []
    grads = []
    for idx in range(len(pediatric_dataset)):
        x, y, *_ = pediatric_dataset[idx]
        x = x.unsqueeze(0).cuda()
        y = y.unsqueeze(0).cuda().float()  # Ensure labels are float
        grad = calc_loss_gradient(model, loss_fn, x, y)
        grads.append(grad)
        grad_flat = torch.cat([g.flatten() for g in grad]).cuda()
        ref_flat = torch.cat([g.flatten().cuda() for g in reference_grad])
        sim = torch.nn.functional.cosine_similarity(grad_flat, ref_flat, dim=0)
        similarities.append(sim.item())
    topk_indices = np.argsort(similarities)[-k:]
    return [pediatric_dataset[i] for i in topk_indices]

def extract_features(model, dataset, device='cuda'):
    """Extract feature embeddings from the model encoder for each dataset sample.

    The function executes the encoder on each input example, collects the final
    bottleneck feature tensor, and maps it by case name for later analysis.

    Parameters:
    - model: PyTorch model with an encoder method that returns skip connections.
    - dataset: dataset returning (input, label, idx, case_name) tuples.
    - device: device on which to run inference, usually 'cuda' or 'cpu'.

    Returns:
    - dict mapping case_name to numpy feature vectors.
    """
    model.eval()
    features = {}
    with torch.no_grad():
        for idx in range(len(dataset)):
            x, _, _, case_name = dataset[idx]
            x = x.unsqueeze(0).to(device)
            skips = model.encoder(x, return_skips=True)
            bottleneck = skips[-1].flatten().cpu().numpy()
            features[case_name] = bottleneck
    return features



def get_param_indices_by_prefix(model, prefix, specific_layers=None):
    """
    Select parameter indices from model whose name starts with the given prefix.
    If specific_layers is provided (list of suffixes), only select those layers.
    Example:
        get_param_indices_by_prefix(model, 'encoder')
        get_param_indices_by_prefix(model, 'encoder', ['conv_block.conv1.conv.weight'])
    """
    param_names = list(model.named_parameters())
    selected_indices = []
    for idx, (name, _) in enumerate(param_names):
        if name.startswith(prefix):
            if specific_layers is None:
                selected_indices.append(idx)
            else:
                for layer in specific_layers:
                    if name.endswith(layer):
                        selected_indices.append(idx)
    print(f"Total selected parameters for '{prefix}': {len(selected_indices)}")
    for idx in selected_indices:
        print(f"  - {param_names[idx][0]}")
    return selected_indices


class TrainModel:
    """Encapsulates the training workflow and meta-learning weight calculation."""

    def determine_meta_weight(self, image, label):
        """Compute sample weights based on gradient similarity with meta-data.

        This method computes per-sample gradients over the current training batch,
        compares those gradients with meta-gradient values, and returns a normalized
        weight vector. The weights are used to emphasize samples that match the
        meta-gradient direction.

        Parameters:
        - image: batch of training images.
        - label: batch of training labels.

        Returns:
        - tensor of normalized weights, one per sample in the batch.
        """
        loss_func = self.loss_2  # same as the loss function used in the outer loop
        net_state_dict = torchopt.extract_state_dict(self.model)

        train_per_sample_gradients = compute_sample_grads(
            self.model, loss_func, image, label
        )

        torchopt.recover_state_dict(self.model, net_state_dict)
        self.optimizer.zero_grad()

        meta_gradients = calc_loss_gradient(self.model, loss_func, self.meta_inputs, self.meta_labels)



        encoder_indices = get_param_indices_by_prefix(self.model, 'encoder')
        decoder_indices = get_param_indices_by_prefix(self.model, 'decoder')
        selected_indices = encoder_indices + decoder_indices

        param_names = list(self.model.named_parameters())
        with torch.no_grad():
            cosine_distance = []

            for i, grads in enumerate(train_per_sample_gradients):
                tg_sel = [grads[idx] for idx in selected_indices]
                mg_sel = [meta_gradients[idx] for idx in selected_indices]
                dot_product_term = sum([torch.sum(tg * mg)
                                        for (tg, mg) in zip(tg_sel, mg_sel)])
                tg_norm_term = sum([torch.sum(tg * tg) for tg in tg_sel])
                vg_norm_term = sum([torch.sum(mg * mg) for mg in mg_sel])
                cosine_distance_term = dot_product_term / (torch.sqrt(tg_norm_term) * torch.sqrt(vg_norm_term))
                cosine_distance.append(cosine_distance_term)
            cosine_distance = torch.stack(cosine_distance)
            cosine_distance = torch.clamp(cosine_distance, min=0)
            norm_v = torch.sum(cosine_distance)
            if norm_v != 0:
                w_v = cosine_distance / norm_v
            else:
                w_v = cosine_distance
        return w_v.detach()

    def main(self):
        """Run the training loop including meta-case selection and validation.

        This method initializes the model, loads data, optionally loads pretrained
        weights, and performs epoch-wise training. It also selects meta-cases,
        computes weights, updates the model, and runs validation and testing.
        """

        args = parse_seg_args()
        logger, writer = initialization(args)


        train_cases, val_cases, test_cases,meta_cases = load_cases_split(args.cases_split)
        val_loader = brats2021.get_infer_loader(args, val_cases)
        test_loader = brats2021.get_infer_loader(args, test_cases)
        train_cases2, val_cases, test_cases,meta_cases = load_cases_split(args.cases_split_2)
        train_loader = brats2021.create_combined_loader(args, train_cases, train_cases2)
        pediatric_loader = brats2021.get_pediatric_loader(args, train_cases) # Now returns a dataset
        pediatric_dataset = pediatric_loader.dataset  # ✅ correct
        self.model = get_unet(args).cuda()
        if args.data_parallel:
            self.model = nn.DataParallel(self.model).cuda()
        self.optimizer = get_optimizer(args, self.model)
        scheduler = get_scheduler(args, self.optimizer)
        self.loss_fn = clean_SoftDiceBCEWithLogitsLoss().cuda()
        self.loss_2 =SoftDiceBCEWithLogitsLoss().cuda()




        def load_pretrained_model(args, model, logger):
            if args.weight_path is not None:
                logger.info("==> Loading pretrain model...")
                assert args.weight_path.endswith(".pth"), "Weight path should end with .pth"

                checkpoint = torch.load(args.weight_path)

                if isinstance(checkpoint['model'], torch.nn.Module):
                    model_state = checkpoint['model'].state_dict()
                    logger.info("Extracted state_dict from model instance.")
                else:
                    model_state = checkpoint['model']
                    logger.info("Loaded state_dict directly from checkpoint.")

                model.load_state_dict(model_state)

        load_pretrained_model(args, self.model, logger)
        logger.info("==> Training starts...")
        best_model = {}
        loss_history = []  # Initialize an empty list to store loss values
        loss_history_2 = []
        loss_history_meta = []
        learning_rate_history = []  # Initialize an empty list to store learning rates

        val_leaderboard = LeaderboardBraTS()

        meta_cases_cache = None  # Cache for meta-cases
        for epoch in range(args.epochs):
            self.model.train()
            if epoch % 5 == 0 or meta_cases_cache is None:
                reference_grad = compute_reference_gradient(self.model, self.loss_fn, pediatric_loader)
                meta_cases_cache = select_topk_meta_cases(self.model, self.loss_fn, pediatric_dataset, reference_grad, k=10)
                log_meta_case_selection(meta_cases_cache, epoch)
                for case in meta_cases_cache:
                    try:
                        case_name = case[3] # case = (x, y, idx, case_name)
                        print(f"- {case_name}")
                    except IndexError:
                        print("- [Warning] Case name not found in sample. Make sure dataset returns it.")
            data_time = AverageMeter('Data', ':6.3f')
            batch_time = AverageMeter('Time', ':6.3f')
            bce_meter = AverageMeter('BCE', ':.4f')
            dsc_meter = AverageMeter('Dice', ':.4f')
            loss_meter = AverageMeter('Loss', ':.4f')
            loss_meter_2= AverageMeter('Loss_2', ':.4f')
            loss_meter_3= AverageMeter('Loss_3', ':.4f')

            progress = ProgressMeter(
                len(train_loader),
                [batch_time, data_time, bce_meter, dsc_meter, loss_meter,loss_meter_2,loss_meter_3],
                prefix=f"Train: [{epoch}]")
            end = time.time()

            for i, (image, label, _, _) in enumerate(train_loader):
                image, label = image.cuda(), label.float().cuda()
                bsz = image.size(0)
                data_time.update(time.time() - end)
                self.optimizer.zero_grad()
                meta_cases = meta_cases_cache
                self.meta_inputs = torch.stack([x for x, y, *_ in meta_cases]).cuda()
                self.meta_labels = torch.stack([y for x, y, *_  in meta_cases]).cuda()

                weights = self.determine_meta_weight(image, label)
                output = self.model(image)
                bce_loss, dsc_loss = self.loss_fn(output, label)
                total_loss_per_sample = bce_loss + dsc_loss
                mean_loss_per_sample = total_loss_per_sample.mean(dim=[1, 2, 3, 4])
                loss = mean_loss_per_sample * weights
                total_loss = torch.sum(loss)
                total_loss.backward()
                self.optimizer.step()
                torch.cuda.synchronize()
                loss_meter.update(total_loss.item(), bsz)
                batch_time.update(time.time() - end)
                if (i == 0) or (i + 1) % args.print_freq == 0:
                    progress.display(i + 1, logger)
                end = time.time()
            if scheduler is not None:
                scheduler.step()

            train_tb = {
                'bce_loss': bce_meter.avg,
                'dsc_loss': dsc_meter.avg,
                'total_loss': loss_meter.avg,
                'lr': self.optimizer.state_dict()['param_groups'][0]['lr'],
            }

            for key, value in train_tb.items():
                writer.add_scalar(f"train/{key}", value, epoch)



            if (epoch > 81):
                logger.info(f"==> Validation starts...")
                val_metrics = infer(args, epoch, self.model, val_loader, writer, logger, mode='val')


                val_leaderboard.update(epoch, val_metrics)
                best_model.update({epoch: deepcopy(self.model.state_dict())})
                logger.info(f"==> Validation ends...")
                val_leaderboard.update(epoch, val_metrics)
            torch.cuda.empty_cache()

        val_leaderboard.output(args.exp_dir)
        logger.info("==> Testing starts...")
        best_epoch = val_leaderboard.get_best_epoch()
        best_model = best_model[best_epoch]
        self.model.load_state_dict(best_model)
        infer(args, best_epoch, self.model, test_loader, writer, logger, mode='test', save_pred=args.save_pred)
        if args.save_model:
            logger.info("==> Saving...")
            state = {'model': best_model, 'epoch': best_epoch, 'args': args}
            torch.save(state, os.path.join(
                args.exp_dir, f"test_epoch_{best_epoch:02d}", f'best_ckpt.pth'))
        logger.info("==> Testing ends...")

if __name__ == '__main__':
    trainer = TrainModel()
    trainer.main()

