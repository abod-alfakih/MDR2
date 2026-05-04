import csv
import numpy as np
import torch


def calc_loss_gradient(model, loss_fn, sample, target):
    """Compute gradients of the loss with respect to model parameters.

    This helper converts inputs to float32, enables gradient tracking for the
    sample, performs a forward pass, and returns a list of gradients for every
    trainable parameter.

    Parameters:
    - model: PyTorch model instance.
    - loss_fn: loss function that returns a tuple of losses.
    - sample: input tensor for a single sample or batch.
    - target: ground-truth tensor aligned with sample.

    Returns:
    - list of gradient tensors for each parameter in the model.
    """
    sample = sample.to(dtype=torch.float32)
    target = target.to(dtype=torch.float32)

    sample.requires_grad_(True)
    model.train()

    prediction = model(sample)
    bce_loss, dsc_loss = loss_fn(prediction, target)
    loss = (bce_loss + dsc_loss).mean()

    return torch.autograd.grad(loss, list(model.parameters()))


def compute_sample_grads(model, loss_fn, data, targets):
    """Compute per-sample gradients for a batch of inputs.

    Parameters:
    - model: PyTorch model instance.
    - loss_fn: loss function applied to each sample.
    - data: batch tensor of inputs.
    - targets: batch tensor of labels.

    Returns:
    - list of gradient tuples, one tuple per sample in the batch.
    """
    batch_size = len(data)
    sample_grads = [
        calc_loss_gradient(model, loss_fn, data[i:i+1], targets[i:i+1])
        for i in range(batch_size)
    ]
    return sample_grads


def compute_reference_gradient(model, loss_fn, pediatric_loader):
    """Compute the average gradient across a pediatric dataset loader.

    Parameters:
    - model: PyTorch model instance.
    - loss_fn: loss function used for gradient computation.
    - pediatric_loader: data loader yielding pediatric samples.

    Returns:
    - list of averaged gradient tensors, one per model parameter.
    """
    grads = []
    model.eval()

    for images, labels, *_ in pediatric_loader:
        images = images.cuda()
        labels = labels.cuda().float()
        grad = calc_loss_gradient(model, loss_fn, images, labels)
        grads.append([g.detach().cpu() for g in grad])

    avg_grad = [
        torch.mean(torch.stack([g[i] for g in grads]), dim=0)
        for i in range(len(grads[0]))
    ]
    return avg_grad


def select_topk_meta_cases(model, loss_fn, pediatric_dataset, reference_grad, k=10):
    """Select the top-k pediatric samples by gradient similarity.

    Parameters:
    - model: PyTorch model instance.
    - loss_fn: loss function used for gradient computation.
    - pediatric_dataset: dataset returning samples as (x, y, *extra).
    - reference_grad: list of gradient tensors to compare against.
    - k: number of top samples to return.

    Returns:
    - list of pediatric dataset entries corresponding to the top-k similarity.
    """
    similarities = []
    for idx in range(len(pediatric_dataset)):
        x, y, *_ = pediatric_dataset[idx]
        x = x.unsqueeze(0).cuda()
        y = y.unsqueeze(0).cuda().float()
        grad = calc_loss_gradient(model, loss_fn, x, y)

        grad_flat = torch.cat([g.flatten() for g in grad]).cuda()
        ref_flat = torch.cat([g.flatten().cuda() for g in reference_grad])
        sim = torch.nn.functional.cosine_similarity(grad_flat, ref_flat, dim=0)
        similarities.append(sim.item())

    topk_indices = np.argsort(similarities)[-k:]
    return [pediatric_dataset[i] for i in topk_indices]


def get_param_indices_by_prefix(model, prefix, specific_layers=None):
    """Return parameter indices whose names start with a given prefix.

    Parameters:
    - model: PyTorch model instance.
    - prefix: string prefix to match against parameter names.
    - specific_layers: optional list of suffix strings to further filter.

    Returns:
    - list of integer indices for matching model parameters.
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
    return selected_indices


def compute_meta_weights(model, loss_fn, batch_inputs, batch_targets, meta_inputs, meta_labels, prefix_list=('encoder', 'decoder')):
    """Compute normalized sample weights from meta-gradient similarity.

    Parameters:
    - model: PyTorch model instance.
    - loss_fn: loss function used for training and meta-gradient computation.
    - batch_inputs: training batch inputs.
    - batch_targets: training batch targets.
    - meta_inputs: meta sample inputs used for reference gradient.
    - meta_labels: meta sample labels used for reference gradient.
    - prefix_list: list of parameter prefixes to include in similarity.

    Returns:
    - tensor of normalized weights for each sample in batch_inputs.
    """
    train_per_sample_gradients = compute_sample_grads(model, loss_fn, batch_inputs, batch_targets)
    meta_gradients = calc_loss_gradient(model, loss_fn, meta_inputs, meta_labels)

    selected_indices = []
    for prefix in prefix_list:
        selected_indices.extend(get_param_indices_by_prefix(model, prefix))

    with torch.no_grad():
        cosine_distance = []
        for grads in train_per_sample_gradients:
            tg_sel = [grads[idx] for idx in selected_indices]
            mg_sel = [meta_gradients[idx] for idx in selected_indices]
            dot_product_term = sum(torch.sum(tg * mg) for tg, mg in zip(tg_sel, mg_sel))
            tg_norm_term = sum(torch.sum(tg * tg) for tg in tg_sel)
            mg_norm_term = sum(torch.sum(mg * mg) for mg in mg_sel)
            cosine_distance_term = dot_product_term / (torch.sqrt(tg_norm_term) * torch.sqrt(mg_norm_term))
            cosine_distance.append(cosine_distance_term)

        cosine_distance = torch.stack(cosine_distance)
        cosine_distance = torch.clamp(cosine_distance, min=0)
        norm_v = torch.sum(cosine_distance)
        return cosine_distance / norm_v if norm_v != 0 else cosine_distance


def log_meta_case_selection(meta_cases, epoch, log_path="meta_case_log.csv"):
    """Write selected meta-case names to a CSV file for later tracking."""
    with open(log_path, "a", newline='') as csvfile:
        writer = csv.writer(csvfile)
        for _, _, _, case_name in meta_cases:
            writer.writerow([epoch, case_name])
