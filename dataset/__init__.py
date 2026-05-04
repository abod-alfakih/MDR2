# # # from .dataset_utils import nib_load, RobustZScoreNormalization
# # import nibabel as nib
# #
# # # Example path, replace this with one of the printed paths from your debugging
# # test_path = 'C:\\Users\\USER\\Master-Degree\\DATA\\BraTS2021\\archive\\BraTS2021_Training_Data\\BraTS2021_00000\\BraTS2021_00000_flair.nii.gz'
# #             C:\Users\USER\Master-Degree\DATA\BraTS2021\archive\BraTS2021_Training_Data\BraTS2021_01658\BraTS2021_01658_flair.nii.gz
# # img = nib.load(test_path)
# # print(img.shape)  # This should print the dimensions of the image if successful
#
# # import torch
# # import torch.nn.functional as F
# #
# # # Define example tensors
# # tensor1 = torch.tensor([1.0, 1.0, 3.0])
# # tensor2 = torch.tensor([-1.0, 0.1, 0.0])
# #
# # # Reshape tensors to (3, 1) to make them two-dimensional
# # # tensor1_reshaped = tensor1.view(-1, 1)
# # # print(tensor1_reshaped)
# # # tensor2_reshaped = tensor2.view(-1, 1)
# # # print(tensor2_reshaped)
# # # Compute cosine similarity along the columns (dim=1)
# # similarity = F.cosine_similarity(tensor1, tensor2, dim=0)
# #
# # print(similarity)
# #
# # import nibabel as nib
# # import numpy as np
# #
# # def dice_coefficient(y_true, y_pred, label):
# #     """Compute Dice score for one label."""
# #     y_true_bin = (y_true == label)
# #     y_pred_bin = (y_pred == label)
# #     intersection = np.sum(y_true_bin & y_pred_bin)
# #     denom = np.sum(y_true_bin) + np.sum(y_pred_bin)
# #     if denom == 0:
# #         return np.nan  # label not present in either
# #     return 2.0 * intersection / denom
# #
# # # --- Paths to your NIfTI files ---
# # gt_path = r"D:\brain\second\data\train_data\brats2021(wo-skull)\BraTS-PED-069-000\BraTS-PED-069-000-remapped-seg.nii.gz"
# # pred_path = r"D:\brain\pediatric\exp\ped+adult+one\test_epoch_83\test_seg_pred\BraTS-PED-069-000.nii.gz"
# #
# # # --- Load the volumes ---
# # gt_img = nib.load(gt_path).get_fdata().astype(np.uint8)
# # pred_img = nib.load(pred_path).get_fdata().astype(np.uint8)
# #
# # # --- Define your labels (e.g., 1: WM, 2: GM, 3: CSF, 4: Subcortical) ---
# # labels = [1, 2, 3, 4]
# #
# # # --- Compute Dice for each label ---
# # dice_scores = {}
# # for label in labels:
# #     dice = dice_coefficient(gt_img, pred_img, label)
# #     dice_scores[label] = dice
# #
# # # --- Compute mean Dice (ignoring NaNs) ---
# # mean_dice = np.nanmean(list(dice_scores.values()))
# #
# # # --- Print results ---
# # print("Dice per label:")
# # for label, dice in dice_scores.items():
# #     print(f"  Label {label}: {dice:.4f}")
# #
# # print(f"\nMean Dice: {mean_dice:.4f}")
#
#
# # ----- PATHS -----
#
#
# # import nibabel as nib
# # import numpy as np
# #
# # def dice_per_slice(gt, pred, label, axis=2):
# #     """
# #     Compute Dice for each slice along `axis` for a given label.
# #     Skips slices where GT does NOT contain this label.
# #     """
# #     assert gt.shape == pred.shape, "GT and prediction must have same shape"
# #     num_slices = gt.shape[axis]
# #     dice_scores = []
# #
# #     for i in range(num_slices):
# #         if axis == 0:
# #             gt_slice = gt[i, :, :]
# #             pred_slice = pred[i, :, :]
# #         elif axis == 1:
# #             gt_slice = gt[:, i, :]
# #             pred_slice = pred[:, i, :]
# #         else:  # axis == 2 (axial)
# #             gt_slice = gt[:, :, i]
# #             pred_slice = pred[:, :, i]
# #
# #         gt_bin = (gt_slice == label)
# #         pred_bin = (pred_slice == label)
# #
# #         # Skip slices where GT has no voxels of this label
# #         if gt_bin.sum() == 0:
# #             dice_scores.append(np.nan)
# #             continue
# #
# #         intersection = np.logical_and(gt_bin, pred_bin).sum()
# #         denom = gt_bin.sum() + pred_bin.sum()
# #
# #         if denom == 0:
# #             dice_scores.append(0.0)
# #         else:
# #             dice_scores.append(2.0 * intersection / denom)
# #
# #     return np.array(dice_scores)
# #
# #
# # # ----- YOUR PATHS -----
# # gt_path    = r"D:\brain\second\data\train_data\brats2021(wo-skull)\BraTS-PED-069-000\BraTS-PED-069-000-remapped-seg.nii.gz"
# # pred1_path = r"D:\brain\pediatric\exp\ped+adult+one\test_epoch_83\test_seg_pred\BraTS-PED-069-000.nii.gz"
# # pred2_path = r"D:\brain\second\exp\ped+adul\test_epoch_84\test_seg_pred\BraTS-PED-069-000.nii.gz"
# #
# # # ----- LOAD VOLUMES -----
# # gt_img    = nib.load(gt_path)
# # pred1_img = nib.load(pred1_path)
# # pred2_img = nib.load(pred2_path)
# #
# # gt    = gt_img.get_fdata().astype(np.uint8)
# # pred1 = pred1_img.get_fdata().astype(np.uint8)
# # pred2 = pred2_img.get_fdata().astype(np.uint8)
# #
# # print("GT shape   :", gt.shape)
# # print("Pred1 shape:", pred1.shape)
# # print("Pred2 shape:", pred2.shape)
# #
# # # Safety check
# # assert gt.shape == pred1.shape == pred2.shape, "All volumes must have the same shape!"
# #
# # # ----- SETTINGS -----
# # label_of_interest = 3   # change if your label 3 is different
# # axis = 2                # axial slices (z axis)
# #
# # # ----- COMPUTE SLICE-WISE DICE -----
# # dice1 = dice_per_slice(gt, pred1, label_of_interest, axis=axis)
# # dice2 = dice_per_slice(gt, pred2, label_of_interest, axis=axis)
# #
# # improvement = dice2 - dice1
# #
# # # Only keep slices where GT has this label (i.e., not NaN)
# # valid = ~np.isnan(improvement)
# #
# # if not valid.any():
# #     print(f"No slices where GT contains label {label_of_interest}.")
# # else:
# #     valid_indices = np.where(valid)[0]
# #     imp_valid = improvement[valid]
# #
# #     best_idx_in_valid = np.argmax(imp_valid)
# #     best_slice = valid_indices[best_idx_in_valid]
# #
# #     print(f"\nBest slice index (axis {axis}) with biggest improvement for label {label_of_interest}: {best_slice}")
# #     print(f"  Dice model 1: {dice1[best_slice]:.4f}")
# #     print(f"  Dice model 2: {dice2[best_slice]:.4f}")
# #     print(f"  Improvement : {improvement[best_slice]:.4f}")
# #
# #     # ---- DEBUG COUNTS FOR THAT SLICE ----
# #     if axis == 0:
# #         gt_slice  = gt[best_slice, :, :]
# #         p1_slice  = pred1[best_slice, :, :]
# #         p2_slice  = pred2[best_slice, :, :]
# #     elif axis == 1:
# #         gt_slice  = gt[:, best_slice, :]
# #         p1_slice  = pred1[:, best_slice, :]
# #         p2_slice  = pred2[:, best_slice, :]
# #     else:
# #         gt_slice  = gt[:, :, best_slice]
# #         p1_slice  = pred1[:, :, best_slice]
# #         p2_slice  = pred2[:, :, best_slice]
# #
# #     print("\nVoxel counts on best slice:")
# #     print("  GT label-3 voxels    :", (gt_slice == label_of_interest).sum())
# #     print("  Model 1 label-3 voxels:", (p1_slice == label_of_interest).sum())
# #     print("  Model 2 label-3 voxels:", (p2_slice == label_of_interest).sum())
# #
# #     # ---- OPTIONAL: TOP-K SLICES ----
# #     top_k = 5
# #     sorted_idx = np.argsort(-imp_valid)  # descending improvement
# #     print(f"\nTop {min(top_k, len(sorted_idx))} slices with largest improvement:")
# #     for rank in range(min(top_k, len(sorted_idx))):
# #         sl = valid_indices[sorted_idx[rank]]
# #         print(f"  Rank {rank+1}: slice {sl}, ΔDice={improvement[sl]:.4f}, "
# #               f"D1={dice1[sl]:.4f}, D2={dice2[sl]:.4f}")
# import nibabel as nib
# import numpy as np
# import matplotlib.pyplot as plt
#
# # ---------- PATHS ----------
# gt_path    = r"D:\brain\second\data\train_data\brats2021(wo-skull)\BraTS-PED-069-000\BraTS-PED-069-000-remapped-seg.nii.gz"
# pred1_path = r"D:\brain\pediatric\exp\ped+adult+one\test_epoch_83\test_seg_pred\BraTS-PED-069-000.nii.gz"
# pred2_path = r"D:\brain\second\exp\ped+adul\test_epoch_84\test_seg_pred\BraTS-PED-069-000.nii.gz"
# t1_path    = r"D:\brain\second\data\train_data\brats2021(wo-skull)\BraTS-PED-069-000\BraTS-PED-069-000-t1n.nii.gz"
#
# # ---------- LOAD ----------
# gt    = nib.load(gt_path).get_fdata().astype(np.uint8)
# pred1 = nib.load(pred1_path).get_fdata().astype(np.uint8)
# pred2 = nib.load(pred2_path).get_fdata().astype(np.uint8)
# t1    = nib.load(t1_path).get_fdata().astype(np.float32)
#
# # ---------- SETTINGS ----------
# label = 3
# slice_idx = 58
# axis = 2
#
# # ---------- SLICE EXTRACTION ----------
# if axis == 0:
#     gt_slice, p1_slice, p2_slice, t1_slice = gt[slice_idx, :, :], pred1[slice_idx, :, :], pred2[slice_idx, :, :], t1[slice_idx, :, :]
# elif axis == 1:
#     gt_slice, p1_slice, p2_slice, t1_slice = gt[:, slice_idx, :], pred1[:, slice_idx, :], pred2[:, slice_idx, :], t1[:, slice_idx, :]
# else:
#     gt_slice, p1_slice, p2_slice, t1_slice = gt[:, :, slice_idx], pred1[:, :, slice_idx], pred2[:, :, slice_idx], t1[:, :, slice_idx]
#
# gt_mask = (gt_slice == label)
# p1_mask = (p1_slice == label)
# p2_mask = (p2_slice == label)
#
# # ---------- NORMALIZE T1 ----------
# t1_norm = (t1_slice - np.min(t1_slice)) / (np.ptp(t1_slice) + 1e-8)
#
# # ---------- OVERLAY FUNCTION ----------
# def overlay(base_gray, mask, color=(1, 0, 0), alpha=0.5):
#     rgb = np.stack([base_gray]*3, axis=-1)
#     mask_bool = mask.astype(bool)
#     rgb[mask_bool] = (1 - alpha) * rgb[mask_bool] + alpha * np.array(color)
#     return rgb
#
# gt_overlay  = overlay(t1_norm, gt_mask,  color=(0, 1, 0), alpha=0.5)  # green
# p1_overlay  = overlay(t1_norm, p1_mask,  color=(1, 0, 0), alpha=0.5)  # red
# p2_overlay  = overlay(t1_norm, p2_mask,  color=(0, 0, 1), alpha=0.5)  # blue
#
# # ---------- FIND ZOOM BOX ----------
# coords = np.argwhere(gt_mask)
# if coords.size > 0:
#     y_min, x_min = coords.min(axis=0)
#     y_max, x_max = coords.max(axis=0)
#     pad = 15
#     y_min, y_max = max(0, y_min - pad), min(gt_mask.shape[0], y_max + pad)
#     x_min, x_max = max(0, x_min - pad), min(gt_mask.shape[1], x_max + pad)
# else:
#     y_min, y_max, x_min, x_max = 0, gt_mask.shape[0], 0, gt_mask.shape[1]
#
# gt_zoom = gt_overlay[y_min:y_max, x_min:x_max]
# p1_zoom = p1_overlay[y_min:y_max, x_min:x_max]
# p2_zoom = p2_overlay[y_min:y_max, x_min:x_max]
#
# # ---------- PLOT FULL AND ZOOM ----------
# fig, axs = plt.subplots(2, 3, figsize=(12, 8))
#
# # Full slice row
# axs[0,0].imshow(gt_overlay); axs[0,0].set_title("GT label 3 (full)")
# axs[0,1].imshow(p1_overlay); axs[0,1].set_title("Model 1 label 3 (full)")
# axs[0,2].imshow(p2_overlay); axs[0,2].set_title("Model 2 label 3 (full)")
#
# # Zoomed row
# axs[1,0].imshow(gt_zoom); axs[1,0].set_title("GT label 3 (zoom)")
# axs[1,1].imshow(p1_zoom); axs[1,1].set_title("Model 1 label 3 (zoom)")
# axs[1,2].imshow(p2_zoom); axs[1,2].set_title("Model 2 label 3 (zoom)")
#
# for ax in axs.flat:
#     ax.axis("off")
#
# plt.tight_layout()
# plt.savefig("overlay_label3_slice58_full_and_zoom.png", bbox_inches="tight", dpi=200)
# plt.show()
#
# print("✅ Saved: overlay_label3_slice58_full_and_zoom.png")
