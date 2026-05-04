# Reweighting and Retrieval Helpers

This folder contains reusable helper code for sample reweighting and pediatric sample retrieval.

## Files

- `methods.py`: contains gradient-based reweighting and selection utilities.
- `__init__.py`: exposes the main helper functions.

## Usage

```python
from reweighting import (
    calc_loss_gradient,
    compute_sample_grads,
    compute_reference_gradient,
    select_topk_meta_cases,
    get_param_indices_by_prefix,
    compute_meta_weights,
    log_meta_case_selection,
)
```

## Example

```python
reference_grad = compute_reference_gradient(model, loss_fn, pediatric_loader)
meta_cases = select_topk_meta_cases(model, loss_fn, pediatric_dataset, reference_grad, k=10)
weights = compute_meta_weights(
    model,
    loss_fn,
    train_inputs,
    train_targets,
    meta_inputs,
    meta_labels,
)
```

This keeps all reweighting/retrieval logic in one portable module for easy reuse or copying into another project.
