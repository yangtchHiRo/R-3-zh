"""H5 training objective and continuous autoregressive inference."""

import numpy as np
import torch
from .data.adapters import adapt
from .data.stream_state import masked_loss
from .models.regions import reference_pack


def masked_rollout_loss(model, b):
    inputs = adapt(b, model.meta)
    result = model(inputs)
    loss = 100 * masked_loss(result["pred"], b["gt_x"], b["valid"])
    loss += model.auxiliary_loss(result, b["gt_x"], b["valid"], inputs["initial_ref"])
    return loss, result


@torch.no_grad()
def predict_sequence(
    model,
    sequence,
    actions,
    future,
    initial_ref,
    initial_ctrl_ref,
    patch,
    device,
    reference_valid=None,
    ctrl_available=None,
):
    """Only initial H3, reference geometry and prescribed controls enter the model.

    Roll forward in chunks of five, exactly as in the experiment implementation.
    Object predictions, not targets, form every subsequent three-frame history.
    """
    model.eval()

    def tensor(x):
        return torch.as_tensor(x, device=device)

    seq = tensor(sequence).float()
    actions = tensor(actions).float()
    future = tensor(future).float()
    ref = tensor(initial_ref).float()
    cr = tensor(initial_ctrl_ref).float()
    valid = (
        torch.ones(ref.shape[0], dtype=torch.bool, device=device)
        if reference_valid is None
        else tensor(reference_valid).bool()
    )
    common = dict(
        initial_ref=ref[None],
        initial_ctrl_ref=cr[None],
        reference_valid=valid[None],
        **{k: tensor(v)[None].float() for k, v in patch.items()}
    )
    if ctrl_available is not None:
        common["ctrl_available"] = ctrl_available
    predictions = []
    for start in range(0, len(actions), 5):
        end = min(len(actions), start + 5)
        b = dict(
            position_sequence=seq,
            actions=actions[None, start:end],
            next_grippers=future[None, start:end],
            **common
        )
        pred = model(adapt(b, model.meta))["pred"]
        if not torch.isfinite(pred).all():
            raise FloatingPointError("Nonfinite free rollout")
        for k in range(end - start):
            next_state = torch.cat(
                (pred[:, k], future[start + k : start + k + 1, :, :3]), 1
            )
            seq = torch.cat((seq[:, :, 1:], next_state[:, :, None]), 2)
        predictions.extend(pred[0].cpu().unbind(0))
    if not predictions:
        raise ValueError("No future controls")
    return torch.stack(predictions)


@torch.no_grad()
def rollout(model, case, ids, device):
    initial = case.x[:3, ids]
    if not np.isfinite(initial).all():
        raise ValueError("Initial H3 must be observed")
    sequence = (
        np.concatenate((initial, case.ctrl[:3]), 1).transpose(1, 0, 2).copy()[None]
    )
    ref = case.x[0, ids].copy()
    rv = case.vis[0, ids] & np.isfinite(ref).all(-1)
    ref[~rv] = 0
    good = case.vis[0] & np.isfinite(case.x[0]).all(-1)
    patch = reference_pack(
        case.x[0, good], case.patch_reference["patch_anchors"].shape[0]
    )
    y = predict_sequence(
        model, sequence, case.g[2:-1], case.g[3:], ref, case.ctrl[0], patch, device, rv
    )
    return np.concatenate((initial, y.numpy())), None
