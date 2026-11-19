import numpy as np
import torch


def metadata(source, dt, scene, dataset):
    m = dict(source)
    for k in ("vel_mean", "vel_std", "acc_std", "connectivity_radius"):
        if k not in m:
            raise ValueError("Source normalization missing " + k)
    # Prepared metadata uses dt=1 discrete differences. Model inputs use physical seconds.
    m.update(
        dt=dt,
        scale=1.0,
        material_id=(
            0
            if "rope" in scene
            else (1 if ("cloth" in scene or "paperbag" in scene) else 2)
        ),
        control_availability_mask=[True] * 6
        + ([False] * 9 if dataset == "phystwin" else [True] * 9),
        normalization=dict(
            legacy_vel_mean=m["vel_mean"],
            legacy_vel_std=m["vel_std"],
            velocity_std=(np.asarray(m["vel_std"]) / dt).tolist(),
            acceleration_std=(np.asarray(m["acc_std"]) / dt**2).tolist(),
        ),
    )
    # Bounds remain metadata provenance, never an input to either model.
    return m


def adapt(b, meta):
    seq = b["position_sequence"]
    B, N, _, _ = seq.shape
    H = b["next_grippers"].shape[1]
    C = b["next_grippers"].shape[2]
    n = N - C
    dt = meta["dt"]
    ch = seq.new_zeros(B, C, 3, 15)
    ch[..., :3] = seq[:, n:]
    # Pose at the current instant is known; interval forward velocities are not copied to future timestamps.
    ch[:, :, -1, 6:] = b["actions"][:, 0, :, 6:]
    path = torch.cat((b["actions"][:, :1], b["next_grippers"]), 1).clone()
    path[..., 3:6] = 0
    path[:, 1:, :, 3:6] = (path[:, 1:, :, :3] - path[:, :-1, :, :3]) / dt
    path[:, 0, :, 3:6] = (seq[:, n:, -1] - seq[:, n:, -2]) / dt
    path[..., 10:13] = 0
    if meta["control_availability_mask"][6]:
        from kornia.geometry.conversions import (
            quaternion_to_rotation_matrix,
            rotation_matrix_to_axis_angle,
        )

        r = quaternion_to_rotation_matrix(path[..., 6:10].reshape(-1, 4)).reshape(
            B, H + 1, C, 3, 3
        )
        omega = (
            rotation_matrix_to_axis_angle(
                (r[:, :-1] @ r[:, 1:].transpose(-1, -2)).reshape(-1, 3, 3)
            ).reshape(B, H, C, 3)
            / dt
        )
        path[:, 1:, :, 10:13] = omega
    result = dict(
        hist=seq[:, :n],
        hist_times=torch.arange(-2, 1, device=seq.device, dtype=seq.dtype)[None].expand(
            B, -1
        )
        * dt,
        ctrl_hist=ch,
        ctrl_path=path,
        path_times=torch.arange(H + 1, device=seq.device, dtype=seq.dtype)[None].expand(
            B, -1
        )
        * dt,
        path_mask=torch.ones(B, H + 1, device=seq.device, dtype=torch.bool),
        query_times=torch.arange(1, H + 1, device=seq.device, dtype=seq.dtype)[
            None
        ].expand(B, -1)
        * dt,
        initial_ref=b["initial_ref"],
        reference_valid=b["reference_valid"],
        initial_ctrl_ref=b["initial_ctrl_ref"],
        ctrl_available=meta["control_availability_mask"],
    )
    result.update(
        {k: b[k] for k in ("patch_anchors", "patch_spacing", "reference_center")}
    )
    if "ctrl_available" in b:
        result["ctrl_available"] = b["ctrl_available"]
    return result
