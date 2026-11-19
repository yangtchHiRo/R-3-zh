import numpy as np
from scipy.spatial import cKDTree
from ..io import finite_mean


def curves(pred, case, ids, surface_count=None):
    T = len(pred)
    gt = case.gt[:T]
    x = case.x[:T]
    vis = case.vis[:T]
    manual_ids = (
        np.flatnonzero(np.isfinite(gt[0]).all(-1)) if gt.shape[1] else np.empty(0, int)
    )
    matched = (
        cKDTree(pred[0]).query(gt[0, manual_ids])[1]
        if len(manual_ids)
        else np.empty(0, int)
    )
    first_distance = (
        cKDTree(pred[0]).query(gt[0, manual_ids])[0] if len(manual_ids) else np.empty(0)
    )
    rows = []
    chain = np.ones(case.x.shape[1], bool)
    for t in range(T):
        if t:
            chain &= case.motion[t - 1]
        count = (
            int((np.isfinite(gt[t, manual_ids]).all(-1)).sum())
            if len(manual_ids)
            else 0
        )
        row = dict(
            frame=t,
            time_s=float(t * case.entry["dt"]),
            cd_l1_m=None,
            track_l2_m=None,
            track_official_empty_zero_m=None,
            pseudo_mde_m=None,
            pseudo_mse_m2=None,
            manual_points=count,
            observed_points=0,
            pseudo_points=0,
            prediction_finite=bool(np.isfinite(pred[t]).all()),
        )
        obs = vis[t] & np.isfinite(x[t]).all(-1)
        row["observed_points"] = int(obs.sum())
        if row["prediction_finite"]:
            if obs.any():
                row["cd_l1_m"] = float(
                    cKDTree(pred[t, :surface_count]).query(x[t, obs], p=1)[0].mean()
                )
            if len(manual_ids):
                good = np.isfinite(gt[t, manual_ids]).all(-1)
                if good.any():
                    row["track_l2_m"] = float(
                        np.linalg.norm(
                            pred[t, matched[good]] - gt[t, manual_ids[good]], axis=-1
                        ).mean()
                    )
            row["track_official_empty_zero_m"] = (
                row["track_l2_m"]
                if row["track_l2_m"] is not None
                else (0.0 if len(manual_ids) else None)
            )
            if ids is not None:
                mask = obs[ids] & chain[ids]
                row["pseudo_points"] = int(mask.sum())
                if mask.any():
                    d = pred[t, mask] - x[t, ids[mask]]
                    row["pseudo_mde_m"] = float(np.linalg.norm(d, axis=-1).mean())
                    row["pseudo_mse_m2"] = float((d * d).mean())
        rows.append(row)
    return rows, dict(
        manual_initial_points=len(manual_ids),
        initial_manual_nn_distance_m=first_distance.tolist(),
        manual_particle_ids=matched.tolist(),
        manual_track_ids=manual_ids.tolist(),
    )


def aggregate(rows, start, end):
    part = rows[start:end]
    good = bool(part) and all(r["prediction_finite"] for r in part)
    keys = [
        "cd_l1_m",
        "track_l2_m",
        "track_official_empty_zero_m",
        "pseudo_mde_m",
        "pseudo_mse_m2",
    ]
    result = dict(
        start_frame=start,
        end_frame_exclusive=end,
        frames=len(part),
        prediction_complete=good,
        finite_prediction_frames=sum(r["prediction_finite"] for r in part),
        manual_valid_frames=sum(r["manual_points"] > 0 for r in part),
        manual_point_observations=sum(r["manual_points"] for r in part),
    )
    # A partial crashed trajectory never receives a competitive truncated mean.
    for key in keys:
        result[key] = finite_mean([r[key] for r in part]) if good else None
    result["last_track_l2_m"] = part[-1]["track_l2_m"] if good else None
    return result
