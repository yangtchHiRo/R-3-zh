"""TRAIN-only PGND random-restart streams; episode-local identities and H3/H5.
A refresh samples a legal training window uniformly (thus episodes proportional
 to window counts). A continuation keeps episode, particle IDs, reference and
 the previous final predicted H3; no prefix regeneration or target history use.
"""

import bisect
import numpy as np
import torch
from .stream_state import tree
from ..models.regions import reference_pack
from .pgnd import encode_window


class PGNDRandomStream:
    def __init__(self, case, seed, state=None, continue_probability=0.5):
        self.case = case
        self.probability = float(continue_probability)
        self.rng = np.random.default_rng(seed)
        self.policy_rng = np.random.default_rng(seed + 100003)
        self.state = None
        self.completed = 0
        self.restart_count = 0
        self.continue_count = 0
        if state:
            if state["continue_probability"] != self.probability:
                raise ValueError("Restart probability changed")
            self.rng.bit_generator.state = state["rng"]
            self.policy_rng.bit_generator.state = state["policy_rng"]
            self.state = state["stream"]
            self.completed = state["completed"]
            self.restart_count = state["restart_count"]
            self.continue_count = state["continue_count"]
            if self.state is not None:
                ep = self.state["ep"]
                if case.entries[ep]["episode"] != self.state["episode"]:
                    raise ValueError("PGND train episode order changed")

    def reset(self):
        c = self.case
        i = int(self.rng.integers(len(c)))
        ep = bisect.bisect_right(c.cumulative, i)
        start = i - (c.cumulative[ep - 1] if ep else 0)
        first = c._load(c.frames[ep][0])
        ss = [c._load(f) for f in c.frames[ep][start : start + 3]]
        n = int(c.meta["n_particles"])
        ids = torch.randperm(len(ss[0]["x"]))[:n]
        if len(ids) != n:
            raise ValueError("Too few PGND particles for restart")
        if any(len(s["x"]) != len(first["x"]) for s in ss):
            raise ValueError("PGND within-episode particle indexing changed")
        seq = (
            torch.cat(
                (
                    torch.stack([s["x"][ids] for s in ss]),
                    torch.stack([s["grippers"][:, :3] for s in ss]),
                ),
                1,
            )
            .permute(1, 0, 2)
            .contiguous()[None]
        )
        if not torch.isfinite(seq).all():
            raise FloatingPointError("Invalid PGND real H3")
        self.state = dict(
            ep=ep,
            episode=c.entries[ep]["episode"],
            ids=ids,
            t=start + 2,
            start_t=start + 2,
            sequence=seq,
            memory=None,
            blocks=0,
        )

    def batch(self):
        s = self.state
        c = self.case
        if s is None:
            reason = "initial"
        elif s["t"] + 5 >= len(c.frames[s["ep"]]):
            reason = "boundary"
        elif self.policy_rng.random() >= self.probability:
            reason = "random"
        else:
            reason = None
        if reason is not None:
            if s is not None:
                self.completed += 1
            self.reset()
            self.restart_count += 1
        else:
            self.continue_count += 1
        s = self.state
        ep = s["ep"]
        t = s["t"]
        ids = s["ids"]
        ss = [c._load(f) for f in c.frames[ep][t - 2 : t + 6]]
        if len(ss) != 8:
            raise ValueError("Expected full H3+H5 PGND window")
        first = c._load(c.frames[ep][0])
        if any(len(q["x"]) != len(first["x"]) for q in ss):
            raise ValueError("PGND within-episode identity count changed")
        row = encode_window(
            torch.stack([q["x"] for q in ss]),
            torch.stack([q["grippers"] for q in ss]),
            ids,
        )
        row.update(
            particle_ids=ids,
            initial_ref=first["x"][ids].float(),
            initial_ctrl_ref=first["grippers"][:, :3].float(),
            reference_valid=torch.ones(len(ids), dtype=torch.bool),
            valid=torch.ones(5, len(ids), dtype=torch.bool),
            history_seed_observed=torch.tensor(reason is not None),
        )
        if not torch.isfinite(row["gt_x"]).all():
            raise FloatingPointError("Nonfinite native PGND target")
        if ep not in c.patch_references:
            c.patch_references[ep] = reference_pack(first["x"].numpy(), c.patches)
        row.update(
            {k: torch.from_numpy(v.copy()) for k, v in c.patch_references[ep].items()}
        )
        b = {k: v[None] for k, v in row.items()}
        # The only particle history consumed by the model is this GGG/PPP state.
        b["position_sequence"] = s["sequence"]
        self.pending_controls = row["next_grippers"][..., :3].detach().cpu()
        return (
            b,
            None,
            dict(
                episode=s["episode"],
                stream_t=t,
                stream_age=t - s["start_t"],
                start_t=s["start_t"],
                stream_blocks=s["blocks"],
                completed_streams=self.completed,
                stream_horizon=5,
                stream_valid_per_frame=[len(ids)] * 5,
                input_kind="GGG" if reason is not None else "PPP",
                restart_reason=reason,
                restarts=self.restart_count,
                continues=self.continue_count,
            ),
        )

    def advance(self, result):
        s = self.state
        pred = result["pred"].detach().cpu()
        seq = s["sequence"].cpu()
        if pred.shape != (1, 5, len(s["ids"]), 3) or not torch.isfinite(pred).all():
            raise FloatingPointError("Invalid PGND final rollout")
        for q in range(5):
            nxt = torch.cat((pred[:, q], self.pending_controls[q : q + 1]), 1)
            seq = torch.cat((seq[:, :, 1:], nxt[:, :, None]), 2)
        s.update(sequence=seq, memory=None, t=s["t"] + 5, blocks=s["blocks"] + 1)

    def snapshot(self):
        return dict(
            rng=self.rng.bit_generator.state,
            policy_rng=self.policy_rng.bit_generator.state,
            stream=tree(self.state, "cpu"),
            completed=self.completed,
            continue_probability=self.probability,
            restart_count=self.restart_count,
            continue_count=self.continue_count,
        )


class PGNDStreams:
    def __init__(self, case, seed, state=None, continue_probability=0.5):
        if state and state["kind"] != "pgnd_random_restart_two_streams":
            raise ValueError("Wrong PGND stream state")
        self.streams = [
            PGNDRandomStream(
                case,
                seed + q * 200003,
                state["streams"][q] if state else None,
                continue_probability,
            )
            for q in range(2)
        ]

    def snapshot(self):
        return dict(
            kind="pgnd_random_restart_two_streams",
            streams=[s.snapshot() for s in self.streams],
        )
