"""Two independent GGG/PPP streams; random legal GT restart, H5 TBPTT.
No mixed GPP/GGP inputs, no extra burn-in, no validation/test targets.
"""

import numpy as np
import torch
from .stream_state import Stream, tree
from .phystwin import window_info, choose_ids


class RandomStream(Stream):
    def __init__(self, case, seed, state=None, continue_probability=0.5):
        super().__init__(case, seed)
        self.probability = float(continue_probability)
        self.policy_rng = np.random.default_rng(seed + 100003)
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

    def reset(self):
        c = self.case
        t = int(self.rng.choice(c.windows_for(5)))
        _, good, labels = window_info(c.x, c.vis, c.motion, t, c.b, 5)
        pool = np.flatnonzero(good)
        n = min(c.plan["particles"], len(pool))
        if n < c.plan["min_particles"]:
            raise ValueError("No valid real H3 restart")
        ids = choose_ids(pool, labels, n, self.rng)
        seq = (
            np.concatenate((c.x[t - 2 : t + 1, ids], c.ctrl[t - 2 : t + 1]), 1)
            .transpose(1, 0, 2)
            .copy()
        )
        self.state = dict(
            ids=ids,
            t=t,
            start_t=t,
            sequence=torch.from_numpy(seq)[None],
            memory=None,
            chain=np.ones(n, bool),
            blocks=0,
        )

    def batch(self):
        if self.state is None:
            reason = "initial"
        elif self.state["t"] + 5 >= self.case.b:
            reason = "boundary"
        elif self.policy_rng.random() >= self.probability:
            reason = "random"
        else:
            reason = None
        if reason is not None:
            if self.state is not None:
                self.completed += 1
            self.reset()
            self.restart_count += 1
        else:
            self.continue_count += 1
        b, m, stats = super().batch()
        if self.pending[0] != 5:
            raise AssertionError("Random restart uses exactly H5")
        stats.update(
            stream_age=self.state["t"] - self.state["start_t"],
            start_t=self.state["start_t"],
            input_kind="GGG" if reason else "PPP",
            restart_reason=reason,
            restarts=self.restart_count,
            continues=self.continue_count,
        )
        return b, m, stats

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


class Streams:
    def __init__(self, case, seed, state=None, continue_probability=0.5):
        if state and state["kind"] != "random_restart_two_streams":
            raise ValueError("Wrong stream checkpoint")
        self.streams = [
            RandomStream(
                case,
                seed + q * 200003,
                state["streams"][q] if state else None,
                continue_probability,
            )
            for q in range(2)
        ]

    def snapshot(self):
        return dict(
            kind="random_restart_two_streams",
            streams=[s.snapshot() for s in self.streams],
        )
