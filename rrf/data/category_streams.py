"""Two serial streams, equal-scene choice on each real-H3 random restart.
Continuations retain the same scene, particle IDs and predicted last three frames.
"""

import numpy as np
from .restart import RandomStream


class CategoryStream(RandomStream):
    def __init__(self, cases, seed, state=None, continue_probability=0.5):
        self.cases = cases
        self.scene_rng = np.random.default_rng(seed + 600011)
        index = state["case_index"] if state else 0
        if state:
            self.scene_rng.bit_generator.state = state["scene_rng"]
        self.case_index = index
        super().__init__(cases[index], seed, state, continue_probability)

    def reset(self):
        self.case_index = int(self.scene_rng.integers(len(self.cases)))
        self.case = self.cases[self.case_index]
        super().reset()

    def batch(self):
        b, m, s = super().batch()
        s["scene"] = self.case.entry["id"]
        return b, m, s

    def snapshot(self):
        return dict(
            super().snapshot(),
            case_index=self.case_index,
            scene_rng=self.scene_rng.bit_generator.state,
        )


class Streams:
    def __init__(self, cases, seed, state=None, continue_probability=0.5):
        if state and state["kind"] != "category_random_restart":
            raise ValueError("Wrong stream checkpoint")
        self.streams = [
            CategoryStream(
                cases,
                seed + q * 200003,
                state["streams"][q] if state else None,
                continue_probability,
            )
            for q in range(2)
        ]

    def snapshot(self):
        return dict(
            kind="category_random_restart", streams=[s.snapshot() for s in self.streams]
        )
