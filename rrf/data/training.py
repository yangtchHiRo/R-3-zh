"""Two independent serial streams; each update averages their H5 objectives."""

from pathlib import Path
import numpy as np
from ..io import read, digest
from .adapters import metadata
from ..models.regions import reference_pack


class TrainingData:
    def __init__(self, root, config, device):
        root = Path(root)
        self.config = config
        self.manifest = read(root / "manifest.json")
        self.provenance = dict(manifest=digest(self.manifest))
        self.model_extras = {}
        if config["dataset"] == "phystwin":
            from .phystwin import Case

            self.cases = [
                Case(self.manifest, n) for n in self.manifest["group"]["train"]
            ]
            if not self.cases or any(not c.windows_for(5) for c in self.cases):
                raise ValueError("Every training scene needs a valid H3+H5 window")
            for c in self.cases:
                good = c.vis[0] & np.isfinite(c.x[0]).all(-1)
                c.patch_reference = reference_pack(
                    c.x[0, good], config["model"]["regions"]
                )
            dt = self.cases[0].entry["dt"]
            if any(abs(c.entry["dt"] - dt) > 1e-9 for c in self.cases):
                raise ValueError("Mixed frame rates")
            self.metadata = metadata(
                self.manifest["metadata"], dt, "source", "phystwin"
            )
        else:
            from .pgnd import NativeWindowDataset

            paths = read(root / "paths.json")
            self.case = NativeWindowDataset(
                root, config["data"]["frame_cache_size"], paths["native_state"]
            )
            self.case.patch_references = {}
            self.case.patches = config["model"]["regions"]
            self.metadata = metadata(self.case.meta, 0.1, self.manifest["task"], "pgnd")
            self.model_extras = dict(pgnd_config=read(root / "pgnd_config.json"))
        self.load_state_dict(None)

    def load_state_dict(self, state):
        cfg = self.config
        seed = cfg["seed"] + 400009
        prob = cfg["data"]["continue_probability"]
        if cfg["dataset"] == "pgnd":
            from .pgnd_restart import PGNDStreams

            self.bank = PGNDStreams(self.case, seed, state, prob)
        elif len(self.cases) == 1:
            from .restart import Streams

            self.bank = Streams(self.cases[0], seed, state, prob)
        else:
            from .category_streams import Streams

            self.bank = Streams(self.cases, seed, state, prob)
        self.streams = self.bank.streams

    def state_dict(self):
        return self.bank.snapshot()


def build_training_data(root, config, device):
    return TrainingData(root, config, device)
