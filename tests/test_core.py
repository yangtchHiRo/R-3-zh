"""CPU checks for causal controls, feedback gradients and stream continuation."""

import copy
import unittest
import numpy as np
import torch
from rrf.models.network import RRF
from rrf.models.regions import reference_pack
from rrf.data.adapters import metadata, adapt
from rrf.data.stream_state import masked_loss
from rrf.data.pgnd_restart import PGNDRandomStream


def fixture():
    torch.manual_seed(10)
    meta = metadata(
        dict(
            vel_mean=[0] * 3,
            vel_std=[0.01] * 3,
            acc_std=[0.001] * 3,
            connectivity_radius=0.1,
        ),
        1 / 30,
        "cloth",
        "phystwin",
    )
    model = RRF(meta, dict(candidates=4))
    n, c, h = 12, 2, 5
    x = torch.rand(n, 3) * 0.1
    seq = torch.cat(
        (
            x[:, None].expand(-1, 3, -1) + torch.arange(3)[None, :, None] * 0.0001,
            torch.rand(c, 3, 3) * 0.05,
        ),
        0,
    )[None]
    a = torch.zeros(1, h, c, 15)
    a[..., 6] = 1
    a[..., :3] = seq[:, n:, -1][:, None]
    f = a.clone()
    f[..., :3] += 0.0003
    raw = dict(
        position_sequence=seq,
        actions=a,
        next_grippers=f,
        initial_ref=x[None],
        reference_valid=torch.ones(1, n, dtype=torch.bool),
        initial_ctrl_ref=seq[:, n:, 0],
        **{
            k: torch.from_numpy(v)[None]
            for k, v in reference_pack(x.numpy(), 6).items()
        }
    )
    return model, adapt(raw, meta)


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_future_controls_cannot_change_first_prediction(self):
        model, b = fixture()
        model.eval()
        c = copy.deepcopy(b)
        c["ctrl_path"][:, 2:, :, :3] += 1
        with torch.no_grad():
            a = model(b)
            z = model(c)
        torch.testing.assert_close(
            a["pred"][:, 0], z["pred"][:, 0], rtol=1e-5, atol=1e-7
        )

    def test_feedback_and_masked_loss(self):
        model, b = fixture()
        model.train()
        b["reference_valid"][:, 0] = False
        b["initial_ref"][:, 0] = 0
        out = model(b, diagnostics=True)
        self.assertFalse(out["draft"].requires_grad)
        target = torch.zeros_like(out["pred"])
        valid = torch.ones(target.shape[:-1], dtype=torch.bool)
        loss = 100 * masked_loss(out["pred"], target, valid) + model.auxiliary_loss(
            out, target, valid, b["initial_ref"]
        )
        loss.backward()
        for name in ["response_out", "response_region"]:
            grad = getattr(model, name)[-1].weight.grad
            self.assertIsNotNone(grad)
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(float(grad.abs().sum()), 0)
        valid.zero_()
        self.assertEqual(float(masked_loss(out["pred"], target, valid)), 0.0)
        self.assertEqual(
            float(model.auxiliary_loss(out, target, valid, b["initial_ref"])), 0.0
        )

    def test_pgnd_continuation_and_resume(self):
        # Minimal native-state source: real PGND adapter uses the same interface.
        class Source:
            meta = {"n_particles": 10}
            entries = [{"episode": 0}]
            frames = [list(range(40))]
            cumulative = [33]
            patches = 6

            def __init__(self):
                self.patch_references = {}

            def __len__(self):
                return 33

            def _load(self, t):
                x = (
                    torch.arange(36, dtype=torch.float32).reshape(12, 3) * 0.001
                    + t * 0.0001
                )
                g = torch.zeros(1, 15)
                g[:, 6] = 1
                g[:, :3] = 0.1 + t * 0.001
                return dict(x=x, grippers=g)

        s = PGNDRandomStream(Source(), 6, continue_probability=1.0)
        b, _, stats = s.batch()
        # Start near beginning so a continuation fits.
        while s.state["t"] > 20:
            s.reset()
            b, _, stats = s.batch()
        pred = b["gt_x"] + 0.002
        ids = s.state["ids"].clone()
        old_t = s.state["t"]
        s.advance({"pred": pred})
        state = copy.deepcopy(s.snapshot())
        b1, _, info = s.batch()
        r = PGNDRandomStream(Source(), 6, state, 1.0)
        b2, _, info2 = r.batch()
        self.assertEqual(info["input_kind"], "PPP")
        self.assertEqual(s.state["t"], old_t + 5)
        torch.testing.assert_close(
            b1["position_sequence"][0, : len(ids)], pred[0, -3:].transpose(0, 1)
        )
        torch.testing.assert_close(s.state["ids"], ids)
        for key in b1:
            torch.testing.assert_close(b1[key], b2[key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
