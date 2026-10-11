"""CPU tests of train.fit: records, early stopping and reloadable weights."""

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules import configure_trainable, load_weights  # noqa: E402
from test_modules import sd_schedule, tiny_embeddings, tiny_unet  # noqa: E402
from train import EncodedSplit, TrainConfig, fit  # noqa: E402


class FitTest(unittest.TestCase):
    def test_fit_writes_records_and_weights_that_reload(self):
        unet = tiny_unet()
        parameters = configure_trainable(unet, "unet")
        generator = torch.Generator().manual_seed(4)
        split = EncodedSplit(
            torch.randn((8, 4, 8, 8), generator=generator),
            torch.full((8, 4, 8, 8), 0.1),
            torch.tensor([0, 1] * 4),
        )
        config = TrainConfig(
            epochs=2,
            batch_size=4,
            lr=1e-3,
            warmup_steps=2,
            log_every=1,
            save_epochs=(1,),
        )
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            summary = fit(
                config,
                unet,
                parameters,
                sd_schedule(),
                tiny_embeddings(),
                split,
                split,
                0.18215,
                output,
                output / "weights",
            )
            self.assertEqual(summary["steps"], 4)
            self.assertEqual(summary["stop_reason"], "completed")
            with (output / "epochs.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([r["epoch"] for r in rows], ["1", "2"])
            self.assertTrue((output / "curves.png").stat().st_size > 0)
            self.assertEqual(
                len(summary["weights"]), 2
            )  # epoch 1 by request, 2 as last
            # A differently initialised UNet must end up holding the saved weights.
            fresh = tiny_unet(seed=7)
            self.assertEqual(load_weights(fresh, Path(summary["weights"][-1])), "unet")
            untouched = tiny_unet(seed=7)
            self.assertTrue(
                any(
                    not torch.equal(a, b)
                    for a, b in zip(fresh.parameters(), untouched.parameters())
                )
            )

    def test_max_steps_stops_inside_an_epoch_and_still_saves(self):
        unet = tiny_unet()
        parameters = configure_trainable(unet, "lora", lora_rank=2)
        split = EncodedSplit(
            torch.randn(8, 4, 8, 8), torch.zeros(8, 4, 8, 8), torch.tensor([0, 1] * 4)
        )
        config = TrainConfig(
            epochs=3, batch_size=2, max_steps=3, log_every=50, save_epochs=()
        )
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            summary = fit(
                config,
                unet,
                parameters,
                sd_schedule(),
                tiny_embeddings(),
                split,
                split,
                0.18215,
                output,
                output / "weights",
                weight_prefix="lora",
            )
            self.assertEqual(
                (summary["steps"], summary["stop_reason"]), (3, "max_steps")
            )
            self.assertEqual(len(summary["weights"]), 1)
            fresh = tiny_unet()
            self.assertEqual(load_weights(fresh, Path(summary["weights"][0])), "lora")


if __name__ == "__main__":
    unittest.main()
