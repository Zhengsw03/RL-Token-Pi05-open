"""Tests for the critical-phase classifier pipeline: recorder log format,
toggle-to-label expansion, network shape, and detector hysteresis."""
import importlib.util
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

LEROBOT_SRC = Path(__file__).resolve().parents[1] / "lerobot" / "src"
if str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))


def _load_module(name: str, script: str):
    path = Path(__file__).parents[1] / "scripts" / script
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


from lerobot.policies.pi05_rlt.critical import CriticalPhaseClassifier, CriticalPhaseDetector  # noqa: E402


class CriticalPhaseRecorderTest(unittest.TestCase):
    def test_recorder_log_expands_toggles_into_labels(self):
        stage2 = _load_module("train_rlt_stage2_pi05", "train_rlt_stage2_pi05.py")
        train = _load_module("train_critical_classifier", "train_critical_classifier.py")

        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "critical.log"
            recorder = stage2.CriticalPhaseRecorder(log)
            # Episode 0: critical from step 100 to 200.
            for step in (50, 100, 150, 250, 300):
                recorder.record_sample(0, step, np.zeros(4, dtype=np.float32), np.zeros(6, dtype=np.float32))
            recorder.record_toggle(0, 100, True)
            recorder.record_toggle(0, 200, False)
            # Episode 1: critical from step 10, never turned off (open interval).
            recorder.record_sample(1, 5, np.zeros(4, dtype=np.float32), np.zeros(6, dtype=np.float32))
            recorder.record_sample(1, 15, np.zeros(4, dtype=np.float32), np.zeros(6, dtype=np.float32))
            recorder.record_toggle(1, 10, True)
            recorder.close()

            features, _ = train.load_record_log(str(log))
            labels = features["labels"]
            self.assertEqual(list(labels), [0, 1, 1, 0, 0, 0, 1])

    def test_recorder_samples_carry_zrl_and_proprio(self):
        stage2 = _load_module("train_rlt_stage2_pi05", "train_rlt_stage2_pi05.py")
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "critical.log"
            recorder = stage2.CriticalPhaseRecorder(log)
            recorder.record_sample(0, 3, np.arange(4, dtype=np.float32), np.arange(6, dtype=np.float32))
            recorder.close()
            with open(log, "rb") as file:
                record = pickle.load(file)
            self.assertEqual(record["kind"], "sample")
            self.assertEqual(record["episode"], 0)
            self.assertEqual(record["step"], 3)
            np.testing.assert_array_equal(record["z_rl"], np.arange(4, dtype=np.float32))
            np.testing.assert_array_equal(record["proprio"], np.arange(6, dtype=np.float32))


class CriticalPhaseClassifierTest(unittest.TestCase):
    def test_classifier_output_shape(self):
        model = CriticalPhaseClassifier(z_dim=8, proprio_dim=6, hidden_dim=16)
        z = torch.randn(4, 8)
        p = torch.randn(4, 6)
        logits = model(z, p)
        self.assertEqual(tuple(logits.shape), (4, 2))

    def test_detector_loads_checkpoint_and_applies_hysteresis(self):
        torch.manual_seed(0)  # classifier training below is seed-sensitive
        model = CriticalPhaseClassifier(z_dim=4, proprio_dim=2, hidden_dim=8)
        # Train a trivially separable classifier: z[:, 0] > 0 -> critical.
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        z_train = torch.cat([torch.randn(64, 4) + 1.0, torch.randn(64, 4) - 1.0])
        p_train = torch.randn(128, 2)
        labels = torch.cat([torch.ones(64, dtype=torch.long), torch.zeros(64, dtype=torch.long)])
        for _ in range(200):
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(z_train, p_train), labels)
            loss.backward()
            optimizer.step()

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "cls.pt"
            torch.save({
                "classifier_state_dict": model.state_dict(),
                "proprio_mean": p_train.mean(0).numpy().astype(np.float32),
                "proprio_std": p_train.std(0).numpy().astype(np.float32) + 1e-6,
                "z_dim": 4,
                "proprio_dim": 2,
                "hidden_dim": 8,
                "num_layers": 2,
            }, checkpoint_path)

            detector = CriticalPhaseDetector(
                str(checkpoint_path), "cpu",
                threshold_on=0.7, threshold_off=0.3, smooth_steps=5,
            )
            # Positive z -> on after a few updates; negative z -> off.
            state = None
            for _ in range(10):
                candidate = detector.update(np.array([2.0, 0.0, 0.0, 0.0], dtype=np.float32),
                                            np.array([0.0, 0.0], dtype=np.float32))
                if candidate is not None:
                    state = candidate
            self.assertTrue(state)
            detector.reset()
            state = None
            for _ in range(10):
                candidate = detector.update(np.array([-2.0, 0.0, 0.0, 0.0], dtype=np.float32),
                                            np.array([0.0, 0.0], dtype=np.float32))
                if candidate is not None:
                    state = candidate
            self.assertFalse(state)

    def test_detector_rejects_wrong_z_dim(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "cls.pt"
            torch.save({
                "classifier_state_dict": CriticalPhaseClassifier(
                    z_dim=4, proprio_dim=2, hidden_dim=8).state_dict(),
                "proprio_mean": np.zeros(2, dtype=np.float32),
                "proprio_std": np.ones(2, dtype=np.float32),
                "z_dim": 4,
                "proprio_dim": 2,
                "hidden_dim": 8,
                "num_layers": 2,
            }, checkpoint_path)
            detector = CriticalPhaseDetector(str(checkpoint_path), "cpu")
            with self.assertRaises(ValueError):
                detector.update(np.zeros(8, dtype=np.float32), np.zeros(2, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()


class CriticalTransitionSavingTest(unittest.TestCase):
    def test_save_critical_transition_writes_csv_and_frames(self):
        import tempfile
        from pathlib import Path
        import numpy as np

        stage2 = _load_module("train_rlt_stage2_pi05", "train_rlt_stage2_pi05.py")
        with tempfile.TemporaryDirectory() as directory:
            frames_dir = Path(directory) / "frames"
            obs = {
                "top": np.zeros((64, 64, 3), dtype=np.uint8),
                "wrist": np.full((64, 64, 3), 255, dtype=np.uint8),
            }
            stage2.save_critical_transition(
                frames_dir, ["top", "wrist"],
                episode=3, step=123, direction="on", source="auto", prob=0.87, obs=obs,
            )
            stage2.save_critical_transition(
                frames_dir, ["top", "wrist"],
                episode=3, step=456, direction="off", source="manual", prob=None, obs=obs,
            )
            # CSV rows (header + 2 events)
            lines = (frames_dir / "critical_events.csv").read_text().strip().splitlines()
            self.assertEqual(len(lines), 3)
            self.assertEqual(lines[0], "episode,step,direction,source,prob,wall_time")
            self.assertTrue(lines[1].startswith("3,123,on,auto,0.870,"))
            self.assertTrue(lines[2].startswith("3,456,off,manual,,"))
            # Frame files
            jpgs = sorted(p.name for p in frames_dir.glob("*.jpg"))
            self.assertIn("ep003_step00123_on_auto_top.jpg", jpgs)
            self.assertIn("ep003_step00123_on_auto_wrist.jpg", jpgs)
            self.assertIn("ep003_step00456_off_manual_wrist.jpg", jpgs)

    def test_save_critical_transition_tolerates_missing_obs(self):
        import tempfile
        from pathlib import Path

        stage2 = _load_module("train_rlt_stage2_pi05", "train_rlt_stage2_pi05.py")
        with tempfile.TemporaryDirectory() as directory:
            frames_dir = Path(directory) / "frames"
            # obs=None (e.g. HIL without a captured observation) must not crash,
            # and still writes the CSV event.
            stage2.save_critical_transition(
                frames_dir, ["top", "wrist"],
                episode=1, step=2, direction="on", source="manual", prob=None, obs=None,
            )
            lines = (frames_dir / "critical_events.csv").read_text().strip().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(len(list(frames_dir.glob("*.jpg"))), 0)


class CriticalTransitionColorTest(unittest.TestCase):
    def test_saved_frames_convert_rgb_to_bgr(self):
        """Camera frames are RGB; cv2.imwrite needs BGR, so a pure-blue RGB
        frame must come back blue (BGR high B, low R) after saving."""
        import tempfile
        from pathlib import Path

        import cv2

        stage2 = _load_module("train_rlt_stage2_pi05", "train_rlt_stage2_pi05.py")
        with tempfile.TemporaryDirectory() as directory:
            frames_dir = Path(directory) / "frames"
            rgb_blue = np.zeros((32, 32, 3), dtype=np.uint8)   # RGB: R=0 G=0 B=255
            rgb_blue[:, :, 2] = 255
            rgb_yellow = np.zeros((32, 32, 3), dtype=np.uint8)  # RGB: R=255 G=255 B=0
            rgb_yellow[:, :, 0] = 255
            rgb_yellow[:, :, 1] = 255
            obs = {"top": rgb_blue, "wrist": rgb_yellow}
            stage2.save_critical_transition(
                frames_dir, ["top", "wrist"],
                episode=4, step=7, direction="on", source="auto", prob=0.9, obs=obs,
            )
            blue_back = cv2.imread(str(frames_dir / "ep004_step00007_on_auto_top.jpg"))
            yellow_back = cv2.imread(str(frames_dir / "ep004_step00007_on_auto_wrist.jpg"))
            # BGR stored: blue -> (255, 0, 0), yellow -> (0, 255, 255).
            # JPEG is lossy, so allow small deviations.
            np.testing.assert_allclose(blue_back[0, 0], [255, 0, 0], atol=4)
            np.testing.assert_allclose(yellow_back[0, 0], [0, 255, 255], atol=4)
            # The swapped layout (would be [0, 0, 255] for blue) is far away.
            self.assertGreater(blue_back[0, 0][0], blue_back[0, 0][2] + 200)
