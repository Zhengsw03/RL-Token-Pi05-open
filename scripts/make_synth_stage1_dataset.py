"""Build a tiny synthetic v3 dataset carrying stage1 vlm_embeddings features."""
import os, sys, numpy as np, shutil
from pathlib import Path
_repo = Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME', str(_repo / '.hf_cache'))
os.environ.setdefault('HF_DATASETS_CACHE', str(_repo / '.hf_cache' / 'datasets'))
LEROBOT_SRC = _repo / 'lerobot' / 'src'
sys.path.insert(0, str(LEROBOT_SRC))
from lerobot.datasets.lerobot_dataset import LeRobotDataset

root = Path(os.environ.get('RLT_SYNTH_ROOT', _repo / 'outputs/synth_rlt_stage1'))
shutil.rmtree(root, ignore_errors=True)
features = {
    "observation.state": {"dtype": "float32", "shape": (6,), "names": ["j1","j2","j3","j4","j5","j6"]},
    "action": {"dtype": "float32", "shape": (6,), "names": ["j1","j2","j3","j4","j5","j6"]},
    "vlm_embeddings": {"dtype": "float32", "shape": (16, 32)},
    "prefix_mask": {"dtype": "bool", "shape": (16,)},
}
ds = LeRobotDataset.create('local/synth_rlt_stage1', fps=30, features=features, root=root,
                           robot_type='so_follower', use_videos=False)
rng = np.random.default_rng(7)
# 2 episodes x 20 frames, embeddings random noise (trainable pattern: first dim correlates with state)
for ep in range(2):
    for f in range(20):
        state = rng.standard_normal(6).astype(np.float32)
        emb = rng.standard_normal((16, 32)).astype(np.float32)
        emb[:, 0] += 2.0 * state[0]  # weak signal for overfit sanity
        ds.add_frame({
            "observation.state": state,
            "action": rng.standard_normal(6).astype(np.float32),
            "vlm_embeddings": emb,
            "prefix_mask": np.ones(16, dtype=np.bool_),
            "task": "Insert the straw into the cup",
        })
    ds.save_episode()
ds.finalize()
print('[OK] synth dataset at', root, '| frames:', len(LeRobotDataset('local/synth_rlt_stage1', root=root)))
