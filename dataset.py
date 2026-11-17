import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

SPLIT_FILE = "splits.json"


def find_dataset_root(candidate_name='dataset'):
    cwd = Path.cwd()
    for p in [cwd / candidate_name, cwd.parent / candidate_name]:
        if p.exists():
            return p
    return cwd / candidate_name


def load_split(split, split_file=SPLIT_FILE):
    if not Path(split_file).exists():
        raise FileNotFoundError(f"'{split_file}' not found. Run `python splits.py` first.")
    with open(split_file) as f:
        return json.load(f)[split]


def subsample_balanced(records, per_class, seed=0):
    """Deterministic class-balanced subset (used for val/test to bound runtime)."""
    rng = random.Random(seed)
    by_cls = {}
    for r in records:
        by_cls.setdefault(r["cls"], []).append(r)
    out = []
    for cls in sorted(by_cls):
        items = sorted(by_cls[cls], key=lambda r: r["sar"])
        rng.shuffle(items)
        out.extend(items[:per_class])
    return out


def read_sar(path):
    """8-bit Sentinel-1 tile -> float32 [H,W] in [0,1].

    The v_2 PNG tiles are already quantised backscatter; we only rescale to [0,1].
    (The former "dB calibration" mapped [0,1]->[-25,0]dB->[0,1] and was an identity.)
    """
    return np.asarray(Image.open(path).convert('L'), dtype=np.float32) / 255.0


def read_opt(path):
    return np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 255.0


class PairedSARDataset(Dataset):
    """Paired S1 -> S2 tiles from splits.json.

    Training: random crop + horizontal/vertical flips + 90-degree rotations
    (identical transform applied to SAR and optical).
    """
    def __init__(self, split="train", crop=None, augment=False, per_class=None, seed=0):
        recs = load_split(split)
        if per_class is not None:
            recs = subsample_balanced(recs, per_class, seed)
        self.records = recs
        self.crop = crop
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[int(idx)]
        sar = read_sar(r["sar"])
        opt = read_opt(r["opt"])
        if self.crop:
            h, w = sar.shape
            y = random.randint(0, h - self.crop)
            x = random.randint(0, w - self.crop)
            sar = sar[y:y + self.crop, x:x + self.crop]
            opt = opt[y:y + self.crop, x:x + self.crop]
        if self.augment:
            k = random.randint(0, 3)
            sar, opt = np.rot90(sar, k), np.rot90(opt, k)
            if random.random() < 0.5:
                sar, opt = sar[:, ::-1], opt[:, ::-1]
        sar_t = torch.from_numpy(np.ascontiguousarray(sar))[None]
        opt_t = torch.from_numpy(np.ascontiguousarray(opt)).permute(2, 0, 1)
        return {"sar": sar_t, "optical": opt_t, "cls": r["cls"], "filename": Path(r["sar"]).name}
