"""
Region-of-interest (ROI) level train / val / test split for the SEN12MS-derived
v_2 dataset (agri, barrenland, grassland, urban).

Every tile belongs to one SEN12MS scene, e.g. ``ROIs1970_fall_s1_115``. A scene
can appear in more than one terrain folder, so the split is made on the scene id
across *all* classes: no scene contributes tiles to more than one split. This
prevents the spatial leakage that occurs when overlapping patches of the same
scene land in both training and test sets.

Run once:  python splits.py   ->  writes splits.json
"""
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

CLASSES = ["agri", "barrenland", "grassland", "urban"]
ROI_RE = re.compile(r"(ROIs\d+_[a-z]+)_s1_(\d+)_p(\d+)", re.IGNORECASE)


def scan(root="dataset/v_2"):
    """Return list of dicts {sar, opt, cls, roi} for every complete S1/S2 pair."""
    root = Path(root)
    records = []
    for cls in CLASSES:
        s1_dir, s2_dir = root / cls / "s1", root / cls / "s2"
        for s1 in sorted(s1_dir.glob("*.png")):
            s2 = s2_dir / s1.name.replace("_s1_", "_s2_")
            if not s2.exists():
                continue
            m = ROI_RE.search(s1.name)
            roi = f"{m.group(1)}_{m.group(2)}" if m else s1.stem
            records.append({"sar": str(s1), "opt": str(s2), "cls": cls, "roi": roi})
    return records


def make_split(records, test_frac=0.15, val_frac=0.07, seed=42):
    """Greedy, class-stratified scene split.

    For each class (rarest first) scenes are shuffled and assigned to test until the
    class has ~test_frac of its tiles in test, then to val, rest to train. A scene
    already assigned (because it also occurs in another class) keeps its split.
    """
    rng = np.random.default_rng(seed)
    tiles = defaultdict(lambda: defaultdict(int))  # cls -> roi -> n
    for r in records:
        tiles[r["cls"]][r["roi"]] += 1

    assign = {}
    for cls in CLASSES:
        rois = list(tiles[cls].keys())
        rng.shuffle(rois)
        total = sum(tiles[cls].values())
        got = defaultdict(int)
        for roi in rois:
            if roi in assign:
                got[assign[roi]] += tiles[cls][roi]
        for roi in rois:
            if roi in assign:
                continue
            n = tiles[cls][roi]
            # never put the biggest scene in test/val alone (keeps train large)
            if got["test"] < test_frac * total and got["test"] + n <= 0.30 * total:
                assign[roi] = "test"
            elif got["val"] < val_frac * total and got["val"] + n <= 0.15 * total:
                assign[roi] = "val"
            else:
                assign[roi] = "train"
            got[assign[roi]] += n
    return assign


if __name__ == "__main__":
    recs = scan()
    assign = make_split(recs)
    out = {"train": [], "val": [], "test": []}
    for r in recs:
        out[assign[r["roi"]]].append(r)
    with open("splits.json", "w") as f:
        json.dump({"roi_assignment": assign, **out}, f, indent=1)

    print(f"Total pairs: {len(recs)}  scenes: {len(assign)}")
    for s in ["train", "val", "test"]:
        per = defaultdict(int)
        for r in out[s]:
            per[r["cls"]] += 1
        scenes = sorted({r["roi"] for r in out[s]})
        print(f"{s:5s}: {len(out[s]):6d} tiles  {dict(per)}  scenes={len(scenes)}")
    # leakage check
    sets = {s: {r["roi"] for r in out[s]} for s in out}
    assert not (sets["train"] & sets["test"]) and not (sets["train"] & sets["val"]) and not (sets["val"] & sets["test"])
    print("Leakage check passed: no scene appears in two splits.")
