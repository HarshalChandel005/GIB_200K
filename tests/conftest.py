from pathlib import Path

import pytest

from gib.build.coco_vg import build
from gib.build.fixtures import make_fixture
from gib.build.subset import sample_subset, write_subset
from gib.config import load_config


@pytest.fixture(scope="session")
def cfg():
    return load_config(overrides={
        "build": {"per_type": {"object": 16, "attribute": 16, "relation": 16}},
        "subsets": {"eval": 48, "dev": 8},
        "analysis": {"bootstrap_iters": 200},
        "probes": {"caption_frac": 0.5},
    })


@pytest.fixture(scope="session")
def dataset(tmp_path_factory, cfg):
    root = tmp_path_factory.mktemp("gib")
    roots = make_fixture(root / "fixture", n_images=60, seed=0)
    items = build(root / "data", cfg, roots["coco_root"], splits=["val2017", "train2017"], vg_root=roots["vg_root"])
    for name, n in cfg["subsets"].items():
        write_subset(root / "data" / "subsets" / f"{name}.txt", sample_subset(items, n, cfg["seed"]))
    return {"root": root, "items": items, "image_root": roots["coco_root"], "vg_root": roots["vg_root"],
            "data": root / "data"}
