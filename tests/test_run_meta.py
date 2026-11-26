import json

import pytest

from gib.evaluate import evaluate, load_items
from gib.models.simulated import SimulatedVLM


class StopsOnThirdGroup(SimulatedVLM):
    """Stops the run like Ctrl-C on its third image group, after reading meta.json mid-run."""

    def __init__(self, *a, meta_path, **kw):
        super().__init__(*a, **kw)
        self.groups, self.meta_path, self.seen = 0, meta_path, None

    def run_group(self, image, queries):
        self.groups += 1
        if self.groups == 2:
            self.seen = json.loads(self.meta_path.read_text(encoding="utf-8"))
        if self.groups == 3:
            raise KeyboardInterrupt
        return super().run_group(image, queries)

    def meta(self):
        return {**super().meta(), "groups_answered": self.groups}  # stands in for self-check results


def test_meta_records_interrupted_and_resumed_runs(dataset, cfg, tmp_path):
    items = load_items(dataset["data"], "dev")
    by_id = {it.gib_id: it for it in items}
    run_dir = tmp_path / "run"
    sim = StopsOnThirdGroup("m", by_id, {}, meta_path=run_dir / "meta.json")
    with pytest.raises(KeyboardInterrupt):
        evaluate(sim, items, cfg, dataset["image_root"], run_dir, progress=False)
    mid = sim.seen  # written after the first group, so the adapter state from that group is recorded
    assert mid["status"] == "running" and mid["adapter"]["groups_answered"] == 1 and mid["new_predictions"] > 0
    stopped = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    assert stopped["status"] == "interrupted" and stopped["error"].startswith("KeyboardInterrupt")
    assert stopped["new_predictions"] > mid["new_predictions"] and stopped["adapter"]["groups_answered"] == 3

    done = evaluate(SimulatedVLM("m", by_id, {}), items, cfg, dataset["image_root"], run_dir, progress=False)
    assert done["status"] == "finished"
    assert [h["status"] for h in done["history"]] == ["interrupted", "finished"]
    assert stopped["new_predictions"] + done["new_predictions"] == done["n_queries"]
    assert not (run_dir / "meta.json.tmp").exists()
