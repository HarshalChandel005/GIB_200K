import threading
import time

from gib.evaluate import evaluate, load_items
from gib.models.api import _APIBase
from gib.models.simulated import SimulatedVLM
from gib.schema import Query, read_jsonl


class SlowAPI(_APIBase):
    """Answers 'A' after a short delay and records how many calls overlap (no network)."""

    def __init__(self, delay=0.05, **kw):
        super().__init__("slow", "fake-model", **kw)
        self.delay, self.active, self.peak = delay, 0, 0
        self.guard = threading.Lock()

    def _call(self, jpeg, text, max_tokens, logprobs):
        with self.guard:
            self.active += 1
            self.peak = max(self.peak, self.active)
        time.sleep(self.delay)
        with self.guard:
            self.active -= 1
        return {"text": "A", "top": [["A", -0.1], ["B", -2.4]]}


def _q(i, question):
    return Query(qid=f"x{i}|mc", gib_id=f"x{i}", condition="clean", axis="clean", probe="mc",
                 image_key="blank", image_recipe={"method": "blank"}, docs=[], question=question,
                 options=["cat", "dog"])


def test_calls_run_in_parallel_up_to_the_bound(tmp_path):
    api = SlowAPI(max_workers=4, cache_dir=tmp_path)
    qs = [_q(i, f"Question {i}?") for i in range(20)]
    preds = api.run_group(None, qs)
    assert [p.qid for p in preds] == [q.qid for q in qs]  # results keep the query order
    assert all(p.error is None and p.option_probs[0] > 0.8 for p in preds)
    assert 2 <= api.peak <= 4 and api.calls == 20
    again = SlowAPI(max_workers=4, cache_dir=tmp_path)  # same cache file, read from the worker threads
    again.run_group(None, qs)
    assert again.calls == 0


def test_identical_requests_in_flight_are_sent_once(tmp_path):
    api = SlowAPI(delay=0.2, max_workers=8, cache_dir=tmp_path)
    preds = api.run_group(None, [_q(i, "Same question?") for i in range(8)])
    assert api.calls == 1 and len(preds) == 8 and all(p.error is None for p in preds)


def test_parallel_groups_write_the_same_predictions(dataset, cfg, tmp_path):
    items = load_items(dataset["data"], "dev")
    by_id = {it.gib_id: it for it in items}
    out = {}
    for n_par in (1, 4):
        sim = SimulatedVLM("par", by_id, {})
        sim.parallel_groups = n_par  # the simulator is stateless, so it is safe to call from threads
        evaluate(sim, items, cfg, dataset["image_root"], tmp_path / str(n_par), progress=False)
        out[n_par] = sorted(read_jsonl(tmp_path / str(n_par) / "predictions.jsonl"), key=lambda r: r["qid"])
    assert out[1] == out[4] and len(out[1]) > 100
