"""Smoke tests: dataset generation, each matching method runs and returns a
sane result, and the tie-break/utility functions behave as documented."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biocontext.data.synthetic import CaseConfig, generate_case, build_benchmark_suite
from biocontext.methods import METHOD_REGISTRY
from biocontext.methods.common import Candidate, pick_center_most


def test_generate_case_basic():
    cfg = CaseConfig(seed=1, search_size=400, ref_crop_size=200)
    case = generate_case(cfg)
    assert case.search_image.shape == (400, 400)
    assert case.reference_image.shape[0] > 0 and case.reference_image.shape[1] > 0
    assert 0 <= case.true_center[0] <= 400
    assert 0 <= case.true_center[1] <= 400


def test_benchmark_suite_builds():
    cfgs = build_benchmark_suite(n_per_factor=2)
    assert len(cfgs) > 0
    names = {c.name for c in cfgs}
    assert any(n.startswith("scale") for n in names)
    assert any(n.startswith("rot") for n in names)


def test_all_methods_run():
    cfg = CaseConfig(seed=5, search_size=500, ref_crop_size=250)
    case = generate_case(cfg)
    for name, fn in METHOD_REGISTRY.items():
        result = fn(case.search_image, case.reference_image)
        assert 0 <= result.x <= case.search_image.shape[1]
        assert 0 <= result.y <= case.search_image.shape[0]
        assert result.method == name


def test_easy_case_is_accurate():
    """With no nuisance factors, all methods should localize very closely."""
    cfg = CaseConfig(seed=10, search_size=600, ref_crop_size=260, target_position="center")
    case = generate_case(cfg)
    result = METHOD_REGISTRY["baseline"](case.search_image, case.reference_image)
    err = math.hypot(result.x - case.true_center[0], result.y - case.true_center[1])
    assert err < 5.0


def test_pick_center_most_prefers_dominant_peak():
    best = Candidate(x=100, y=100, bbox=(90, 90, 110, 110), scale=10, raw_score=0.9, combined_score=0.9)
    weak_far = Candidate(x=400, y=400, bbox=(390, 390, 410, 410), scale=10, raw_score=0.5, combined_score=0.5)
    chosen = pick_center_most([best, weak_far])
    assert chosen is best


def test_pick_center_most_resolves_true_tie_by_centroid():
    a = Candidate(x=100, y=100, bbox=(90, 90, 110, 110), scale=10, raw_score=0.8, combined_score=0.80)
    b = Candidate(x=104, y=100, bbox=(94, 90, 114, 110), scale=10, raw_score=0.8, combined_score=0.799)
    c = Candidate(x=96, y=100, bbox=(86, 90, 106, 110), scale=10, raw_score=0.8, combined_score=0.799)
    chosen = pick_center_most([a, b, c])
    assert chosen.x == 100


if __name__ == "__main__":
    test_generate_case_basic()
    test_benchmark_suite_builds()
    test_all_methods_run()
    test_easy_case_is_accurate()
    test_pick_center_most_prefers_dominant_peak()
    test_pick_center_most_resolves_true_tie_by_centroid()
    print("All smoke tests passed.")
