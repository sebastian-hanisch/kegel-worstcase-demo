"""Randfälle, an denen das PEP erfahrungsgemäß bricht (DEMO-PLAYBOOK §12)."""

import pickle

import numpy as np
import pytest

import kwc_constants as C
import kwc_evaluation as ev
import kwc_pep as P


def test_one_step_is_the_smallest_program():
    for method in P.METHODS:
        res = P.worst_case(method, 1, 1.0)
        assert res.pep.k == 3 and res.pep.m == 2 and res.pep.n_rows == 6 and res.status == "optimal" and 0.1 < res.value < 0.2
    assert P.worst_case("fgm", 1).value == pytest.approx(P.worst_case("gd", 1, 1.0).value, abs=1e-8)


def test_tiny_and_large_step_lengths_follow_the_formula_where_the_solver_converges():
    """h → 0: der Wert geht gegen 1/2; h = 2: Wert 1/2 (kein Fortschritt, Überschießen); h > 2: der Wert wächst ((h − 1)^{2N}/2: das Verfahren divergiert). Der Status des Lösers kann „nan“ oder „stalled“ sein."""
    assert P.worst_case("gd", 1, 0.01).value == pytest.approx(P.gd_formula(1, 0.01), abs=1e-4)
    assert P.worst_case("gd", 3, 2.0).value == pytest.approx(0.5, abs=1e-5)
    assert P.worst_case("gd", 3, 2.2).value == pytest.approx(P.gd_formula(3, 2.2), rel=1e-5) and P.worst_case("gd", 3, 3.0).value > 20.0


def test_a_big_mu_gives_small_values_whatever_the_status_of_the_solver():
    """Der Status hängt von der CPU ab: auf dem Entwicklungsrechner „stalled“, auf dem Linux-Runner der CI „optimal“; der Wert ist in beiden Fällen derselbe (Band)."""
    big = P.worst_case("gd", 6, 1.0, 0.5)
    assert isinstance(big.status, str) and 0 < big.value < 1e-3 and big.value == pytest.approx(0.5 ** 12 / 4, abs=2e-5)            # (1 − μ)^{2N} / 4: der Wert ist winzig
    assert P.worst_case("gd", 3, 1.0, 0.9).value < P.worst_case("gd", 3, 1.0, 0.5).value
    row = ev.result_row(big, None)
    assert ev.verdict_result(ev.Settings(n=6, mu=0.5), row)[0] in {"success", "warning"}


def test_rates_for_a_big_mu_stop_at_the_solver_limit():
    rows = ev.rate_rows("gd", 1.0, 0.5)
    assert [r["N"] for r in rows] == [1, 2, 3, 4, 5] and all(r["reference"] is None for r in rows) and len(ev.rate_rows("gd", 1.0, 0.0, nmax=3)) == 3


def test_every_corner_of_the_controls_gives_a_finite_result_and_a_verdict():
    for method in P.METHODS:
        for n in (C.N_MIN, C.N_MAX):
            for h in (C.H_CHOICES[0], C.H_CHOICES[-1]):
                for mu in (C.MU_CHOICES[0], C.MU_CHOICES[-1]):
                    s = ev.Settings(method, n, h, mu)
                    res = ev.main_result(s)
                    row = ev.result_row(res, P.reference(method, n, h, mu))
                    assert np.isfinite(res.value) and res.value > 0 and ev.verdict_result(s, row)[0] in {"success", "warning"}, (method, n, h, mu)


def test_the_cached_results_can_be_pickled():
    """Streamlit speichert die Rückgabewerte mit pickle: das Handbeispiel (früher mit lokaler Funktion) und die Ergebnisse müssen serialisierbar sein."""
    hand = ev.hand_view()
    assert pickle.loads(pickle.dumps(hand))["value_hand"] == hand["value_hand"]
    res = ev.main_result(ev.Settings())
    obj = (res, ev.result_row(res, None), ev.proof_view(res))
    back = pickle.loads(pickle.dumps(obj))
    assert back[0].value == res.value and back[2]["positive"] == obj[2]["positive"]
    assert pickle.loads(pickle.dumps(ev.step_scan(1, 0.0, grid=(0.5, 1.0))))[0]["h"] == 0.5


def test_the_tolerance_of_the_solver_changes_the_last_digits_only():
    loose = P.worst_case("gd", 3, 1.0, tol=1e-6).value
    tight = P.worst_case("gd", 3, 1.0, tol=1e-10).value
    assert abs(loose - 1 / 14) < 1e-5 and abs(tight - 1 / 14) < 1e-9


def test_the_scan_grid_and_choices_are_consistent():
    assert len(C.H_SCAN) == 20 and C.H_SCAN[0] == 0.1 and C.H_SCAN[-1] == 1.95 and max(C.H_CHOICES) < 2.0 and 1.0 in C.H_CHOICES and 0.0 in C.MU_CHOICES
    assert all(mu < 1.0 for mu in C.MU_CHOICES) and set(C.MU_NMAX) == set(C.MU_CHOICES)
