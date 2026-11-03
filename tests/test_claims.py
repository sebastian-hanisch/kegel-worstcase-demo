"""Jede Zahl aus README.md und den App-Texten wird hier über dieselben Auswertungsfunktionen nachgerechnet (DEMO-PLAYBOOK §4).

Die Werte sind deterministisch bis auf die Rundung der BLAS-Routinen (Toleranz des Lösers 10⁻⁹): Bänder 10⁻⁷ für Werte; Zeiten und Iterationszahlen nur in weiten Bändern."""

import hashlib
from pathlib import Path

import numpy as np
import pytest

import kwc_constants as C
import kwc_evaluation as ev
import kwc_pep as P


@pytest.fixture(scope="module")
def gd():
    return ev.rate_rows("gd", 1.0, 0.0)


@pytest.fixture(scope="module")
def methods():
    return ev.methods_rows(0.0)


@pytest.fixture(scope="module")
def scan():
    return ev.step_scan(4, 0.0)


def test_the_gradient_method_reaches_L_R2_over_4N_plus_2(gd):
    """README: Gradientenverfahren mit Schritt 1/L: N = 1 / 2 / 3 / 4 / 8: 0,1666666662 / 0,1 / 0,0714285714 / 0,0555555553 / 0,0294117641 = 1/(4N+2) auf 10⁻⁹; Gram-Matrix 3×3 bis 10×10, Rang 1, 13 bis 25 Iterationen (Band 8 bis 40)."""
    assert [r["N"] for r in gd] == list(range(1, 13)) and all(r["status"] == "optimal" for r in gd)
    assert all(abs(r["value"] - 1.0 / (4 * r["N"] + 2)) < 1e-8 and r["deviation"] is not None and abs(r["deviation"]) < 1e-8 for r in gd)
    assert gd[0]["value"] == pytest.approx(1.0 / 6.0, abs=1e-8) and gd[7]["value"] == pytest.approx(0.0294117647, abs=1e-8)
    assert [r["gram"] for r in gd][:3] == [3, 4, 5] and gd[7]["gram"] == 10 and [r["rows"] for r in gd][:4] == [6, 12, 20, 30] and gd[7]["rows"] == 90
    assert all(r["rank"] == 1 and abs(r["gap"]) < 1e-8 and 8 <= r["iterations"] <= 40 for r in gd)


def test_the_size_and_the_time_of_the_program(gd):
    """README: Gram 4×4 (N = 2): 13 Variablen, 23 Zeilen; 6×6 (N = 4): 26, 52; 14×14 (N = 12): 118, 288; die Zeit wächst mit N (N = 12 mindestens 3-mal N = 2; Band 5 bis 5000 ms)."""
    by = {r["N"]: r for r in gd}
    assert (by[2]["variables"], by[2]["gram"]) == (13, 4) and (by[4]["variables"], by[4]["gram"]) == (26, 6) and (by[12]["variables"], by[12]["gram"]) == (118, 14)
    assert by[12]["seconds"] > 3.0 * by[2]["seconds"] and all(5e-3 <= r["seconds"] <= 5.0 for r in gd)


def test_the_step_length_matches_the_formula_and_the_best_step_is_above_one_over_L(scan):
    """README (N = 4): h = 0,25 / 0,5 / 0,75 / 1 / 1,25 / 1,5 / 1,75 / 1,9: 0,16666667 / 0,1 / 0,07142857 / 0,05555556 / 0,04545454 / 0,03846154 / 0,05005646 / 0,2152336 (Abweichung ≤ 10⁻⁷);
    das Minimum der 20 Schrittlängen liegt bei h = 1,7 (0,03425), bei h = 1/L sind es 0,05556; die beste Schrittlänge der Formel h_opt = 1,71."""
    by = {r["h"]: r for r in scan}
    assert len(scan) == 20 and max(abs(r["deviation"]) for r in scan) < 1e-7
    for h, v in ((0.25, 0.16666667), (0.5, 0.1), (0.75, 0.07142857), (1.0, 0.05555556), (1.25, 0.04545454), (1.5, 0.03846154), (1.75, 0.05005646), (1.9, 0.2152336)):
        assert P.worst_case("gd", 4, h).value == pytest.approx(v, abs=1e-7), h
    assert by[1.9]["value"] > by[1.5]["value"] and by[0.2]["value"] > by[1.0]["value"] > by[1.5]["value"]
    best = ev.best_step(scan)
    assert best["h"] == 1.7 and best["value"] == pytest.approx(0.03425, abs=5e-5) and P.gd_optimal_step(4) == pytest.approx(1.7145, abs=1e-3)
    assert [r["proven"] for r in scan].count(True) == 10 and by[1.0]["proven"] and not by[1.1]["proven"]


def test_the_optimal_step_grows_with_N_and_stays_below_two():
    """README: h_opt aus der Formel N = 1 / 2 / 3 / 4 / 6 / 8 / 12: 1,50 / 1,61 / 1,67 / 1,71 / 1,77 / 1,81 / 1,85 (die Quelle nennt 3/2 ≤ h_opt < 2 und h_opt → 2); das SDP trifft den Wert der Formel bei diesen Schrittlängen."""
    hs = [P.gd_optimal_step(N) for N in (1, 2, 3, 4, 6, 8, 12)]
    assert hs == sorted(hs) and 1.5 <= hs[0] and hs[-1] < 2.0 and [round(h, 2) for h in hs] == [1.5, 1.61, 1.67, 1.71, 1.77, 1.81, 1.85]
    assert [round(P.gd_formula(N, P.gd_optimal_step(N)), 5) for N in (1, 2, 3, 4, 6, 8, 12)] == [0.125, 0.06736, 0.04536, 0.03398, 0.02246, 0.0167, 0.011]
    for N in (1, 4):
        assert P.worst_case("gd", N, P.gd_optimal_step(N)).value == pytest.approx(P.gd_formula(N, P.gd_optimal_step(N)), abs=1e-7)


def test_acceleration_halves_the_value_and_changes_the_slope(methods):
    """README (N = 12): Gradientenverfahren 0,02 / FGM 0,00915 / OGM 0,00463; OGM = 1/(2θ_N²) auf 10⁻⁸; OGM/FGM von 0,750 (N = 1) bis 0,507 (N = 12); Steigungen (N = 4 bis 12) −0,93 / −1,49 / −1,56."""
    r = methods["rows"]
    assert r["gd"][-1]["value"] == pytest.approx(0.02, abs=1e-7) and r["fgm"][-1]["value"] == pytest.approx(0.00914552, abs=1e-6) and r["ogm"][-1]["value"] == pytest.approx(0.0046341, abs=1e-6)
    assert all(abs(x["deviation"]) < 1e-8 for x in r["ogm"]) and r["ogm"][0]["value"] == pytest.approx(0.125, abs=1e-8)
    ratio = [o["value"] / f["value"] for o, f in zip(r["ogm"], r["fgm"])]
    assert ratio[0] == pytest.approx(0.75, abs=1e-6) and ratio[-1] == pytest.approx(0.507, abs=2e-3) and all(a >= b for a, b in zip(ratio, ratio[1:]))
    sl = methods["slopes"]
    assert -1.0 <= sl["gd"] <= -0.85 and -1.6 <= sl["fgm"] <= -1.4 and -1.7 <= sl["ogm"] <= -1.45 and sl["gd"] > sl["fgm"] > sl["ogm"]
    assert r["fgm"][0]["value"] == pytest.approx(r["gd"][0]["value"], abs=1e-7) and r["fgm"][2]["value"] < r["gd"][2]["value"] and all(f["value"] < P.fgm_bound(f["N"]) for f in r["fgm"])


def test_strong_convexity_lowers_the_value():
    """README (GD, Schritt 1/L): μ = 0,1 / 0,3 / 0,5, N = 1 / 2 / 4: 0,14944649 / 0,08010793 / 0,03513562; 0,11187215 / 0,0432909 / 0,00901082; 0,07142857 / 0,01612903 / 0,00097847."""
    rows = {r["mu"]: r["values"] for r in ev.mu_table()}
    for mu, vals in ((0.1, (0.14944649, 0.08010793, 0.03513562)), (0.3, (0.11187215, 0.0432909, 0.00901082)), (0.5, (0.07142857, 0.01612903, 0.00097847))):
        assert [rows[mu][n]["value"] for n in C.MU_TABLE_N] == pytest.approx(list(vals), abs=1e-6), mu
    assert [rows[0.0][n]["value"] for n in C.MU_TABLE_N] == pytest.approx([1 / 6, 0.1, 1 / 18], abs=1e-8)
    assert all(rows[0.1][n]["value"] < rows[0.0][n]["value"] and rows[0.3][n]["value"] < rows[0.1][n]["value"] and rows[0.5][n]["value"] < rows[0.3][n]["value"] for n in C.MU_TABLE_N)
    assert all(v["value"] < (1 - mu) ** (2 * n) / 2 for mu in (0.1, 0.3, 0.5) for n, v in rows[mu].items())


def test_the_hand_example(gd):
    """README: N = 1, Schritt 1/L: Wert 1/6, Gram-Matrix v vᵀ mit v = (1, 1/3, 1/3), Rang 1, die Daten von f₁ sind zulässig (kleinste Zeile ≥ −10⁻¹²), vier Zeilen sind aktiv."""
    hand = ev.hand_view()
    assert hand["result"].value == pytest.approx(1.0 / 6.0, abs=1e-8) and hand["result"].rank == 1 and hand["min_slack_hand"] > -1e-12 and hand["value_hand"] == pytest.approx(1.0 / 6.0)
    assert hand["G_hand"] == pytest.approx(np.outer([1, 1 / 3, 1 / 3], [1, 1 / 3, 1 / 3])) and sum(1 for r in hand["rows"] if abs(r["hand"]) < 1e-12) == 4 and len(hand["rows"]) == 6
    assert hand["result"].G == pytest.approx(hand["G_hand"], abs=1e-5)                       # das SDP findet dieselbe Gram-Matrix


def test_the_proof_by_dual_multipliers():
    """README (GD, Schritt 1/L, N = 4): Schranke 0,0555555556 = Primalwert, Identität auf 10⁻¹⁵ (Band 10⁻⁹), 25 von 30 Multiplikatoren positiv (Band 15 bis 30), Rang von S 5 von 6."""
    res = P.worst_case("gd", 4, 1.0)
    p = ev.proof_view(res)
    assert p["bound"] == pytest.approx(1.0 / 18.0, abs=1e-8) and abs(p["gap"]) < 1e-8 and p["identity"] < 1e-9 and p["rows"] == 30
    assert 15 <= p["positive"] <= 30 and p["S_rank"] == 5 and p["S_size"] == 6 and p["lam_matrix"].shape == (6, 6) and p["names"] == ["*", "0", "1", "2", "3", "4"] and p["lam_min"] > -1e-9


def test_the_rank_is_one_for_the_gradient_method_and_fgm_but_not_for_ogm():
    """README: die optimale Gram-Matrix hat Rang 1 für GD (N = 3, 6; h = 1,9; μ = 0,3) und FGM (N = 4, 8): die schlechteste Funktion ist eindimensional. Beim OGM ist die Menge der schlechtesten Funktionen groß:
    der Löser liefert den Mittelpunkt der optimalen Menge, Rang N + 1 von N + 2 (N = 1 … 6)."""
    for method, N, h, mu in (("gd", 3, 1.0, 0.0), ("gd", 6, 1.0, 0.0), ("gd", 4, 1.9, 0.0), ("gd", 4, 1.0, 0.3), ("fgm", 4, 1.0, 0.0), ("fgm", 8, 1.0, 0.0)):
        res = P.worst_case(method, N, h, mu)
        assert res.rank == 1 and res.eig[1] < 1e-7 * res.eig[0], (method, N)
    for N in range(1, 9):
        res = P.worst_case("ogm", N)
        assert res.rank == N + 1 and res.pep.k == N + 2 and res.eig[-1] < 1e-7, N


def test_the_preset_help_texts_name_the_values():
    """Hilfetexte der Presets: 1/18 bei N = 4; h = 1,75: 0,05006 gegen h = 1,5: 0,03846; OGM N = 8: 0,00904, FGM 0,01756, Gradientenverfahren 0,02941; FGM-Schranke 0,0247."""
    assert P.worst_case("gd", 4, 1.0).value == pytest.approx(1 / 18, abs=1e-8)
    assert P.worst_case("gd", 4, 1.75).value == pytest.approx(0.05006, abs=5e-6) and P.worst_case("gd", 4, 1.5).value == pytest.approx(0.03846, abs=5e-6)
    assert P.worst_case("ogm", 8).value == pytest.approx(0.00904, abs=5e-6) and P.worst_case("fgm", 8).value == pytest.approx(0.01756, abs=5e-6) and P.worst_case("gd", 8, 1.0).value == pytest.approx(0.02941, abs=5e-6)
    assert P.fgm_bound(8) == pytest.approx(0.0247, abs=5e-5)


def test_the_kernel_is_byte_identical_in_the_series():
    """README: kegel_kern.py ist wortgleich mit Stück 2 bis 10: Prüfsumme wie im Schwester-Repo, sofern es nebenan liegt."""
    here = Path(__file__).resolve().parent.parent / "kegel_kern.py"
    sibling = here.parent.parent / "kegel-kreuzung-demo" / "kegel_kern.py"
    if not sibling.exists():
        pytest.skip("Schwester-Repo liegt nicht nebenan")
    assert hashlib.sha256(here.read_bytes()).hexdigest() == hashlib.sha256(sibling.read_bytes()).hexdigest()
