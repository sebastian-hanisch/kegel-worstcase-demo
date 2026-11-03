"""Das PEP (kwc_pep): Handbeispiel, Aufbau gegen unabhängige Rechnung, Beweis in beide Richtungen, Eigenschaften, Fehlerfälle."""

import numpy as np
import pytest

import kwc_pep as P


def test_the_hand_data_are_a_feasible_point_that_hits_the_optimum():
    """N = 1, h = 1: f₁(x) = |x|/3 − 1/18 (|x| ≥ 1/3): x₀ = 1 (g₀ = 1/3, f₀ = 5/18), x₁ = 2/3 (g₁ = 1/3, f₁ = 1/6); Gram-Matrix v vᵀ mit v = (1, 1/3, 1/3), Rang 1."""
    f, df, xs, value = P.hand_function(1, 1.0)
    assert xs == pytest.approx([1.0, 2.0 / 3.0]) and f(1.0) == pytest.approx(5.0 / 18.0) and df(1.0) == pytest.approx(1.0 / 3.0) and value == pytest.approx(1.0 / 6.0) and f(0.0) == 0.0
    pep = P.build("gd", 1, 1.0)
    v = np.array([1.0, 1.0 / 3.0, 1.0 / 3.0])
    z = P.variables(pep, [f(xs[0]), f(xs[1])], np.outer(v, v))
    s = P.slacks(pep, z)
    assert pep.n_rows == 6 and s[:6].min() > -1e-12 and s[6] == pytest.approx(0.0, abs=1e-15)                    # alle Interpolationszeilen ≥ 0, G₀₀ = 1 (Normierung aktiv)
    assert -pep.problem.c @ z == pytest.approx(1.0 / 6.0)                                                        # Zielwert des zulässigen Punkts
    assert P.worst_case("gd", 1, 1.0).value == pytest.approx(1.0 / 6.0, abs=1e-8)                               # der Löser findet nichts Besseres: der Punkt ist optimal


def test_the_row_pattern_of_the_hand_example():
    """Zeile (i, j) = f_i − f_j − g_j (x_i − x_j) − (g_i − g_j)²/2 an den Daten von f₁ (von Hand): (x₀, *) = 5/18 − 1/18 = 2/9, (x₁, *) = 1/6 − 1/18 = 1/9, die vier übrigen Zeilen sind 0 (aktiv)."""
    f, df, xs, _ = P.hand_function(1, 1.0)
    pep = P.build("gd", 1, 1.0)
    v = np.array([1.0, 1.0 / 3.0, 1.0 / 3.0])
    s = dict(zip(pep.pairs, P.slacks(pep, P.variables(pep, [f(xs[0]), f(xs[1])], np.outer(v, v)))[:6]))
    assert set(s) == {(-1, 0), (0, -1), (-1, 1), (1, -1), (0, 1), (1, 0)}
    assert s[(0, -1)] == pytest.approx(2.0 / 9.0) and s[(1, -1)] == pytest.approx(1.0 / 9.0)
    assert all(s[p] == pytest.approx(0.0, abs=1e-12) for p in ((-1, 0), (-1, 1), (0, 1), (1, 0)))


@pytest.mark.parametrize("N,h,expected", [(1, 1.0, 1.0 / 6.0), (2, 1.0, 0.1), (1, 1.5, 0.125), (3, 1.0, 1.0 / 14.0)])
def test_known_values(N, h, expected):
    assert P.worst_case("gd", N, h).value == pytest.approx(expected, abs=1e-8)


@pytest.mark.parametrize("method", P.METHODS)
def test_the_method_points_reproduce_the_numerical_iterates(method):
    """Jeder Abfragepunkt, aus den Koeffizienten über der Basis (x₀, g₀, …) zusammengesetzt, ist der Punkt, den `simulate` auf einer Funktion rechnet."""
    rng = np.random.default_rng(1)
    d, N = 9, 5
    Q = np.diag(rng.uniform(0.1, 1.0, d))
    grad = lambda x: Q @ x
    x0 = rng.normal(size=d)
    pts, k = P.method_points(method, N, 1.3)
    sim = P.simulate(method, N, grad, x0, 1.3)
    assert len(pts) == len(sim) == N + 1 and k == N + 2
    basis = [x0] + [grad(p) for p in sim]
    for (xc, gc), x_sim, g_sim in zip(pts, sim, basis[1:]):
        assert sum(c * b for c, b in zip(xc, basis)) == pytest.approx(x_sim, abs=1e-10)
        assert sum(c * b for c, b in zip(gc, basis)) == pytest.approx(g_sim, abs=1e-10)


@pytest.mark.parametrize("method,h", [("gd", 1.0), ("gd", 1.5), ("fgm", 1.0), ("ogm", 1.0)])
def test_the_value_bounds_every_function_from_above(method, h):
    """Der Worst-Case ist eine obere Schranke: Quadriken mit Eigenwerten in [0,05; 1] und Zufallsstart mit ‖x₀‖ ≤ 1 (x* = 0, f* = 0) bleiben darunter."""
    N = 4
    bound = P.worst_case(method, N, h).value
    rng = np.random.default_rng(2)
    worst = 0.0
    for _ in range(60):
        d = int(rng.integers(1, 8))
        q = rng.uniform(0.05, 1.0, d)
        x0 = rng.normal(size=d)
        x0 /= max(1.0, np.linalg.norm(x0))
        pts = P.simulate(method, N, lambda x: q * x, x0, h)
        worst = max(worst, 0.5 * float(q @ pts[-1] ** 2))
    assert worst <= bound + 1e-9 and worst > 0.05 * bound


def test_the_two_one_dimensional_functions_reach_the_value_from_below():
    """Gradientenverfahren auf f₁ und f₂(x) = x²/2 ab x₀ = 1: die beiden Terme der Formel; ihr Maximum ist der PEP-Wert (Beweis von oben und von unten)."""
    for N, h in ((1, 1.0), (2, 1.0), (4, 0.5), (4, 1.5), (3, 1.9)):
        f1_value = P.hand_function(N, h)[3]
        f2_value = 0.5 * (1.0 - h) ** (2 * N)
        assert f1_value == pytest.approx(0.5 / (2 * N * h + 1))
        assert max(f1_value, f2_value) == pytest.approx(P.gd_formula(N, h))
        assert P.worst_case("gd", N, h).value == pytest.approx(max(f1_value, f2_value), abs=1e-7)


def test_the_dual_gives_the_proof_as_an_identity():
    res = P.worst_case("gd", 3, 1.0)
    assert res.status == "optimal" and abs(res.gap) < 1e-8 and res.bound == pytest.approx(1.0 / 14.0, abs=1e-8) and P.identity_deviation(res, samples=8, seed=3) < 1e-9
    assert res.lam.min() > -1e-9 and np.linalg.eigvalsh(res.S).min() > -1e-9 and res.rank == 1 and res.pep.n_rows == 20


def test_properties_of_the_values():
    vals = {m: [P.worst_case(m, N).value for N in range(1, 7)] for m in P.METHODS}
    for m in P.METHODS:
        assert all(b < a for a, b in zip(vals[m], vals[m][1:]))                                  # fallend in N
    assert all(o <= f + 1e-8 and f <= g + 1e-8 for g, f, o in zip(vals["gd"], vals["fgm"], vals["ogm"])) and vals["fgm"][0] == pytest.approx(vals["gd"][0]) and vals["fgm"][1] == pytest.approx(vals["gd"][1])
    mu = [P.worst_case("gd", 3, 1.0, m).value for m in (0.0, 0.1, 0.3)]
    assert mu[0] > mu[1] > mu[2]
    assert all(v < P.fgm_bound(N) for N, v in zip(range(1, 7), vals["fgm"])) and vals["gd"][0] < P.worst_case("gd", 1, 0.5).value                # ein längerer Schritt (1/L) ist besser als 1/(2L)


def test_the_formulas_and_the_optimal_step():
    assert P.gd_formula(1, 1.0) == pytest.approx(1.0 / 6.0) and P.gd_formula(4, 1.0) == pytest.approx(1.0 / 18.0) and P.gd_formula(1, 1.5) == pytest.approx(0.125) and P.gd_formula(1, 2.0) == 0.5
    assert [round(P.gd_optimal_step(N), 2) for N in (1, 2, 3, 4, 6, 8)] == [1.5, 1.61, 1.67, 1.71, 1.77, 1.81]
    assert P.gd_proven(1.0) and not P.gd_proven(1.25) and P.ogm_theta(1) == pytest.approx(2.0) and P.ogm_formula(1) == pytest.approx(0.125) and P.fgm_bound(3) == pytest.approx(0.125)
    assert P.reference("gd", 2, 1.0, 0.0) == pytest.approx(0.1) and P.reference("ogm", 1, 1.0, 0.0) == pytest.approx(0.125) and P.reference("fgm", 2, 1.0, 0.0) is None and P.reference("gd", 2, 1.0, 0.1) is None


def test_bad_arguments_are_rejected():
    for bad in (lambda: P.worst_case("gd", 0), lambda: P.worst_case("gd", 2, 1.0, 1.0), lambda: P.worst_case("gd", 2, 1.0, -0.1), lambda: P.method_points("newton", 2)):
        with pytest.raises(ValueError):
            bad()
    with pytest.raises(ValueError):
        P.simulate("newton", 2, lambda x: x, np.ones(1))
