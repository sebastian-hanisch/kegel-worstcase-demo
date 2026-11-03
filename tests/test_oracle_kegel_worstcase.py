"""Orakel (DEMO-PLAYBOOK §12, ORACLE-PLAYBOOK §5): das PEP gegen einen anderen Rechenweg.

- cvxpy formuliert das PEP **selbst** (eigene Variablen `G`, `f`, eigene Zeilen aus den Koeffizienten der Verfahren, `cp.PSD`) und löst es mit Clarabel und SCS: gleicher Wert wie unser Aufbau mit dem Löser aus Stück 6.
  Das prüft Aufbau und Löser zugleich, weil nichts aus kwc_pep.build übernommen wird außer den Abfragepunkten des Verfahrens.
- die Formeln der Literatur (Gradientenverfahren, OGM) für mehrere Paare (N, h) in einem Band von 10⁻⁷

Fehlt cvxpy im Test-venv, überspringen sich diese Tests (pytest.importorskip), sie raten nie."""

import numpy as np
import pytest

import kwc_pep as P


def _cp():
    cp = pytest.importorskip("cvxpy")
    if "CLARABEL" not in cp.installed_solvers():
        pytest.skip("Clarabel fehlt")
    return cp


def _cvxpy_value(cp, method, N, h=1.0, mu=0.0, solver=None):
    pts, k = P.method_points(method, N, h)
    G = cp.Variable((k, k), PSD=True)
    f = cp.Variable(len(pts))
    allp = [(np.zeros(k), np.zeros(k), None)] + [(x, g, i) for i, (x, g) in enumerate(pts)]
    cons = [G[0, 0] <= 1]
    for a, (xi, gi, fi) in enumerate(allp):
        for b, (xj, gj, fj) in enumerate(allp):
            if a == b:
                continue
            fi_e = f[fi] if fi is not None else 0.0
            fj_e = f[fj] if fj is not None else 0.0
            ip = lambda u, v: u @ G @ v
            row = fi_e - fj_e - ip(gj, xi - xj)
            dg, dx = gi - gj, xi - xj
            if mu == 0.0:
                row = row - ip(dg, dg) / 2.0
            else:
                row = row - (ip(dg, dg) + mu * ip(dx, dx) - 2.0 * mu * ip(gj - gi, xj - xi)) / (2.0 * (1.0 - mu))
            cons.append(row >= 0)
    prob = cp.Problem(cp.Maximize(f[len(pts) - 1]), cons)
    kw = dict(solver=cp.CLARABEL, tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10) if solver is None else dict(solver=solver, eps=1e-9, max_iters=200000)
    prob.solve(**kw)
    return float(prob.value)


@pytest.mark.filterwarnings("ignore:Solution may be inaccurate")
@pytest.mark.parametrize("method,N,h,mu", [("gd", 3, 1.0, 0.0), ("gd", 3, 1.5, 0.0), ("gd", 2, 0.5, 0.0), ("fgm", 3, 1.0, 0.0), ("ogm", 3, 1.0, 0.0), ("gd", 3, 1.0, 0.3)])
def test_cvxpy_finds_the_same_worst_case_value_with_clarabel(method, N, h, mu):
    cp = _cp()
    assert P.worst_case(method, N, h, mu).value == pytest.approx(_cvxpy_value(cp, method, N, h, mu), abs=1e-6)


@pytest.mark.filterwarnings("ignore:Solution may be inaccurate")
def test_scs_the_admm_of_cvxpy_finds_the_same_value():
    cp = _cp()
    if "SCS" not in cp.installed_solvers():
        pytest.skip("SCS fehlt")
    assert P.worst_case("gd", 3, 1.0).value == pytest.approx(_cvxpy_value(cp, "gd", 3, 1.0, solver=cp.SCS), abs=1e-5)


@pytest.mark.parametrize("N", [1, 2, 3, 5, 8])
def test_the_literature_formulas_for_several_pairs(N):
    for h in (0.3, 1.0, 1.4, 1.8):
        assert P.worst_case("gd", N, h).value == pytest.approx(P.gd_formula(N, h), abs=1e-7), (N, h)
    assert P.worst_case("ogm", N).value == pytest.approx(P.ogm_formula(N), abs=1e-7)
