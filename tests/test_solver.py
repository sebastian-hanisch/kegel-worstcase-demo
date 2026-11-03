"""Der primal-duale Löser (kwc_solver, aus Stück 6 übernommen): die Mini-Programme des Kerns mit bekanntem Optimum, Primal- und Dualwert."""

import pytest

import kwc_solver as PD


@pytest.mark.parametrize("name,scaling", [("lp", "nt"), ("soc", "nt"), ("sdp", "nt"), ("soc", "jordan"), ("sdp", "jordan")])
def test_the_hand_programs_are_solved(tiny_cones, name, scaling):
    prob, opt = tiny_cones[name]
    r = PD.solve_pd(prob, tol=1e-9, scaling=scaling)
    assert r.status == "optimal" and r.value == pytest.approx(opt, abs=1e-7) and r.dual_value == pytest.approx(opt, abs=1e-7) and abs(r.certificate_gap) < 1e-7


def test_the_dual_point_lies_in_the_cone_and_is_feasible(tiny_cones):
    import numpy as np

    prob, _ = tiny_cones["soc"]
    r = PD.solve_pd(prob, tol=1e-10)
    assert np.abs(prob.A.T @ r.z + prob.c).max() < 1e-8 and r.z[0] >= np.linalg.norm(r.z[1:]) - 1e-9


@pytest.mark.parametrize("h", [0.1, 0.15, 0.2, 0.25, 0.3])
def test_the_status_follows_the_residuals_not_the_way_the_solver_stopped(h):
    """Bei kleinem h stoppt der Löser an der Rechengrenze (früher als „nan“ angezeigt). Der Status hängt von der CPU ab und wird nicht festgeschrieben;
    festgeschrieben ist der Zusammenhang: ein angenommener Status (optimal / rounding) ⇔ die Residuen und die Lücke, hier unabhängig aus x, s, z nachgerechnet, liegen unter der Schwelle."""
    import numpy as np

    import kwc_pep as P

    pep = P.build("gd", 4, h, 0.0)
    prob, tol = pep.problem, 1e-9
    r = PD.solve_pd(prob, tol=tol)
    pres = np.linalg.norm(prob.A @ r.x + r.s - prob.b) / (1 + np.linalg.norm(prob.b))
    dres = np.linalg.norm(prob.A.T @ r.z + prob.c) / (1 + np.linalg.norm(prob.c))
    gap = float(r.s @ r.z)
    scale = max(1.0, abs(float(prob.c @ r.x)))
    inside = max(pres, dres) <= PD.ACCEPT_FACTOR * tol and gap <= PD.ACCEPT_FACTOR * tol * scale
    assert (r.status in {"optimal", "rounding"}) == inside
    assert r.status != "nan" or not inside                            # ein „nan“ bleibt nur, wo die Genauigkeit nicht erreicht ist
    assert r.stop in {"optimal", "nan", "stalled", "linalg", "max_iter", "left_cone"} and (r.status == "rounding") <= (r.stop in PD.MECHANICAL_STOPS)
    assert r.status == r.stop or r.status == "rounding"
    assert -r.value == pytest.approx(0.5 * max(1 / (2 * 4 * h + 1), (1 - h) ** 8), abs=1e-7)       # der Wert stimmt unabhängig vom Status (Formel der Literatur)


def test_the_acceptance_threshold_is_a_thousand_times_the_tolerance():
    ok = {"pres": 9e-7, "dres": 1e-9, "gap": 9e-7, "pobj": 0.2}
    assert PD._within_accept(ok, 1e-9)
    for bad in ({"pres": 2e-6}, {"dres": 2e-6}, {"gap": 2e-6}, {"gap": 2e-3, "pobj": 1000.0 * 0 + 0.2}):
        assert not PD._within_accept({**ok, **bad}, 1e-9)
    assert PD._within_accept({**ok, "gap": 5e-4, "pobj": 1000.0}, 1e-9)          # die Lücke ist relativ zu max(1, |Zielwert|)
