"""Abbildungen: Achsen gesperrt, Spuren stimmen mit den Zahlen überein."""

import pytest

import kwc_constants as C
import kwc_evaluation as ev
import kwc_pep as P
import kwc_visualization as V


def _locked(fig):
    axes = [v for k, v in fig.layout.to_plotly_json().items() if k.startswith(("xaxis", "yaxis"))]
    return bool(axes) and all(ax.get("fixedrange") is True for ax in axes)


@pytest.fixture(scope="module")
def rows():
    return ev.rate_rows("gd", 1.0, 0.0, nmax=4)


def test_rate_chart_has_the_values_the_formula_and_the_selected_point(rows):
    fig = V.build_rate(rows, "Gradientenverfahren", C.COLORS["gd"], selected_n=3)
    assert _locked(fig) and [t.name for t in fig.data] == ["Gradientenverfahren", "Formel der Literatur", "N = 3"] and fig.layout.yaxis.type == "log" and fig.layout.xaxis.type == "linear"
    assert list(fig.data[0].x) == [1, 2, 3, 4] and list(fig.data[0].y) == pytest.approx([r["value"] for r in rows]) and list(fig.data[2].x) == [3]
    loglog = V.build_rate(rows, "GD", C.COLORS["gd"], loglog=True)
    assert loglog.layout.xaxis.type == "log" and len(loglog.data) == 2
    assert len(V.build_rate(ev.rate_rows("gd", 1.0, 0.3, nmax=2), "GD", C.COLORS["gd"]).data) == 1                    # μ > 0: keine Vergleichsformel


def test_hand_chart_has_the_curve_and_three_points():
    hand = ev.hand_view()
    fig = V.build_hand(hand)
    assert _locked(fig) and len(fig.data) == 2 and len(fig.data[0].x) == 200 and list(fig.data[1].x) == pytest.approx([0.0, 1.0, 2.0 / 3.0]) and list(fig.data[1].y) == pytest.approx([0.0, 5.0 / 18.0, 1.0 / 6.0])
    assert len(fig.layout.annotations) == 1


def test_step_chart_marks_the_proven_range_and_the_optimum():
    scan = ev.step_scan(2, 0.0, grid=(0.5, 1.0, 1.5, 1.9))
    fig = V.build_step(scan, 2, 0.0)
    assert _locked(fig) and [t.name for t in fig.data][0] == "PEP-Wert (SDP)" and len(fig.data) == 3 and len(fig.layout.shapes) == 3 and fig.layout.yaxis.type == "log"
    assert fig.layout.shapes[2].x0 == pytest.approx(P.gd_optimal_step(2))                                                   # Linie bei h_opt
    assert len(V.build_step(ev.step_scan(2, 0.3, grid=(0.5, 1.0)), 2, 0.3).data) == 1


def test_methods_chart_has_three_lines_with_slopes_and_the_bound():
    m = ev.methods_rows(0.0, nmax=6)
    fig = V.build_methods(m)
    assert _locked(fig) and len(fig.data) == 4 and fig.data[3].name == "2/(N+1)²" and "Steigung" in fig.data[0].name and fig.layout.xaxis.type == "log" and fig.layout.yaxis.type == "log"
    assert list(fig.data[2].y) == pytest.approx([r["value"] for r in m["rows"]["ogm"]])


def test_proof_chart_is_a_heatmap_of_the_multipliers():
    p = ev.proof_view(P.worst_case("gd", 3, 1.0))
    fig = V.build_proof(p)
    assert _locked(fig) and fig.data[0].type == "heatmap" and fig.data[0].z.shape == (5, 5) and list(fig.data[0].x) == ["*", "0", "1", "2", "3"]
    assert fig.data[0].z.sum() == pytest.approx(p["lam_matrix"].sum())


def test_mu_chart_has_a_line_per_mu():
    rows_ = ev.mu_table()
    fig = V.build_mu(rows_)
    assert _locked(fig) and len(fig.data) == 4 and fig.layout.yaxis.type == "log" and [t.name for t in fig.data][0] == "μ = 0,0"
