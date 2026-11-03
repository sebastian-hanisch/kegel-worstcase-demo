"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier."""

import numpy as np
import pytest

import kwc_constants as C
import kwc_evaluation as ev


def test_formatting_helpers():
    assert ev.de(2.3456, 3) == "2,346" and ev.de(-1.0, 0) == "−1" and ev.sci(1.7e-10) == "1,7·10⁻¹⁰" and ev.sci(2.5e5) == "2,5·10⁵" and ev.sci(float("inf")) == "unendlich"
    assert ev.val(0.0555555556) == "0,05556" and ev.val(6.1e-5) == "6,10·10⁻⁵" and ev.val(1 / 6) == "0,16667"
    assert ev.h_text(1.0) == "1" and ev.h_text(1.75) == "1,75" and ev.h_text(0.5) == "0,5" and ev.step_text(1.0) == "1/L" and ev.step_text(1.75) == "1,75/L"
    assert ev.method_label("gd", 1.0) == "Gradientenverfahren, Schritt 1/L" and ev.method_label("ogm", 1.0) == "Optimiertes Gradientenverfahren (OGM)"
    assert ev.slope([1, 2], [1, 8]) == pytest.approx(3.0) and ev.slope([1], [1]) is None


def _row(**kw):
    base = {"N": 4, "value": 1 / 18, "dual": 1 / 18, "gap": 2.9e-10, "reference": 1 / 18, "deviation": 2.2e-10, "status": "optimal", "iterations": 13, "seconds": 0.05, "rank": 1, "relative_gap": 5e-9, "gram": 6, "rows": 30,
            "variables": 26, "bound": 1 / 18}
    base.update(kw)
    return base


def test_result_verdict_names_value_formula_size_gap_and_rank():
    kind, text = ev.verdict_result(ev.Settings(), _row())
    assert kind == "success"
    for part in ("Gradientenverfahren, Schritt 1/L, N = 4, μ = 0: der scharfe Worst-Case-Wert von f(x_N) − f* ist **0,05556** (in Einheiten von L R²)", "die Formel der Literatur (1/2) max{1/(2Nh + 1), (1 − h)^{2N}} gibt 0,05556 (Abweichung 2,2·10⁻¹⁰)",
                 "Das SDP hat eine 6×6-Matrix, 30 Interpolationszeilen und 13 Iterationen", "Dualwert und Primalwert unterscheiden sich um 2,9·10⁻¹⁰, die Gram-Matrix hat Rang 1."):
        assert part in text, part
    kind, text = ev.verdict_result(ev.Settings(method="ogm", n=4, mu=0.3), _row(reference=None, deviation=None))
    assert "Optimiertes Gradientenverfahren (OGM), N = 4, μ = 0,3" in text and "Formel der Literatur" not in text
    assert "1/(2θ_N²) gibt" in ev.verdict_result(ev.Settings(method="ogm"), _row())[1]


def test_result_verdict_for_a_solver_that_stops_early():
    kind, text = ev.verdict_result(ev.Settings(), _row(status="stalled", relative_gap=1e-6))
    assert kind == "success" and "Status „stalled“" in text and "auf 1,0·10⁻⁶ (relativ)" in text
    kind, text = ev.verdict_result(ev.Settings(), _row(status="stalled", relative_gap=0.01))
    assert kind == "warning" and "**Status des Lösers: stalled**, relative Lücke 1,0·10⁻²" in text and "nicht verlässlich" in text


def test_hand_verdict_names_optimum_rank_and_the_function():
    hand = ev.hand_view()
    kind, text = ev.verdict_hand(hand)
    assert kind == "success"
    for part in ("3×3-Matrix und sechs Interpolationszeilen", "sein Optimum ist **0,16667** (1/6 = 0,16667), die Gram-Matrix hat Rang 1", "f₁(x) = |x|/3 − 1/18 (für |x| ≥ 1/3)", "die Punkte x₀ = 1, x₁ = 2/3 mit f(x₁) = 0,16667",
                 "die Schranke 1/6 ist scharf"):
        assert part in text, part


def _rows(values, ref=True):
    return [{"N": n, "value": v, "reference": v if ref else None, "deviation": 1e-10 if ref else None, "seconds": 0.02 * n, "iterations": 10 + n, "status": "optimal"} for n, v in enumerate(values, start=1)]


def test_rate_verdict_names_decay_slope_formula_and_cost():
    vals = [1 / (4 * n + 2) for n in range(1, 7)]
    kind, text = ev.verdict_rate(_rows(vals), 1.0, 0.0)
    sl = np.polyfit(np.log([4, 5, 6]), np.log(vals[3:]), 1)[0]
    assert kind == "info"
    for part in ("Schritt 1/L, μ = 0: der Worst-Case-Wert fällt von 0,16667 (N = 1) auf 0,03846 (N = 6)", f"ab N = 4 ist {ev.de(sl, 2)}", "Für h = 1/L ist die Formel L R²/(4N + 2): größte Abweichung 1,0·10⁻¹⁰ über N = 1 … 6",
                 "Zeit des Lösers 20 bis 120 ms, 11 bis 16 Iterationen"):
        assert part in text, part
    assert "Die Formel (1/2) max{1/(2Nh + 1)" in ev.verdict_rate(_rows(vals), 1.5, 0.0)[1] and "Formel" not in ev.verdict_rate(_rows(vals, ref=False), 1.0, 0.3)[1]


def _scan(mu=0.0):
    rows = []
    for h, v in ((0.5, 0.1), (1.0, 0.0555), (1.5, 0.0385), (1.75, 0.0501)):
        rows.append({"h": h, "value": v, "reference": v if mu == 0.0 else None, "deviation": 2e-9 if mu == 0.0 else None, "proven": h <= 1.0, "status": "optimal"})
    return rows


def test_step_verdict_names_the_best_step_the_optimum_and_the_status_of_the_formula():
    kind, text = ev.verdict_step(_scan(), 4, 0.0)
    assert kind == "info"
    for part in ("Bei N = 4 Schritten und μ = 0 ist der Worst-Case-Wert am kleinsten bei h = 1,5/L (0,03850); bei h = 1/L sind es 0,05550", "Die beste Schrittlänge der Formel ist h = 1,71/L: größere Schritte als 1/L sind besser",
                 "für h ≤ 1 bewiesen (Drori, Teboulle) und für 1 < h < 2 eine Vermutung (ein Preprint von 2025 behauptet den Beweis, hier nicht nachgeprüft); das SDP bestätigt sie hier mit größter Abweichung 2,0·10⁻⁹ (davon 2 der 4 Schrittlängen über 1/L)"):
        assert part in text, part
    assert "Für μ > 0 gibt es hier keine Vergleichsformel" in ev.verdict_step(_scan(0.3), 4, 0.3)[1]


def _methods():
    def rows(vals, ref):
        return [{"N": n, "value": v, "reference": r, "deviation": None if r is None else 1e-10} for n, (v, r) in enumerate(zip(vals, ref), start=1)]
    gd = [1 / (4 * n + 2) for n in range(1, 13)]
    fgm = [0.16667, 0.1, 0.0661, 0.04683, 0.03489, 0.027, 0.02152, 0.01756, 0.0146, 0.01234, 0.01056, 0.00915]
    ogm = [0.125, 0.06189, 0.03769, 0.02558, 0.01859, 0.01416, 0.01116, 0.00904, 0.00747, 0.00629, 0.00536, 0.00463]
    out = {"gd": rows(gd, gd), "fgm": rows(fgm, [None] * 12), "ogm": rows(ogm, ogm)}
    return {"rows": out, "slopes": {"gd": -0.931, "fgm": -1.494, "ogm": -1.56}}


def test_methods_verdict_gives_values_ratio_slopes_and_bounds():
    kind, text = ev.verdict_methods(_methods(), 0.0)
    assert kind == "info"
    for part in ("Nach N = 12 Schritten: Gradientenverfahren 0,02000, FGM 0,00915, OGM 0,00463 (μ = 0)", "OGM ist das 0,51fache von FGM", "−0,93 (Gradientenverfahren), −1,49 (FGM) und −1,56 (OGM): 1/N gegen 1/N²",
                 "OGM stimmt auf 1,0·10⁻¹⁰ mit 1/(2θ_N²) überein; FGM liegt unter der Schranke 2/(N + 1)² (0,01183 bei N = 12)"):
        assert part in text, part
    assert "stimmt auf" not in ev.verdict_methods(_methods(), 0.3)[1]


def test_proof_verdict_states_the_identity_and_the_count_of_multipliers():
    p = {"bound": 1 / 18, "value": 1 / 18 - 1e-10, "gap": 2.9e-10, "identity": 8.9e-16, "positive": 25, "rows": 30, "lam_min": 7.9e-11, "S_rank": 5, "S_size": 6}
    kind, text = ev.verdict_proof(p)
    assert kind == "success"
    for part in ("liefert die Schranke 0,05556", "Primal- und Dualwert unterscheiden sich um 2,9·10⁻¹⁰", "f_N = Schranke − Σ λ_r s_r − ⟨S, G⟩ (die Summe über alle Zeilen, auch die Normierung 1 − G₀₀)",
                 "kleinster Wert 7,9·10⁻¹¹", "Rang 5 von 6", "gilt sie auf 8,9·10⁻¹⁶", "25 der 30 Multiplikatoren sind positiv", "Mittelpunkt der optimalen Duallösungen"):
        assert part in text, part


def test_mu_verdict_lists_the_values_per_mu_and_the_solver_limit():
    rows = [{"mu": 0.0, "values": {1: {"value": 1 / 6}, 2: {"value": 0.1}}}, {"mu": 0.5, "values": {1: {"value": 0.0714}, 2: {"value": 6.1e-5}}}]
    kind, text = ev.verdict_mu(rows)
    assert kind == "info"
    for part in ("μ = 0: N = 1: 0,16667, N = 2: 0,10000", "μ = 0,5: N = 1: 0,07140, N = 2: 6,10·10⁻⁵", "Je größer μ, desto kleiner der Wert", "Bei μ = 0,5 und N ≥ 6 liegt der Wert unter der Toleranz des Lösers"):
        assert part in text, part
    assert C.MU_NMAX[0.5] == 5
