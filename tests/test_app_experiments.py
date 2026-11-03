"""Pfade der App, die der Rauchtest nicht erreicht: jedes Verfahren, die fünf Experimente im Seiteninhalt, PDF, Eckfälle aller Regler zugleich (DEMO-PLAYBOOK §1d)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import kwc_constants as C
import kwc_pep as P

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _widget(at, key):
    if key == "method_select":
        return at.sidebar.selectbox(key=key)
    if key == "n_slider":
        return at.sidebar.slider(key=key)
    return at.sidebar.select_slider(key=key)


def _app(**state) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    for key, value in state.items():
        _widget(at, key).set_value(value)
    if state:
        at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _texts(at):
    return " ".join(str(e.value) for e in list(at.success) + list(at.info) + list(at.warning) + list(at.error) + list(at.markdown))


def test_default_page_shows_the_metrics_and_all_experiment_verdicts():
    at = _app()
    values = {m.label: m.value for m in at.metric}
    assert values["Worst-Case-Wert"] == "0,05556" and values["Formel der Literatur"] == "0,05556" and values["Rang der Gram-Matrix"] == "1" and values["Größe des SDP"] == "6×6, 30 Zeilen" and values["Dualwert − Primalwert"].endswith("⁻¹⁰")
    text = _texts(at)
    for part in ("Gradientenverfahren, Schritt 1/L, N = 4, μ = 0: der scharfe Worst-Case-Wert", "Bei N = 1 und Schritt 1/L hat das SDP eine 3×3-Matrix", "Schritt 1/L, μ = 0: der Worst-Case-Wert fällt von 0,16667", "Bei N = 4 Schritten und μ = 0 ist der Worst-Case-Wert am kleinsten bei h = ",
                 "Nach N = 12 Schritten: Gradientenverfahren 0,02000, FGM 0,00915, OGM 0,00463", "Das Dualprogramm liefert die Schranke 0,05556", "Gradientenverfahren mit Schritt 1/L, PEP-Werte: μ = 0: N = 1: 0,16667"):
        assert part in text, part


def test_the_five_experiments_the_limits_table_and_the_formulation_are_on_the_page():
    at = _app()
    headings = [s.value for s in at.subheader]
    assert [h.split(":")[0] for h in headings if h.startswith("🔬")] == [f"🔬 Experiment {i}" for i in range(1, 6)] and "🚧 Wo die Annahmen enden" in headings
    assert [e.label for e in at.expander] == ["So funktioniert es", "📐 Mathematische Formulierung"]
    assert len(at.dataframe) == 6 and not at.error


def test_the_experiment_tables_have_the_expected_rows():
    at = _app()
    frames = [d.value for d in at.dataframe]
    assert len(frames[0]) == 6 and frames[0].iloc[:, 1].tolist().count("0") == 4                  # sechs Zeilen des Handbeispiels, vier aktive
    assert list(frames[2]["N"]) == list(range(1, 13)) and len(frames[3]) == 20 and list(frames[4]["N"]) == list(range(1, 13)) and len(frames[5]) == 4


def test_pdf_button_produces_a_download():
    at = _app()
    at.button(key="pdf_make").click().run()
    assert not at.exception
    assert at.session_state["pdf_made"][1][:4] == b"%PDF"


@pytest.mark.parametrize("method", P.METHODS)
def test_every_method_renders(method):
    at = _app(method_select=method)
    assert not at.error and any("der scharfe Worst-Case-Wert" in s.value for s in list(at.success) + list(at.warning))


@pytest.mark.parametrize("mu", C.MU_CHOICES)
def test_every_mu_renders(mu):
    at = _app(mu_select=mu)
    assert not at.error and not at.exception


@pytest.mark.parametrize("h", [C.H_CHOICES[0], C.H_CHOICES[-1]])
def test_the_extreme_step_lengths_render(h):
    at = _app(h_select=h)
    assert not at.error and not at.exception


@pytest.mark.parametrize("n", [C.N_MIN, C.N_MAX])
def test_the_extreme_numbers_of_steps_render(n):
    at = _app(n_slider=n)
    assert not at.error and {m.label: m.value for m in at.metric}["Größe des SDP"].startswith(f"{n + 2}×{n + 2}")


def test_the_corner_with_a_big_mu_and_many_steps_renders_with_a_note():
    at = _app(mu_select=0.5, n_slider=C.N_MAX, method_select="gd")
    assert not at.error and not at.exception
    assert any("Status" in s.value for s in list(at.success) + list(at.warning))
