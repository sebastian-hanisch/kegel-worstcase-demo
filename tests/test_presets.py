"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; Permalink verwirft ungültige Werte; die Aussagen der Hilfetexte."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import kwc_constants as C
import kwc_evaluation as ev
import kwc_pep as P
import kwc_presets as PR

APP = str(Path(__file__).resolve().parent.parent / "app.py")
DEFAULT = "Standardfall: Gradientenverfahren, Schritt 1/L, N = 4"


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(PR.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_control_bounds():
    for preset in C.PRESETS.values():
        assert preset["method"] in P.METHODS and C.N_MIN <= preset["n"] <= C.N_MAX and preset["h"] in C.H_CHOICES and preset["mu"] in C.MU_CHOICES


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS[DEFAULT]) == ev.Settings() and next(iter(C.PRESETS)) == DEFAULT


def test_setting_specs_cover_every_preset_key():
    assert set(PR.PRESET_KEYS.values()) == set(PR.SETTING_SPECS)


def _value(name):
    s = ev.Settings(**C.PRESETS[name])
    return P.worst_case(s.method, s.n, s.h, s.mu).value


def test_the_help_texts_name_what_the_presets_produce():
    assert _value(DEFAULT) == pytest.approx(1 / 18, abs=1e-8) and "1/18" in C.PRESET_HELP[DEFAULT]
    assert _value("N = 1: das Handbeispiel (Wert 1/6)") == pytest.approx(1 / 6, abs=1e-8) and "1/6" in C.PRESET_HELP["N = 1: das Handbeispiel (Wert 1/6)"]
    assert _value("Größere Schritte sind besser: h = 1,75") == pytest.approx(0.05006, abs=5e-6) and "0,05006" in C.PRESET_HELP["Größere Schritte sind besser: h = 1,75"]
    ogm, fgm = _value("Beschleunigung: OGM"), _value("Nesterov (FGM) im Vergleich")
    assert ogm == pytest.approx(0.00904, abs=5e-6) and fgm == pytest.approx(0.01756, abs=5e-6) and ogm < fgm and "0,00904" in C.PRESET_HELP["Beschleunigung: OGM"] and "0,01756" in C.PRESET_HELP["Nesterov (FGM) im Vergleich"]
    assert _value("Stark konvex: μ = 0,3") < 1 / 18


def test_permalink_takes_valid_values_and_drops_invalid_ones():
    at = AppTest.from_file(APP, default_timeout=300)
    at.query_params["method"] = "ogm"
    at.query_params["n"] = "999"            # über der Grenze: wird begrenzt
    at.query_params["h"] = "1.75"
    at.query_params["mu"] = "0.7"           # kein erlaubter Wert: bleibt Standard
    at.run()
    assert not at.exception
    assert at.sidebar.selectbox(key="method_select").value == "ogm" and at.sidebar.slider(key="n_slider").value == C.N_MAX
    assert at.session_state["h_select"] == 1.75 and at.session_state["mu_select"] == C.DEFAULT_MU
    bad = AppTest.from_file(APP, default_timeout=300)
    bad.query_params["method"] = "newton"
    bad.query_params["n"] = "abc"
    bad.run()
    assert not bad.exception and bad.sidebar.selectbox(key="method_select").value == C.DEFAULT_METHOD and bad.sidebar.slider(key="n_slider").value == C.DEFAULT_N
