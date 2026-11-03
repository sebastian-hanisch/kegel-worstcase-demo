"""Konstanten: Regler-Grenzen, feste Reihen der Experimente, Farben und Presets."""

import kwc_pep as PEP

N_MIN, N_MAX, DEFAULT_N = 1, 12, 4
H_CHOICES = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 1.9)
DEFAULT_H = 1.0
MU_CHOICES = (0.0, 0.1, 0.3, 0.5)
DEFAULT_MU = 0.0
DEFAULT_METHOD = "gd"
MU_NMAX = {0.0: 12, 0.1: 12, 0.3: 12, 0.5: 5}          # größtes N je μ: bei μ = 0,5 und N ≥ 6 liegt der Wert unter der Toleranz des Lösers (Status „stalled“)
H_SCAN = tuple(round(0.1 * i, 2) for i in range(1, 20)) + (1.95,)
MU_TABLE_N = (1, 2, 4)                                  # N der μ-Tabelle in Experiment 5
SLOPE_FROM = 4                                          # Steigungen der Raten ab diesem N
TOL = 1e-9

COLORS = {"gd": "#888888", "fgm": "#b08a2e", "ogm": "#2F6B65", "ref": "#c9c9c9", "bound": "#c0504d", "point": "#4c78a8"}
COLOR_OK = "#2F6B65"
COLOR_WARN = "#b08a2e"
COLOR_BAD = "#b34a4a"

_BASE = {"n": DEFAULT_N, "h": DEFAULT_H, "mu": DEFAULT_MU, "method": DEFAULT_METHOD}
PRESETS = {
    "Standardfall: Gradientenverfahren, Schritt 1/L, N = 4": dict(_BASE),
    "N = 1: das Handbeispiel (Wert 1/6)": dict(_BASE, n=1),
    "Größere Schritte sind besser: h = 1,75": dict(_BASE, h=1.75),
    "Beschleunigung: OGM": dict(_BASE, n=8, method="ogm"),
    "Nesterov (FGM) im Vergleich": dict(_BASE, n=8, method="fgm"),
    "Stark konvex: μ = 0,3": dict(_BASE, mu=0.3),
}
PRESET_HELP = {
    "Standardfall: Gradientenverfahren, Schritt 1/L, N = 4": "Das Gradientenverfahren mit der Schrittlänge 1/L nach vier Schritten: der Worst-Case-Wert ist L R²/(4N + 2) = 1/18, auf zehn Stellen aus dem SDP.",
    "N = 1: das Handbeispiel (Wert 1/6)": "Ein einziger Schritt: das SDP hat eine 3×3-Matrix und sechs Interpolationszeilen, der Wert ist 1/6, und die schlechteste Funktion ist eindimensional (Experiment 1).",
    "Größere Schritte sind besser: h = 1,75": "Schrittlänge 1,75/L nach vier Schritten: der Wert 0,05006 ist schlechter als bei h = 1,5 (0,03846), aber die beste Schrittlänge liegt über 1/L (Experiment 3).",
    "Beschleunigung: OGM": "Das optimierte Gradientenverfahren nach acht Schritten: 0,00904 gegen 0,01756 (FGM) und 0,02941 (Gradientenverfahren mit 1/L): der Wert ist 1/(2θ_N²).",
    "Nesterov (FGM) im Vergleich": "Nesterovs schnelles Gradientenverfahren nach acht Schritten: 0,01756, unter der Schranke 2/(N + 1)² = 0,0247, aber doppelt so groß wie OGM.",
    "Stark konvex: μ = 0,3": "Mit der Stark-Konvexität μ = 0,3 fällt der Wert geometrisch: das SDP nutzt die zusätzliche Interpolationsbedingung (Experiment 5).",
}
assert set(PRESETS) == set(PRESET_HELP) and all(p["n"] in range(N_MIN, N_MAX + 1) and p["h"] in H_CHOICES and p["mu"] in MU_CHOICES and p["method"] in PEP.METHODS for p in PRESETS.values())
