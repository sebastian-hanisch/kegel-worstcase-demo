"""Worst-Case-Raten als SDP – Performance Estimation mit dem PSD-Löser der Reihe - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 11 der Reihe „Konvexe und konische Optimierung“: Der Worst-Case eines Verfahrens erster Ordnung über alle L-glatten konvexen Funktionen ist ein semidefinites Programm: die Interpolationsbedingungen machen
„für alle Funktionen“ endlich, der primal-duale Löser aus Stück 6 findet die scharfe Konstante, und das Dual ist ein numerisch geprüftes Beweiszertifikat. Die Demo rechnet es für das Gradientenverfahren, Nesterovs FGM und das optimierte Gradientenverfahren.
Aufbau und Regeln: DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import pandas as pd
import streamlit as st

import kwc_constants as C
import kwc_pep as PEP
from kwc_evaluation import (
    Settings,
    de,
    h_text,
    hand_view,
    main_result,
    method_label,
    methods_rows,
    mu_table,
    proof_view,
    rate_rows,
    result_row,
    sci,
    step_scan,
    val,
    verdict_hand,
    verdict_methods,
    verdict_mu,
    verdict_proof,
    verdict_rate,
    verdict_result,
    verdict_step,
)
from kwc_layout import dunkles_design, kopf_css
from kwc_pdf_export import build_pdf
from kwc_presets import bounds, init_session_state_defaults, load_permalink_settings, sidebar_presets, sync_query_params
from kwc_visualization import build_hand, build_methods, build_mu, build_proof, build_rate, build_step

st.set_page_config(page_title="Worst-Case-Raten als SDP – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _rates(method, h, mu):
    return rate_rows(method, h, mu)


@st.cache_data(show_spinner=False)
def _result(method, n, h, mu):
    res = main_result(Settings(method, n, h, mu))
    return res, result_row(res, PEP.reference(method, n, h, mu)), proof_view(res)


@st.cache_data(show_spinner=False)
def _hand():
    return hand_view()


@st.cache_data(show_spinner=False)
def _scan(n, mu):
    return step_scan(n, mu)


@st.cache_data(show_spinner=False)
def _methods(mu):
    return methods_rows(mu)


@st.cache_data(show_spinner=False)
def _mu_table():
    return mu_table()


def verdict(pair):
    kind, text = pair
    getattr(st, kind)(text)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in kwc_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🧮 Worst-Case-Raten als SDP – Performance Estimation mit dem PSD-Löser der Reihe")
st.markdown(
    """
**Ein Worst-Case-Beweis ist ein semidefinites Programm.** „Für alle L-glatten konvexen Funktionen und alle Startpunkte“ klingt nach unendlich vielen Fällen. Die **Interpolationsbedingungen** machen daraus ein endliches Programm:
Punkte `(xᵢ, gᵢ, fᵢ)` lassen sich genau dann durch eine L-glatte konvexe Funktion verbinden, wenn für jedes Paar `fᵢ − fⱼ − ⟨gⱼ, xᵢ − xⱼ⟩ − ‖gᵢ − gⱼ‖²/(2L) ≥ 0` gilt. Mit der Gram-Matrix von `(x₀ − x*, g₀, …, g_N)` als PSD-Variable ist
alles **linear**, und „maximiere `f(x_N) − f*` unter `‖x₀ − x*‖ ≤ 1`“ ist ein SDP. Der primal-duale Löser aus Stück 6 findet für das Gradientenverfahren mit Schritt 1/L **`L R²/(4N + 2)` auf neun Stellen**; das Dual ist ein numerisch geprüftes Beweiszertifikat.
Stück 11 der Reihe „Konvexe und konische Optimierung“; es schließt den Strang Erste Ordnung: dort Schrittzahlen gemessen, hier Raten hergeleitet.

Weiter unten: **fünf Experimente** (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert es", expanded=True):
    st.markdown(
        """
1. **Das Verfahren:** Gradientenverfahren `x_{i+1} = x_i − (h/L) ∇f(x_i)`, Nesterovs schnelles Verfahren (FGM) oder das optimierte Gradientenverfahren (OGM): die Iterierten sind **lineare** Kombinationen von `x₀` und den bisherigen Gradienten.
2. **Das SDP:** Variablen sind die Funktionswerte `fᵢ` und die Gram-Matrix `G` von `(x₀ − x*, g₀, …, g_N)` (PSD, `(N + 2) × (N + 2)`). Nebenbedingungen: eine Interpolationszeile je Paar von Punkten (mit dem Optimalpunkt `x*`, `g* = 0`, `f* = 0`) und `G₀₀ ≤ 1`. Ziel: `max f_N`.
3. **Der Löser:** der primal-duale innere Punkt aus Stück 6 (Nesterov–Todd-Skalierung, PSD-Kegel), ohne strikt zulässigen Start; er liefert auch die Dualvariablen.
4. **Der Beweis:** das Dual besteht aus nichtnegativen Multiplikatoren `λ` der Interpolationszeilen und einer PSD-Matrix `S`; es gilt `f_N = Schranke − Σ λ_r s_r − ⟨S, G⟩` für **jedes** `(f, G)`, und alle Terme sind nichtnegativ.
5. **Eindimensional:** beim Gradientenverfahren und bei FGM hat die optimale Gram-Matrix Rang 1: die schlechteste Funktion ist eine Funktion einer Variablen (Experiment 1); beim OGM ist die schlechteste Funktion nicht eindeutig.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    method = st.selectbox("Verfahren", PEP.METHODS, key="method_select", format_func=PEP.METHOD_NAMES.get, help="Das Verfahren für Ergebnis und Beweis; Experiment 4 vergleicht alle drei.")
    n = st.slider("Schritte N", *bounds("n_slider"), key="n_slider", help="Wie viele Schritte das Verfahren macht; das SDP hat eine (N + 2) × (N + 2)-Matrix.")
    h = st.select_slider("Schrittlänge h (Gradientenverfahren)", options=C.H_CHOICES, key="h_select", format_func=h_text, help="Schritt h/L; FGM und OGM haben feste Koeffizienten. Bewiesen ist die Formel für h ≤ 1, für 1 < h < 2 ist sie nach unserer Messung eine Vermutung (ein Preprint von 2025 behauptet den Beweis, hier nicht nachgeprüft).")
    mu = st.select_slider("Stark-Konvexität μ", options=C.MU_CHOICES, key="mu_select", format_func=lambda x: h_text(x) if x else "0",
                          help="μ = 0: nur konvex und L-glatt; μ > 0: zusätzlich μ-stark konvex. Bei μ = 0,5 endet die Rate bei N = 5 (der Wert liegt dann unter der Toleranz des Lösers).")

sync_query_params({"method_select": method, "n_slider": int(n), "h_select": float(h), "mu_select": float(mu)})

settings = Settings(method, int(n), float(h), float(mu))
res, row, proof = _result(settings.method, settings.n, settings.h, settings.mu)
rows = _rates(settings.method, settings.h, settings.mu)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
m1, m2, m3 = st.columns(3)
m4, m5 = st.columns(2)
m1.metric("Worst-Case-Wert", val(row["value"]), help="Das scharfe Maximum von f(x_N) − f* über alle L-glatten (μ-stark) konvexen f mit ‖x₀ − x*‖ ≤ 1, in Einheiten von L R².")
m2.metric("Formel der Literatur", val(row["reference"]) if row["reference"] is not None else "–", help="Gradientenverfahren: (1/2) max{1/(2Nh + 1), (1 − h)^{2N}}; OGM: 1/(2θ_N²); sonst keine Vergleichsformel.")
m3.metric("Dualwert − Primalwert", sci(abs(row["gap"])), help="Der Dualwert ist die Schranke des Beweiszertifikats (numerisch geprüft); beide stimmen überein.")
m4.metric("Rang der Gram-Matrix", str(row["rank"]), help="Rang 1: die schlechteste Funktion ist eindimensional (Gradientenverfahren, FGM). Beim OGM gibt es viele schlechteste Funktionen: der Löser liefert den Mittelpunkt der optimalen Menge, Rang N + 1.")
m5.metric("Größe des SDP", f"{row['gram']}×{row['gram']}, {row['rows']} Zeilen", help=f"{row['iterations']} Iterationen, {de(1e3 * row['seconds'], 0)} ms.")
st.plotly_chart(build_rate(rows, method_label(settings.method, settings.h), C.COLORS[settings.method], settings.n, ref_label="Formel der Literatur"), width="stretch", key="main_chart")
st.caption("Der Worst-Case-Wert gegen die Zahl der Schritte (logarithmisch); der Kreis markiert das gewählte N.")

if st.button("📄 PDF des aktuellen Ergebnisses erstellen", key="pdf_make"):
    st.session_state["pdf_made"] = (settings, build_pdf(settings, row, rows))
made = st.session_state.get("pdf_made")
if made and made[0] == settings:
    st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="worst-case-sdp.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Wie schnell ist das Verfahren im schlechtesten Fall?")
verdict(verdict_result(settings, row))

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Das PEP für N = 1 von Hand")
st.caption("Ein Schritt des Gradientenverfahrens mit Schritt 1/L: drei Punkte (der Optimalpunkt `*`, `x₀`, `x₁`), sechs Interpolationszeilen. Rechts die Zeilen an den Daten der Funktion `f₁` und am Optimum des SDP.")
hand = _hand()
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(build_hand(hand), width="stretch", key="hand_chart")
with c2:
    st.dataframe(pd.DataFrame([{"Paar (i, j)": f"({r['i']}, {r['j']})", "Zeile an f₁": sci(r["hand"]) if abs(r["hand"]) > 1e-12 else "0", "Zeile am SDP-Optimum": sci(r["optimum"]) if abs(r["optimum"]) > 1e-9 else "0"} for r in hand["rows"]]),
                 hide_index=True, width="stretch")
    st.dataframe(pd.DataFrame(hand["result"].G, columns=["x₀ − x*", "g₀", "g₁"], index=["x₀ − x*", "g₀", "g₁"]).map(lambda v: de(v, 4)), width="stretch")
verdict(verdict_hand(hand))

st.markdown("---")

st.subheader("🔬 Experiment 2: Gradientenverfahren – die Rate")
st.caption("Der Worst-Case-Wert des Gradientenverfahrens (Schrittlänge und μ der Seitenleiste) für N = 1 bis 12, log-log, neben der Formel; darunter Größe und Zeit des SDP.")
gd_rows = _rates("gd", settings.h, settings.mu)
st.plotly_chart(build_rate(gd_rows, f"Gradientenverfahren, Schritt {h_text(settings.h)}/L", C.COLORS["gd"], settings.n, loglog=True), width="stretch", key="gd_chart")
st.dataframe(pd.DataFrame([{"N": r["N"], "Worst-Case-Wert": val(r["value"]), "Formel": val(r["reference"]) if r["reference"] is not None else "–", "Abweichung": sci(abs(r["deviation"])) if r["deviation"] is not None else "–",
                            "Gram-Matrix": f"{r['gram']}×{r['gram']}", "Interpolationszeilen": r["rows"], "Variablen": r["variables"], "Iterationen": r["iterations"], "Zeit (ms)": de(1e3 * r["seconds"], 0), "Status": r["status"]}
                           for r in gd_rows]), hide_index=True, width="stretch")
verdict(verdict_rate(gd_rows, settings.h, settings.mu))

st.markdown("---")

st.subheader("🔬 Experiment 3: Die Schrittlänge")
st.caption(f"Der Worst-Case-Wert des Gradientenverfahrens nach N = {settings.n} Schritten für 20 Schrittlängen zwischen 0,1/L und 1,95/L (μ der Seitenleiste). Die Formel hat zwei Terme: zu kleine Schritte (erster) und Überschießen (zweiter).")
scan = _scan(settings.n, settings.mu)
st.plotly_chart(build_step(scan, settings.n, settings.mu), width="stretch", key="step_chart")
st.dataframe(pd.DataFrame([{"h": h_text(r["h"]), "Worst-Case-Wert": val(r["value"]), "Formel": val(r["reference"]) if r["reference"] is not None else "–", "Abweichung": sci(abs(r["deviation"])) if r["deviation"] is not None else "–",
                            "Status der Formel": "bewiesen" if r["proven"] else "Vermutung"} for r in scan]), hide_index=True, width="stretch")
verdict(verdict_step(scan, settings.n, settings.mu))

st.markdown("---")

st.subheader("🔬 Experiment 4: Beschleunigung")
st.caption("Gradientenverfahren (Schritt 1/L), Nesterovs FGM und das optimierte Gradientenverfahren OGM: die Worst-Case-Werte für N = 1 bis 12 (μ der Seitenleiste).")
mrows = _methods(settings.mu)
st.plotly_chart(build_methods(mrows), width="stretch", key="methods_chart")
st.dataframe(pd.DataFrame([{"N": a["N"], "Gradientenverfahren": val(a["value"]), "FGM": val(b["value"]), "OGM": val(c["value"]), "OGM / FGM": de(c["value"] / b["value"], 3),
                            "1/(2θ_N²)": val(PEP.ogm_formula(a["N"])), "2/(N+1)²": val(PEP.fgm_bound(a["N"]))} for a, b, c in zip(mrows["rows"]["gd"], mrows["rows"]["fgm"], mrows["rows"]["ogm"])]), hide_index=True, width="stretch")
verdict(verdict_methods(mrows, settings.mu))

st.markdown("---")

st.subheader("🔬 Experiment 5: Der Beweis")
st.caption("Das Dual des gewählten SDP: Multiplikatoren `λᵢⱼ` der Interpolationszeilen (Heatmap) und die Identität, die aus ihnen einen Beweis macht. Darunter der μ-stark konvexe Fall.")
st.plotly_chart(build_proof(proof), width="stretch", key="proof_chart")
verdict(verdict_proof(proof))
mtable = _mu_table()
c3, c4 = st.columns(2)
with c3:
    st.plotly_chart(build_mu(mtable), width="stretch", key="mu_chart")
with c4:
    st.dataframe(pd.DataFrame([{"μ": h_text(r["mu"]) if r["mu"] else "0", **{f"N = {nn}": val(v["value"]) for nn, v in r["values"].items()}} for r in mtable]), hide_index=True, width="stretch")
verdict(verdict_mu(mtable))

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Formel `(1/2) max{1/(2Nh + 1), (1 − h)^{2N}}` ist bewiesen** | Nur für `h ≤ 1` (Drori, Teboulle). Für `1 < h < 2` ist sie nach unserer Messung eine Vermutung (so Taylor, Hendrickx und Glineur 2017; ein Preprint von 2025 behauptet den Beweis, hier nicht nachgeprüft); das SDP bestätigt sie hier numerisch (Experiment 3), ersetzt aber keinen Beweis. Für `h > 2` wächst der Wert über 1/2 hinaus (das Verfahren divergiert), bei `h = 2` bleibt er bei 1/2; die App begrenzt den Regler auf `h < 2`. | – |
| **Jedes Verfahren erster Ordnung lässt sich so analysieren** | Nur solche, deren Iterierte **feste lineare** Kombinationen der Gradienten sind (Gradientenverfahren, FGM, OGM). Adaptive Schrittlängen, Liniensuche und proximale Schritte brauchen ein anderes Programm. | Stück 16 |
| **Der Beweis ist ablesbar** | Das Zertifikat ist numerisch geprüft (Gleitkomma), kein formaler Beweis. Der Löser liefert den Mittelpunkt der Menge optimaler Duallösungen: die meisten Multiplikatoren sind positiv (Experiment 5). Eine kurze Beweisskizze wie in den Arbeiten bräuchte eine dünne Duallösung. | – |
| **Die Raten sind 1/N und 1/N²** | Das sind Grenzwerte. Gemessen sind für N = 4 bis 12 die Steigungen −0,93 (Gradientenverfahren), −1,49 (FGM) und −1,56 (OGM): OGM liegt bei N = 12 etwa bei der Hälfte von FGM (0,507; bei N = 1 sind es 0,750), aber der asymptotische Abstand zwischen 1/N und 1/N² zeigt sich erst bei größerem N. | – |
| **Der Wert hängt von der Dimension ab** | Nein, ab Dimension N + 2: das SDP ist dort dimensionsunabhängig (die Gram-Matrix hat Größe N + 2); in kleineren Dimensionen kann der Worst-Case kleiner sein. Beim Gradientenverfahren und bei FGM hat die schlechteste Funktion Rang 1, ist also eindimensional; beim OGM ist sie nicht eindeutig (die Matrix des Lösers hat Rang N + 1; eine Nachprüfung fand auch dort eine eindimensionale schlechteste Funktion). | – |
| **Der Löser rechnet beliebig genau** | Bei sehr kleinen Werten (μ = 0,5 und N ≥ 6, h nahe 0) stoppt der Löser vor der Toleranz („stalled“), die letzten Stellen sind dann nicht verlässlich; die App meldet es. | – |
| **Die Zeit bleibt klein** | Das dichte KKT-System wächst mit N: 20 ms bei N = 1 bis 4, rund 0,3 s bei N = 12 (Experiment 2); bei N = 16 etwa eine Sekunde. Zeiten hängen vom Rechner ab und werden nur in weiten Bändern getestet. | – |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Das Performance-Estimation-Problem.** Gesucht ist $\max f(x_N) - f_*$ über alle $f \in \mathcal{F}_{\mu,L}$ und alle $x_0$ mit $\|x_0 - x_*\|^2 \le R^2$ ($L = R = 1$), wobei $x_i$ die Iterierten des Verfahrens sind.
Mit den Abfragepunkten $(x_i, g_i, f_i)$ und dem Optimalpunkt $(x_*, 0, f_*)$ ist es nach dem Interpolationssatz äquivalent zu: finde $f_i$ und eine Matrix $G \succeq 0$ mit
$$f_i - f_j - \langle g_j, x_i - x_j\rangle - \tfrac{1}{2(1-\mu/L)}\Big(\tfrac1L\|g_i - g_j\|^2 + \mu\|x_i - x_j\|^2 - \tfrac{2\mu}{L}\langle g_j - g_i, x_j - x_i\rangle\Big) \ge 0 \quad \forall i \ne j,$$
wobei $x_i, g_i$ lineare Kombinationen der Basis $(x_0 - x_*, g_0, \dots, g_N)$ sind und $\langle a, b\rangle = a^\top G b$. Für $\mu = 0$ ist die Zeile $f_i - f_j - \langle g_j, x_i - x_j\rangle - \tfrac{1}{2L}\|g_i - g_j\|^2 \ge 0$.
Alles ist linear in $(f, G)$: ein SDP in der Kegelform des Kerns, $\min c^\top z$ mit $s = b - Az \in \mathbb{R}^{m}_+ \times \mathbb{S}^{N+2}_+$.

**Gradientenverfahren.** $x_{i+1} = x_i - \tfrac{h}{L}\nabla f(x_i)$. Der Worst-Case ist $\tfrac{LR^2}{2}\max\{\tfrac{1}{2Nh+1}, (1-h)^{2N}\}$ (für $h \le 1$ bewiesen; für $1 < h < 2$ nach unserer Messung eine Vermutung, ein Preprint von 2025 behauptet den Beweis, hier nicht nachgeprüft); für $h = 1/L$ gleich $\tfrac{LR^2}{4N+2}$.
Die beiden Terme gehören zu den eindimensionalen Funktionen $f_1(x) = \tfrac{|x|}{2Nh+1} - \tfrac{1}{2(2Nh+1)^2}$ für $|x| \ge \tfrac{1}{2Nh+1}$ (sonst $x^2/2$) und $f_2(x) = x^2/2$.
**FGM:** $y_{i+1} = x_i - \tfrac1L\nabla f(x_i)$, $\theta_{i+1} = \tfrac{1+\sqrt{4\theta_i^2+1}}{2}$, $x_{i+1} = y_{i+1} + \tfrac{\theta_i - 1}{\theta_{i+1}}(y_{i+1} - y_i)$ (Ergebnis $y_N$); Schranke $\tfrac{2LR^2}{(N+1)^2}$ (Beck, Teboulle).
**OGM:** wie FGM mit dem zusätzlichen Term $\tfrac{\theta_i}{\theta_{i+1}}(y_{i+1} - x_i)$ und $\theta_N = \tfrac{1+\sqrt{8\theta_{N-1}^2+1}}{2}$ im letzten Schritt (Ergebnis $x_N$); der Worst-Case ist $\tfrac{LR^2}{2\theta_N^2}$.

**Das Dual und der Beweis.** Das Dualprogramm hat nichtnegative Multiplikatoren $\lambda_{ij}$ der Interpolationszeilen, einen Multiplikator $\tau$ der Normierung $R^2 - G_{00} \ge 0$ und eine Matrix $S \succeq 0$. Aus der dualen Zulässigkeit folgt für jedes $(f, G)$ die Identität
$$f_N = \tau R^2 - \sum_{i \ne j} \lambda_{ij}\, s_{ij}(f, G) - \langle S, G\rangle - \tau\,(R^2 - G_{00}),$$
und für zulässige $(f, G)$ sind alle abgezogenen Terme nichtnegativ: $f_N \le \tau R^2$. $\tau$ ist die Schranke des Zertifikats (bei exakter Dualzulässigkeit bewiesen, hier numerisch bis auf die Genauigkeit des Lösers geprüft); im Optimum ist sie gleich dem Primalwert.

**Literatur.** A. B. Taylor, J. M. Hendrickx, F. Glineur: *Smooth strongly convex interpolation and exact worst-case performance of first-order methods*, Mathematical Programming 161 (1–2), 2017, S. 307–345 (Interpolationssatz, Vermutung für das Gradientenverfahren, eindimensionale schlechteste Funktionen). Für `1 < h < 2` behauptet ein arXiv-Preprint von 2025 (Jungbin Kim, arXiv:2412.04427) einen Beweis; hier nicht gelesen und nicht nachgeprüft.
Y. Drori, M. Teboulle: *Performance of first-order methods for smooth convex minimization: a novel approach*, Mathematical Programming 145 (1–2), 2014, S. 451–482 (das PEP; die Formel für 0 ≤ h ≤ 1).
D. Kim, J. A. Fessler: *Optimized first-order methods for smooth convex minimization*, Mathematical Programming 159, 2016 (OGM; die Formel 1/(2θ_N²) ist hier nicht am Original nachgelesen, die SDP-Werte bestätigen sie auf neun Stellen).
A. Beck, M. Teboulle: *A fast iterative shrinkage-thresholding algorithm for linear inverse problems*, SIAM Journal on Imaging Sciences 2009 (Schranke für FGM; hier über Taylor, Hendrickx, Glineur belegt). Beweise der Arbeiten nicht nachgelesen.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
