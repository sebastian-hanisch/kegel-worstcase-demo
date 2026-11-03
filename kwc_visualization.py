"""Plotly-Abbildungen. Alle 2D-Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

import kwc_constants as C
import kwc_pep as PEP


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=380, legend_below=True):
    legend = dict(orientation="h", yanchor="top", y=-0.22, x=0) if legend_below else dict(orientation="h", yanchor="bottom", y=1.02, x=0)
    fig.update_layout(height=height + (40 if legend_below else 0), margin=dict(l=10, r=10, t=50, b=10), legend=legend, plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _positive(values, floor=1e-12):
    return [max(float(v), floor) for v in values]


# ----------------------------------------------------------------------------- Hauptabbildung und Experiment 2: Wert gegen N

def build_rate(rows: list, label: str, color: str, selected_n: int | None = None, loglog: bool = False, ref_label: str = "Formel der Literatur") -> go.Figure:
    ns = [r["N"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ns, y=_positive(r["value"] for r in rows), mode="lines+markers", name=label, line=dict(color=color, width=3), hovertemplate="N = %{x}: %{y:.6f}<extra></extra>"))
    ref = [(r["N"], r["reference"]) for r in rows if r["reference"] is not None]
    if ref:
        fig.add_trace(go.Scatter(x=[n for n, _ in ref], y=_positive(v for _, v in ref), mode="lines", name=ref_label, line=dict(color=C.COLORS["ref"], width=2, dash="dot"), hoverinfo="skip"))
    if selected_n is not None and selected_n in ns:
        r = rows[ns.index(selected_n)]
        fig.add_trace(go.Scatter(x=[selected_n], y=[max(r["value"], 1e-12)], mode="markers", name=f"N = {selected_n}", marker=dict(size=13, color=C.COLOR_BAD, symbol="circle-open", line=dict(width=3)), hoverinfo="skip"))
    fig.update_xaxes(title_text="Schritte N", type="log" if loglog else "linear", tickvals=ns if len(ns) <= 12 else None)
    fig.update_yaxes(title_text="Worst-Case-Wert f(x_N) − f* (in L R²)", type="log")
    return _base(fig)


# ----------------------------------------------------------------------------- Experiment 1: die eindimensionale schlechteste Funktion

def build_hand(hand: dict) -> go.Figure:
    xs, ys = hand["curve"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="f₁(x) = |x|/3 − 1/18 (|x| ≥ 1/3), sonst x²/2", line=dict(color=C.COLORS["point"], width=3)))
    fig.add_trace(go.Scatter(x=hand["x_hand"], y=hand["f_hand"], mode="markers+text", text=["x* (f = 0)", "x₀ = 1", "x₁ = 2/3 (f = 1/6)"], textposition=["top right", "top left", "bottom right"], name="die drei Punkte",
                             marker=dict(size=11, color=C.COLOR_BAD), hovertemplate="x = %{x:.3f}, f = %{y:.4f}<extra></extra>"))
    fig.add_annotation(x=hand["x_hand"][2], y=hand["f_hand"][2], ax=hand["x_hand"][1], ay=hand["f_hand"][1], xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=2, arrowcolor="#888888")
    fig.update_xaxes(title_text="x")
    fig.update_yaxes(title_text="f(x)")
    return _base(fig)


# ----------------------------------------------------------------------------- Experiment 3: Schrittlänge

def build_step(rows: list, N: int, mu: float) -> go.Figure:
    hs = [r["h"] for r in rows]
    fig = go.Figure()
    fig.add_vrect(x0=min(hs), x1=1.0, fillcolor="#2F6B65", opacity=0.07, line_width=0, annotation_text="Formel bewiesen", annotation_position="top left")
    fig.add_vrect(x0=1.0, x1=max(hs), fillcolor="#b08a2e", opacity=0.07, line_width=0, annotation_text="Vermutung, numerisch bestätigt", annotation_position="top right")
    fig.add_trace(go.Scatter(x=hs, y=_positive(r["value"] for r in rows), mode="lines+markers", name="PEP-Wert (SDP)", line=dict(color=C.COLORS["ogm"], width=3), hovertemplate="h = %{x}: %{y:.6f}<extra></extra>"))
    if mu == 0.0:
        grid = np.linspace(min(hs), max(hs), 120)
        fig.add_trace(go.Scatter(x=grid, y=[0.5 / (2 * N * h + 1) for h in grid], mode="lines", name="1/(2(2Nh + 1)): zu kleine Schritte", line=dict(color=C.COLORS["ref"], dash="dot", width=2), hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=grid, y=_positive(0.5 * (1.0 - h) ** (2 * N) for h in grid), mode="lines", name="(1 − h)^{2N}/2: Überschießen", line=dict(color=C.COLORS["ref"], dash="dash", width=2), hoverinfo="skip"))
        hopt = PEP.gd_optimal_step(N)
        fig.add_vline(x=hopt, line=dict(color=C.COLOR_BAD, dash="dot"), annotation_text=f"h_opt = {hopt:.2f}", annotation_position="bottom")
    fig.update_xaxes(title_text="Schrittlänge h (in Einheiten von 1/L)", range=[min(hs) - 0.05, max(hs) + 0.05])
    fig.update_yaxes(title_text=f"Worst-Case-Wert nach N = {N} Schritten", type="log")
    return _base(fig)


# ----------------------------------------------------------------------------- Experiment 4: Beschleunigung

def build_methods(m: dict) -> go.Figure:
    fig = go.Figure()
    for method in PEP.METHODS:
        rows = m["rows"][method]
        sl = m["slopes"][method]
        label = f"{PEP.METHOD_NAMES[method]} (Steigung {sl:.2f})".replace(".", ",") if sl is not None else PEP.METHOD_NAMES[method]
        fig.add_trace(go.Scatter(x=[r["N"] for r in rows], y=_positive(r["value"] for r in rows), mode="lines+markers", name=label, line=dict(color=C.COLORS[method], width=3), hovertemplate="N = %{x}: %{y:.6f}<extra></extra>"))
    ns = [r["N"] for r in m["rows"]["gd"]]
    fig.add_trace(go.Scatter(x=ns, y=[PEP.fgm_bound(n) for n in ns], mode="lines", name="2/(N+1)²", line=dict(color=C.COLORS["ref"], dash="dot", width=2), hoverinfo="skip"))
    fig.update_xaxes(title_text="Schritte N", type="log", tickvals=ns)
    fig.update_yaxes(title_text="Worst-Case-Wert f(x_N) − f* (in L R²)", type="log")
    return _base(fig)


# ----------------------------------------------------------------------------- Experiment 5: der Beweis

def build_proof(p: dict) -> go.Figure:
    names = p["names"]
    fig = go.Figure(go.Heatmap(z=p["lam_matrix"], x=names, y=names, colorscale=[[0, "#f4f4f4"], [1, "#2F6B65"]], colorbar=dict(title="λ_ij"), hovertemplate="i = %{y}, j = %{x}: λ = %{z:.4f}<extra></extra>"))
    fig.update_xaxes(title_text="Punkt j (die Zeile verwendet f_j, g_j)", type="category")
    fig.update_yaxes(title_text="Punkt i", type="category", autorange="reversed")
    return _base(fig, legend_below=False)


def build_mu(rows: list) -> go.Figure:
    fig = go.Figure()
    for r in rows:
        ns = list(r["values"])
        fig.add_trace(go.Scatter(x=ns, y=_positive(v["value"] for v in r["values"].values()), mode="lines+markers", name=f"μ = {str(r['mu']).replace('.', ',')}", line=dict(width=3), hovertemplate="N = %{x}: %{y:.6f}<extra></extra>"))
    fig.update_xaxes(title_text="Schritte N", type="category")
    fig.update_yaxes(title_text="Worst-Case-Wert (Gradientenverfahren, Schritt 1/L)", type="log")
    return _base(fig)
