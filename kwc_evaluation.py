"""Kennzahlen, Experimente und Verdikte. Jedes Verdikt ist eine reine Funktion (Art, Text); die App ruft st.<Art>(Text)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from threadpoolctl import threadpool_limits

import kwc_constants as C
import kwc_pep as PEP


@dataclass(frozen=True)
class Settings:
    method: str = C.DEFAULT_METHOD
    n: int = C.DEFAULT_N
    h: float = C.DEFAULT_H
    mu: float = C.DEFAULT_MU


# ----------------------------------------------------------------------------- Formatierung

def de(x: float, digits: int = 2) -> str:
    """Zahl mit deutschem Dezimalkomma und typografischem Minuszeichen."""
    return f"{x:.{digits}f}".replace(".", ",").replace("-", "−")


_SUPER = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def sci(x: float, digits: int = 1) -> str:
    """Zahl in Zehnerpotenz-Schreibweise, z. B. 2,0·10⁻¹⁰."""
    if x == 0.0 or not math.isfinite(x):
        return de(x, digits) if math.isfinite(x) else "unendlich"
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10.0 ** e
    if round(abs(m), digits) >= 10.0:
        m, e = m / 10.0, e + 1
    return f"{de(m, digits)}·10{str(e).translate(_SUPER)}"


def val(x: float) -> str:
    """Ein PEP-Wert: fünf Nachkommastellen, bei kleinen Werten in Zehnerpotenz-Schreibweise."""
    return de(x, 5) if abs(x) >= 1e-3 else sci(x, 2)


def h_text(h: float) -> str:
    return de(h, 2).rstrip("0").rstrip(",")


def step_text(h: float) -> str:
    return "1/L" if h == 1.0 else f"{h_text(h)}/L"


def method_label(method: str, h: float) -> str:
    return f"{PEP.METHOD_NAMES[method]}, Schritt {step_text(h)}" if method == "gd" else PEP.METHOD_NAMES[method]


def slope(xs, ys) -> float | None:
    """Steigung der Ausgleichsgeraden in der log-log-Darstellung (None bei weniger als zwei Punkten)."""
    if len(xs) < 2:
        return None
    return float(np.polyfit(np.log(np.asarray(xs, float)), np.log(np.asarray(ys, float)), 1)[0])


# ----------------------------------------------------------------------------- Ergebnis und Raten

def result_row(res: PEP.PepResult, ref) -> dict:
    return {"N": res.pep.N, "value": res.value, "dual": res.dual_value, "gap": res.gap, "reference": ref, "deviation": None if ref is None else res.value - ref, "status": res.status,
            "iterations": res.iterations, "seconds": res.seconds, "rank": res.rank, "relative_gap": abs(res.gap) / max(res.value, 1e-300), "gram": res.pep.k, "rows": res.pep.n_rows, "variables": len(res.pep.problem.c), "bound": res.bound}


def rate_rows(method: str, h: float, mu: float, nmax: int = C.N_MAX) -> list:
    """Der PEP-Wert für N = 1 … nmax (bei μ = 0,5 höchstens bis MU_NMAX), mit dem Vergleichswert der Literatur, wo es einen gibt."""
    top = min(nmax, C.MU_NMAX.get(mu, nmax))
    rows = []
    with threadpool_limits(1):
        for N in range(1, top + 1):
            rows.append(result_row(PEP.worst_case(method, N, h, mu, C.TOL), PEP.reference(method, N, h, mu)))
    return rows


def main_result(s: Settings):
    with threadpool_limits(1):
        return PEP.worst_case(s.method, s.n, s.h, s.mu, C.TOL)


def step_scan(N: int, mu: float, grid=C.H_SCAN) -> list:
    """PEP-Wert des Gradientenverfahrens nach N Schritten für Schrittlängen h aus `grid`, mit der Formel `(1/2) max{…}` (nur μ = 0) und dem Status „bewiesen“ (h ≤ 1) oder „Vermutung“."""
    rows = []
    with threadpool_limits(1):
        for h in grid:
            res = PEP.worst_case("gd", N, h, mu, C.TOL)
            ref = PEP.gd_formula(N, h) if mu == 0.0 else None
            rows.append({"h": h, "value": res.value, "reference": ref, "deviation": None if ref is None else res.value - ref, "proven": PEP.gd_proven(h), "status": res.status})
    return rows


def best_step(rows: list) -> dict:
    return min(rows, key=lambda r: r["value"])


def methods_rows(mu: float, nmax: int = C.N_MAX) -> dict:
    """GD (Schritt 1/L), FGM und OGM: PEP-Werte für N = 1 … nmax; dazu die Steigungen ab N = SLOPE_FROM und das Verhältnis OGM / FGM."""
    out = {m: rate_rows(m, 1.0, mu, nmax) for m in PEP.METHODS}
    sl = {}
    for m, rows in out.items():
        sel = [r for r in rows if r["N"] >= C.SLOPE_FROM]
        sl[m] = slope([r["N"] for r in sel], [r["value"] for r in sel])
    return {"rows": out, "slopes": sl}


def mu_table(h: float = 1.0) -> list:
    """GD mit Schritt h für jedes μ und N aus MU_TABLE_N (nur N ≤ MU_NMAX[μ])."""
    rows = []
    with threadpool_limits(1):
        for mu in C.MU_CHOICES:
            vals = {}
            for N in C.MU_TABLE_N:
                if N <= C.MU_NMAX[mu]:
                    res = PEP.worst_case("gd", N, h, mu, C.TOL)
                    vals[N] = {"value": res.value, "status": res.status}
            rows.append({"mu": mu, "values": vals})
    return rows


# ----------------------------------------------------------------------------- Experiment 1: das Handbeispiel

def hand_view() -> dict:
    """Das PEP für N = 1, h = 1 (L = R = 1): die sechs Interpolationszeilen an den Daten der eindimensionalen Funktion f₁ und am Optimum des SDP, die Gram-Matrizen und der Rang."""
    pep = PEP.build("gd", 1, 1.0)
    with threadpool_limits(1):
        res = PEP.solve(pep, C.TOL)
    f, df, xs, value = PEP.hand_function(1, 1.0)
    v = np.array([1.0, 1.0 / 3.0, 1.0 / 3.0])
    G_hand = np.outer(v, v)
    z_hand = PEP.variables(pep, [f(xs[0]), f(xs[1])], G_hand)
    s_hand = PEP.slacks(pep, z_hand)[:pep.n_rows]
    z_opt = PEP.variables(pep, res.f, res.G)
    s_opt = PEP.slacks(pep, z_opt)[:pep.n_rows]
    grid = np.linspace(-0.15, 1.2, 200)
    curve = (grid.tolist(), [float(f(x)) for x in grid])                  # die Funktion selbst bleibt draußen (lokale Funktion, nicht zwischenspeicherbar)
    names = {-1: "*", 0: "x₀", 1: "x₁"}
    rows = [{"i": names[i], "j": names[j], "hand": float(a), "optimum": float(b)} for (i, j), a, b in zip(pep.pairs, s_hand, s_opt)]
    return {"pep": pep, "result": res, "rows": rows, "G_hand": G_hand, "f_hand": [float(f(0.0)), float(f(xs[0])), float(f(xs[1]))], "x_hand": [0.0, xs[0], xs[1]], "g_hand": [0.0, float(df(xs[0])), float(df(xs[1]))],
            "value_hand": value, "min_slack_hand": float(s_hand.min()), "curve": curve}


# ----------------------------------------------------------------------------- Experiment 5: der Beweis

def proof_view(res: PEP.PepResult) -> dict:
    pep = res.pep
    names = ["*"] + [str(i) for i in range(pep.m)]
    M = np.zeros((pep.m + 1, pep.m + 1))
    for (i, j), lam in zip(pep.pairs, res.lam):
        M[i + 1, j + 1] = lam
    return {"gap": res.gap, "identity": PEP.identity_deviation(res), "positive": int(np.sum(res.lam > 1e-6)), "rows": pep.n_rows, "bound": res.bound, "value": res.value, "rank": res.rank,
            "S_rank": int(np.sum(np.linalg.eigvalsh(res.S) > 1e-6)), "S_size": pep.k, "lam_matrix": M, "names": names, "lam_min": float(res.lam.min()), "status": res.status}


# ----------------------------------------------------------------------------- Verdikte (reine Funktionen: (Art, Text))

def verdict_result(s: Settings, row: dict) -> tuple:
    head = f"{method_label(s.method, s.h)}, N = {s.n}, μ = {h_text(s.mu) if s.mu else '0'}: der scharfe Worst-Case-Wert von f(x_N) − f* ist **{val(row['value'])}** (in Einheiten von L R²)"
    if row["reference"] is not None:
        name = "(1/2) max{1/(2Nh + 1), (1 − h)^{2N}}" if s.method == "gd" else "1/(2θ_N²)"
        head += f"; die Formel der Literatur {name} gibt {val(row['reference'])} (Abweichung {sci(abs(row['deviation']))})"
    text = head + f". Das SDP hat eine {row['gram']}×{row['gram']}-Matrix, {row['rows']} Interpolationszeilen und {row['iterations']} Iterationen; Dualwert und Primalwert unterscheiden sich um {sci(abs(row['gap']))}, die Gram-Matrix hat Rang {row['rank']}."
    if row["status"] != "optimal":
        if row["relative_gap"] < 1e-4:
            return "success", text + f" Der Löser meldet den Status „{row['status']}“ (er stoppt, bevor die Toleranz erreicht ist); Primal- und Dualwert stimmen dennoch auf {sci(row['relative_gap'])} (relativ) überein."
        return "warning", text + f" **Status des Lösers: {row['status']}**, relative Lücke {sci(row['relative_gap'])}: der Wert liegt unter der Toleranz des Lösers, die letzten Stellen sind nicht verlässlich."
    return "success", text


def verdict_hand(hand: dict) -> tuple:
    res = hand["result"]
    return "success", (f"Bei N = 1 und Schritt 1/L hat das SDP eine 3×3-Matrix und sechs Interpolationszeilen; sein Optimum ist **{val(res.value)}** (1/6 = {val(1 / 6)}), die Gram-Matrix hat Rang {res.rank}. "
                       f"Die eindimensionale Funktion f₁(x) = |x|/3 − 1/18 (für |x| ≥ 1/3) liefert die Punkte x₀ = 1, x₁ = 2/3 mit f(x₁) = {val(hand['value_hand'])}; ihre Daten sind zulässig (kleinste Zeile {sci(hand['min_slack_hand'])}) "
                       f"und treffen den Optimalwert: die Schranke 1/6 ist scharf. Die Zeilen mit dem Wert 0 sind die, die am Optimum gelten müssen.")


def verdict_rate(rows: list, h: float, mu: float) -> tuple:
    first, last = rows[0], rows[-1]
    ref = [r for r in rows if r["reference"] is not None]
    text = f"Schritt {step_text(h)}, μ = {h_text(mu) if mu else '0'}: der Worst-Case-Wert fällt von {val(first['value'])} (N = {first['N']}) auf {val(last['value'])} (N = {last['N']})"
    sl = slope([r["N"] for r in rows if r["N"] >= C.SLOPE_FROM], [r["value"] for r in rows if r["N"] >= C.SLOPE_FROM])
    if sl is not None:
        text += f", die Steigung von log(Wert) gegen log(N) ab N = {C.SLOPE_FROM} ist {de(sl, 2)}"
    if ref and mu == 0.0 and h == 1.0:
        text += f". Für h = 1/L ist die Formel L R²/(4N + 2): größte Abweichung {sci(max(abs(r['deviation']) for r in ref))} über N = 1 … {last['N']}"
    elif ref:
        text += f". Die Formel (1/2) max{{1/(2Nh + 1), (1 − h)^{{2N}}}}: größte Abweichung {sci(max(abs(r['deviation']) for r in ref))}"
    return "info", text + f". Zeit des Lösers {de(1e3 * min(r['seconds'] for r in rows), 0)} bis {de(1e3 * max(r['seconds'] for r in rows), 0)} ms, {min(r['iterations'] for r in rows)} bis {max(r['iterations'] for r in rows)} Iterationen."


def verdict_step(rows: list, N: int, mu: float) -> tuple:
    best = best_step(rows)
    over = [r for r in rows if not r["proven"]]
    text = f"Bei N = {N} Schritten und μ = {h_text(mu) if mu else '0'} ist der Worst-Case-Wert am kleinsten bei h = {h_text(best['h'])}/L ({val(best['value'])}); bei h = 1/L sind es {val(next(r['value'] for r in rows if r['h'] == 1.0))}"
    if mu == 0.0:
        text += f". Die beste Schrittlänge der Formel ist h = {de(PEP.gd_optimal_step(N), 2)}/L: größere Schritte als 1/L sind besser. Die Formel ist für h ≤ 1 bewiesen (Drori, Teboulle) und für 1 < h < 2 eine Vermutung; das SDP bestätigt sie hier mit größter Abweichung {sci(max(abs(r['deviation']) for r in rows))} (davon {len(over)} der {len(rows)} Schrittlängen über 1/L)"
    else:
        text += ". Für μ > 0 gibt es hier keine Vergleichsformel: die Werte sind PEP-Werte"
    return "info", text + "."


def verdict_methods(m: dict, mu: float) -> tuple:
    rows, sl = m["rows"], m["slopes"]
    n = rows["gd"][-1]["N"]
    g, f, o = (rows[x][-1]["value"] for x in ("gd", "fgm", "ogm"))
    text = (f"Nach N = {n} Schritten: Gradientenverfahren {val(g)}, FGM {val(f)}, OGM {val(o)} (μ = {h_text(mu) if mu else '0'}). OGM ist das {de(o / f, 2)}fache von FGM; die Steigungen von log(Wert) gegen log(N) ab N = {C.SLOPE_FROM} sind "
            f"{de(sl['gd'], 2)} (Gradientenverfahren), {de(sl['fgm'], 2)} (FGM) und {de(sl['ogm'], 2)} (OGM): 1/N gegen 1/N² (die Grenzwerte −1 und −2 werden bei N ≤ {n} nicht erreicht)")
    if mu == 0.0:
        dev = max(abs(r["deviation"]) for r in rows["ogm"])
        text += f". OGM stimmt auf {sci(dev)} mit 1/(2θ_N²) überein; FGM liegt unter der Schranke 2/(N + 1)² ({val(PEP.fgm_bound(n))} bei N = {n})"
    return "info", text + "."


def verdict_proof(p: dict) -> tuple:
    return "success", (f"Das Dualprogramm liefert die Schranke {val(p['bound'])} und seinen Wert {val(p['value'] + p['gap'])}: Primal- und Dualwert unterscheiden sich um {sci(abs(p['gap']))}. Der Beweis ist die Identität "
                       f"f_N = Schranke − Σ λ_r s_r − ⟨S, G⟩ (die Summe über alle Zeilen, auch die Normierung 1 − G₀₀) mit nichtnegativen λ_r (kleinster Wert {sci(p['lam_min'])}) und einer PSD-Matrix S (Rang {p['S_rank']} von {p['S_size']}); an zufälligen (f, G) gilt sie auf {sci(p['identity'])}. "
                       f"{p['positive']} der {p['rows']} Multiplikatoren sind positiv: der Löser liefert den Mittelpunkt der optimalen Duallösungen, nicht eine dünne Beweisskizze.")


def verdict_mu(rows: list) -> tuple:
    parts = []
    for r in rows:
        parts.append(f"μ = {h_text(r['mu']) if r['mu'] else '0'}: " + ", ".join(f"N = {n}: {val(v['value'])}" for n, v in r["values"].items()))
    return "info", ("Gradientenverfahren mit Schritt 1/L, PEP-Werte: " + "; ".join(parts) + ". Je größer μ, desto kleiner der Wert: das SDP nutzt die zusätzliche Interpolationsbedingung der Stark-Konvexität. "
                    "Bei μ = 0,5 und N ≥ 6 liegt der Wert unter der Toleranz des Lösers (Status „stalled“); die Tabelle endet dort.")
