"""Performance-Estimation-Problem (PEP) als Kegelprogramm: der Worst-Case eines Verfahrens erster Ordnung als semidefinites Programm.

Worst-Case von  f(x_N) − f*  über alle L-glatten (μ-stark) konvexen Funktionen f und alle Startpunkte mit ‖x_0 − x*‖ ≤ 1 (L = 1) für ein Verfahren, dessen Iterierte
linear von den bisherigen Gradienten abhängen. Basis der Gram-Matrix G: e_0 = x_0 − x*, e_1 … e_m = die Gradienten g_0 … g_{m−1} an den m Abfragepunkten (x* = 0, g* = 0, f* = 0).
Variablen z = (f_0 … f_{m−1}, obere Dreieckseinträge von G). Nebenbedingungen (alle linear in z):
  G ⪰ 0                                                                  PSD-Kegel (svec mit √2 wie im Kern)
  Interpolation für jedes Paar (i, j) der Punkte {*, 0 … m−1}  (Taylor, Hendrickx, Glineur 2017, Satz 4):
    f_i − f_j − ⟨g_j, x_i − x_j⟩ − 1/(2(1 − μ/L)) (‖g_i − g_j‖²/L + μ ‖x_i − x_j‖² − (2μ/L) ⟨g_j − g_i, x_j − x_i⟩) ≥ 0      (μ = 0: f_i − f_j − ⟨g_j, x_i − x_j⟩ − ‖g_i − g_j‖²/(2L) ≥ 0)
  G_00 ≤ 1                                                               (‖x_0 − x*‖² ≤ 1)
Ziel: max f_{m−1} (der letzte Abfragepunkt ist der Ausgang x_N des Verfahrens). Das Programm im Format des Kerns: min cᵀz mit s = b − A z ∈ K, K = Orthant × PSD.
Dual: z_dual = (λ (Interpolationszeilen), Multiplikator der Normierung, svec(S)); es gilt für jedes (f, G):  f_N = (Multiplikator der Normierung) − Σ λ_r s_r − ⟨S, G⟩, die Summe über alle Zeilen (Interpolation und Normierung 1 − G_00) mit s_r ≥ 0 für zulässige Punkte:
das ist der Beweis der Schranke (alle Terme nichtnegativ)."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

import kegel_kern as K
import kwc_solver as PD

METHODS = ("gd", "fgm", "ogm")
METHOD_NAMES = {"gd": "Gradientenverfahren", "fgm": "Nesterov (FGM)", "ogm": "Optimiertes Gradientenverfahren (OGM)"}
L = 1.0


# ----------------------------------------------------------------------------- Verfahren: Abfragepunkte als Koeffizienten über der Basis

def _unit(k: int, i: int) -> np.ndarray:
    v = np.zeros(k)
    v[i] = 1.0
    return v


def gd_points(N: int, h: float):
    """Gradientenverfahren x_{i+1} = x_i − (h/L) g_i, Abfragepunkte x_0 … x_N; Basis (x_0 − x*, g_0 … g_N), Größe N + 2."""
    k = N + 2
    pts = []
    for i in range(N + 1):
        x = _unit(k, 0)
        for j in range(i):
            x = x - h * _unit(k, 1 + j)
        pts.append((x, _unit(k, 1 + i)))
    return pts, k


def fgm_points(N: int):
    """Nesterovs schnelles Gradientenverfahren: y_0 = x_0, x_{i+1} = y_i − g(y_i)/L, t_{i+1} = (1 + √(1 + 4 t_i²))/2, y_{i+1} = x_{i+1} + ((t_i − 1)/t_{i+1})(x_{i+1} − x_i).
    Abfragepunkte y_0 … y_{N−1}, dann das Ergebnis x_N (die „primäre Folge“ bei Taylor, Hendrickx, Glineur)."""
    k = N + 2
    t = 1.0
    y, x_prev = _unit(k, 0), _unit(k, 0)
    pts = []
    for i in range(N):
        g = _unit(k, 1 + i)
        pts.append((y, g))
        x_new = y - g
        t_new = (1.0 + math.sqrt(1.0 + 4.0 * t * t)) / 2.0
        y = x_new + ((t - 1.0) / t_new) * (x_new - x_prev)
        x_prev, t = x_new, t_new
    pts.append((x_prev, _unit(k, 1 + N)))
    return pts, k


def ogm_theta(N: int) -> float:
    """θ_N des optimierten Gradientenverfahrens: θ_0 = 1, θ_{i+1} = (1 + √(4 θ_i² + 1))/2 für i ≤ N − 2, im letzten Schritt (1 + √(8 θ_i² + 1))/2."""
    theta = 1.0
    for i in range(N):
        theta = (1.0 + math.sqrt((4.0 if i <= N - 2 else 8.0) * theta * theta + 1.0)) / 2.0
    return theta


def ogm_points(N: int):
    """Optimiertes Gradientenverfahren (Kim und Fessler, in der Schreibweise von Taylor, Hendrickx, Glineur Abschnitt 4.2): Abfragepunkte x_0 … x_{N−1}, dann das Ergebnis x_N (sekundäre Folge)."""
    k = N + 2
    theta = 1.0
    x, y = _unit(k, 0), _unit(k, 0)
    pts = []
    for i in range(N):
        g = _unit(k, 1 + i)
        pts.append((x, g))
        y_new = x - g
        theta_new = (1.0 + math.sqrt((4.0 if i <= N - 2 else 8.0) * theta * theta + 1.0)) / 2.0
        x = y_new + ((theta - 1.0) / theta_new) * (y_new - y) + (theta / theta_new) * (y_new - x)
        y, theta = y_new, theta_new
    pts.append((x, _unit(k, 1 + N)))
    return pts, k


def method_points(method: str, N: int, h: float = 1.0):
    if N < 1:
        raise ValueError("mindestens ein Schritt")
    if method == "gd":
        return gd_points(N, h)
    if method == "fgm":
        return fgm_points(N)
    if method == "ogm":
        return ogm_points(N)
    raise ValueError(f"unbekanntes Verfahren {method!r}")


def simulate(method: str, N: int, grad, x0: np.ndarray, h: float = 1.0) -> list:
    """Dasselbe Verfahren numerisch auf einer Funktion mit Gradient `grad` (L = 1): die Abfragepunkte (wie in method_points), der letzte ist das Ergebnis x_N."""
    x0 = np.asarray(x0, float)
    pts = []
    if method == "gd":
        x = x0
        for _ in range(N):
            pts.append(x)
            x = x - h * grad(x)
        pts.append(x)
    elif method == "fgm":
        t, y, x_prev = 1.0, x0, x0
        for _ in range(N):
            pts.append(y)
            x_new = y - grad(y)
            t_new = (1.0 + math.sqrt(1.0 + 4.0 * t * t)) / 2.0
            y = x_new + ((t - 1.0) / t_new) * (x_new - x_prev)
            x_prev, t = x_new, t_new
        pts.append(x_prev)
    elif method == "ogm":
        theta, x, y = 1.0, x0, x0
        for i in range(N):
            pts.append(x)
            y_new = x - grad(x)
            theta_new = (1.0 + math.sqrt((4.0 if i <= N - 2 else 8.0) * theta * theta + 1.0)) / 2.0
            x = y_new + ((theta - 1.0) / theta_new) * (y_new - y) + (theta / theta_new) * (y_new - x)
            y, theta = y_new, theta_new
        pts.append(x)
    else:
        raise ValueError(f"unbekanntes Verfahren {method!r}")
    return pts


# ----------------------------------------------------------------------------- Formeln aus der Literatur (Vergleichswerte, keine Eingabe des Programms)

def gd_formula(N: int, h: float) -> float:
    """(L R²/2) max{1/(2 N h + 1), (1 − h)^{2N}} mit L = R = 1: für 0 ≤ h ≤ 1 bewiesen (Drori, Teboulle), für 1 < h < 2 Vermutung (Taylor, Hendrickx, Glineur; ein Preprint von 2025 behauptet den Beweis, hier nicht nachgeprüft), hier numerisch bestätigt."""
    return 0.5 * max(1.0 / (2.0 * N * h + 1.0), (1.0 - h) ** (2 * N))


def gd_proven(h: float) -> bool:
    return h <= 1.0


def gd_optimal_step(N: int) -> float:
    """Die Schrittlänge h > 1, die die Formel minimiert: 1/(2 N h + 1) = (1 − h)^{2N} (der erste Term fällt, der zweite steigt für h > 1); Bisektion."""
    lo, hi = 1.0, 2.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if 1.0 / (2.0 * N * mid + 1.0) > (1.0 - mid) ** (2 * N):
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def ogm_formula(N: int) -> float:
    return 1.0 / (2.0 * ogm_theta(N) ** 2)


def fgm_bound(N: int) -> float:
    """2 L R²/(N + 1)²: die bekannte Schranke für die primäre Folge (Beck, Teboulle; hier über Taylor, Hendrickx, Glineur belegt), nicht scharf."""
    return 2.0 / (N + 1) ** 2


def reference(method: str, N: int, h: float, mu: float):
    """Der Vergleichswert der Literatur für (Verfahren, N, h, μ), oder None, wo keiner bekannt ist (μ > 0)."""
    if mu != 0.0:
        return None
    if method == "gd":
        return gd_formula(N, h)
    if method == "ogm":
        return ogm_formula(N)
    return None


def hand_function(N: int = 1, h: float = 1.0):
    """Die eindimensionale schlechteste Funktion f₁ (Taylor, Hendrickx, Glineur Abschnitt 4.1.1, L = R = 1): f₁(x) = |x|/(2Nh + 1) − 1/(2 (2Nh + 1)²) für |x| ≥ 1/(2Nh + 1), sonst x²/2.
    Liefert (f, f', Wert des Gradientenverfahrens nach N Schritten ab x₀ = 1): der Wert ist 1/(2 (2Nh + 1))."""
    a = 1.0 / (2.0 * N * h + 1.0)

    def f(x):
        x = abs(float(x))
        return x * a - a * a / 2.0 if x >= a else x * x / 2.0

    def df(x):
        x = float(x)
        return math.copysign(a, x) if abs(x) >= a else x

    x = 1.0
    xs = [x]
    for _ in range(N):
        x = x - h * df(x)
        xs.append(x)
    return f, df, xs, f(xs[-1])


# ----------------------------------------------------------------------------- Aufbau

@dataclass(frozen=True, eq=False)
class Pep:
    problem: K.ConeProblem
    method: str
    N: int
    h: float
    mu: float
    m: int                      # Abfragepunkte (f-Variablen)
    k: int                      # Größe der Gram-Matrix
    pairs: list                 # (i, j) je Interpolationszeile; −1 steht für den Optimalpunkt *
    n_rows: int
    points: list                # (x-Koeffizienten, g-Koeffizienten) je Abfragepunkt


def _triu(k: int):
    iu0, iu1 = np.triu_indices(k)
    return iu0, iu1, (iu0 != iu1).astype(float)


def build_from_points(points, k: int, method: str = "custom", N: int | None = None, h: float = 1.0, mu: float = 0.0) -> Pep:
    if not 0.0 <= mu < L:
        raise ValueError("μ muss in [0, L) liegen")
    m = len(points)
    iu0, iu1, off = _triu(k)
    npair = len(iu0)
    nv = m + npair
    index = {(int(a), int(b)): m + n for n, (a, b) in enumerate(zip(iu0, iu1))}

    def ip(a, b):                                                       # ⟨a, G b⟩ als Koeffizientenvektor über den Variablen (nur G-Teil)
        v = np.zeros(nv)
        v[m:] = a[iu0] * b[iu1] + off * a[iu1] * b[iu0]
        return v

    allp = [(np.zeros(k), np.zeros(k), None)] + [(x, g, i) for i, (x, g) in enumerate(points)]            # * hat f = 0
    rows, pairs = [], []
    for pi, (xi, gi, fi) in enumerate(allp):
        for pj, (xj, gj, fj) in enumerate(allp):
            if pi == pj:
                continue
            r = np.zeros(nv)
            if fi is not None:
                r[fi] += 1.0
            if fj is not None:
                r[fj] -= 1.0
            r -= ip(gj, xi - xj)
            d_g, d_x = gi - gj, xi - xj
            if mu == 0.0:
                r -= ip(d_g, d_g) / (2.0 * L)
            else:
                c = 1.0 / (2.0 * (1.0 - mu / L))
                r -= c * (ip(d_g, d_g) / L + mu * ip(d_x, d_x) - (2.0 * mu / L) * ip(gj - gi, xj - xi))
            rows.append(r)
            pairs.append((pi - 1, pj - 1))
    R = np.array(rows)
    norm_row = np.zeros(nv)
    norm_row[index[(0, 0)]] = -1.0                                       # 1 − G_00 ≥ 0
    psd = np.zeros((npair, nv))
    for n, (a, b) in enumerate(zip(iu0, iu1)):
        psd[n, index[(int(a), int(b))]] = 1.0 if a == b else K.SQ2
    A = np.vstack([-R, -norm_row, -psd])
    b = np.concatenate([np.zeros(len(R)), [1.0], np.zeros(npair)])
    c = np.zeros(nv)
    c[m - 1] = -1.0                                                      # max f_N = min −f_N
    cones = (("l", len(R) + 1), ("s", k))
    return Pep(K.ConeProblem(c, A, b, cones, np.zeros(nv)), method, N if N is not None else m - 1, h, mu, m, k, pairs, len(R), points)


def build(method: str, N: int, h: float = 1.0, mu: float = 0.0) -> Pep:
    pts, k = method_points(method, N, h)
    return build_from_points(pts, k, method, N, h, mu)


def variables(pep: Pep, f: np.ndarray, G: np.ndarray) -> np.ndarray:
    """Der Variablenvektor z = (f, obere Dreieckseinträge von G) zu Funktionswerten und einer Gram-Matrix (zum Prüfen von Daten, die nicht aus dem Löser kommen)."""
    iu = np.triu_indices(pep.k)
    return np.concatenate([np.asarray(f, float), np.asarray(G, float)[iu]])


def slacks(pep: Pep, z: np.ndarray) -> np.ndarray:
    """Die Zeilen des Programms zu z: Interpolationszeilen, dann 1 − G_00, dann svec(G)."""
    return pep.problem.b - pep.problem.A @ z


# ----------------------------------------------------------------------------- Lösen

@dataclass(frozen=True, eq=False)
class PepResult:
    pep: Pep
    value: float                # max f_N − f*
    dual_value: float           # Wert des Dualprogramms (Schranke aus dem Beweis)
    status: str
    iterations: int
    seconds: float
    f: np.ndarray
    G: np.ndarray
    lam: np.ndarray             # Multiplikatoren der Interpolationszeilen
    bound: float                # Multiplikator der Normierung: die bewiesene Schranke
    S: np.ndarray               # PSD-Multiplikator
    eig: np.ndarray             # Eigenwerte von G, absteigend
    z_dual: np.ndarray

    @property
    def gap(self) -> float:
        return self.dual_value - self.value

    @property
    def rank(self) -> int:
        return int(np.sum(self.eig > 1e-6 * max(self.eig[0], 1e-300)))


def solve(pep: Pep, tol: float = 1e-9, max_iter: int = 100) -> PepResult:
    t0 = time.perf_counter()
    res = PD.solve_pd(pep.problem, tol=tol, max_iter=max_iter)
    sec = time.perf_counter() - t0
    m, k, nr = pep.m, pep.k, pep.n_rows
    iu = np.triu_indices(k)
    G = np.zeros((k, k))
    G[iu] = res.x[m:]
    G = G + G.T - np.diag(np.diag(G))
    zd = res.z
    return PepResult(pep, -res.value, -res.dual_value, res.status, res.iterations, sec, res.x[:m].copy(), G, zd[:nr].copy(), float(zd[nr]), K.smat(zd[nr + 1:], k), np.sort(np.linalg.eigvalsh(G))[::-1], zd.copy())


def worst_case(method: str, N: int, h: float = 1.0, mu: float = 0.0, tol: float = 1e-9) -> PepResult:
    return solve(build(method, N, h, mu), tol)


def identity_deviation(res: PepResult, samples: int = 5, seed: int = 0) -> float:
    """Der Beweis als Identität: für jedes z gilt f_N = bound − Σ λ_r s_r − ⟨S, G⟩ (mit s = b − A z); die größte Abweichung an zufälligen z."""
    pep = res.pep
    prob = pep.problem
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(samples):
        z = rng.normal(size=len(prob.c))
        s = prob.b - prob.A @ z
        worst = max(worst, abs(z[pep.m - 1] - (prob.b @ res.z_dual - res.z_dual @ s)))
    return worst
