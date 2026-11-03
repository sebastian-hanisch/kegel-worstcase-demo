"""PDF-Export des aktuellen Worst-Case-Wertes mit der Raten-Tabelle (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Sonderzeichen werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

from kwc_evaluation import Settings, h_text, method_label, sci, val

_REPLACE = {"→": "->", "–": "-", "—": "-", "−": "-", "≤": "<=", "≥": ">=", "·": "*", "×": "x", "ν": "nu", "λ": "lambda", "β": "beta", "α": "alpha", "ρ": "rho", "μ": "mu", "τ": "tau", "θ": "theta", "κ": "kappa",
            "€": "EUR", "⁻": "-", "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9", "„": '"', "“": '"', "ᵀ": "^T", "ℓ": "l", "∞": "inf", "√": "sqrt", "‖": "|",
            "°": "o", "⟨": "<", "⟩": ">", "₀": "0", "₁": "1", "∗": "*", "⪰": ">="}


def _s(text) -> str:
    out = str(text).replace("**", "").replace("`", "")
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(s: Settings, row: dict, rows: list) -> bytes:
    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Worst-Case-Wert als SDP (Performance Estimation)"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _s(f"{method_label(s.method, s.h)}, N = {s.n}, mu = {h_text(s.mu) if s.mu else '0'}; L-glatt{', mu-stark' if s.mu else ''} konvex, ||x0 - x*|| <= 1. Alles simuliert (SDP, Werte in Einheiten von L R^2)."),
                   new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    head = ("N", "Worst-Case-Wert", "Formel der Literatur", "Abweichung", "Dual - Primal", "Rang von G", "Iterationen")
    widths = (14, 42, 48, 36, 36, 28, 28)
    pdf.set_font("Helvetica", "B", 9)
    for h, w in zip(head, widths):
        pdf.cell(w, 6, _s(h), border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    for r in rows:
        cells = (r["N"], val(r["value"]), "-" if r["reference"] is None else val(r["reference"]), "-" if r["deviation"] is None else sci(abs(r["deviation"])), sci(abs(r["gap"])), r["rank"], r["iterations"])
        for c, w in zip(cells, widths):
            pdf.cell(w, 5.5, _s(c), border=1)
        pdf.ln()
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Quelle: kegel-worstcase-demo, Sebastian Hanisch."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
