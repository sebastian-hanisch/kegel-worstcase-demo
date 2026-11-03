# Worst-Case-Raten als SDP – Performance Estimation mit dem PSD-Löser der Reihe – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-kegel-worstcase-demo.streamlit.app/)** -->

Stück 11 von 16 der **Konzepte-Reihe „Konvexe und konische Optimierung“** für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Wie jede Konzepte-Demo dreht sich alles um **ein** Verfahren mit wachsendem Beispiel: ein Löser für Kegelprogramme, dessen Kegel von Stück zu Stück wächst. Dieses Stück schließt den Strang **Erste Ordnung**: in Stück 8 bis 10 waren Schrittzahlen **gemessen**, hier werden Raten **hergeleitet**, und der Beweis ist selbst ein SDP, gelöst mit dem PSD-Kegel und dem primal-dualen Löser der Reihe.

## Kernfrage

**Wie schnell ist ein Verfahren erster Ordnung im schlechtesten Fall, und lässt sich das berechnen statt beweisen?** „Für alle L-glatten konvexen Funktionen und alle Startpunkte“ klingt nach unendlich vielen Fällen. Die **Interpolationsbedingungen** (Taylor, Hendrickx, Glineur 2017) machen daraus ein endliches Programm: Punkte `(xᵢ, gᵢ, fᵢ)` lassen sich genau dann durch eine L-glatte konvexe Funktion verbinden, wenn für jedes Paar `fᵢ − fⱼ − ⟨gⱼ, xᵢ − xⱼ⟩ − ‖gᵢ − gⱼ‖²/(2L) ≥ 0` gilt. Mit der Gram-Matrix `G` von `(x₀ − x*, g₀, …, g_N)` als PSD-Variable ist alles **linear in `(f, G)`**, und „maximiere `f(x_N) − f*` unter `‖x₀ − x*‖ ≤ 1`“ ist ein SDP der Größe `(N + 2) × (N + 2)`. Der primal-duale Löser aus Stück 6 findet für das Gradientenverfahren mit Schritt 1/L **`L R²/(4N + 2)` auf zehn Stellen**, für beliebige Schrittlängen die Formel `(L R²/2) max{1/(2Nh + 1), (1 − h)^{2N}}`, und für das **optimierte Gradientenverfahren (OGM)** genau `L R²/(2θ_N²)`, halb so groß wie bei Nesterovs FGM. Das **Dual ist der Beweis**: nichtnegative Multiplikatoren machen aus der Schranke eine Identität.

Das Beispiel ist hier kein Punktwolkenproblem mehr, sondern das **Programm selbst**: die Matrix `A` des SDP wird aus dem Verfahren erzeugt.

## Modell

- **Kern** (`kegel_kern.py`, wortgleich mit Stück 2 bis 10, unverändert): `ConeProblem` (Orthant × PSD), `svec`/`smat`.
- **Löser** (`kwc_solver.py`, `kwc_cones.py`, aus Stück 6 übernommen): primal-dualer innerer Punkt (Mehrotra, Nesterov–Todd-Skalierung) mit Dualvariablen; er braucht keinen strikt zulässigen Start.
- **PEP** (`kwc_pep.py`): die drei Verfahren als Abfragepunkte mit Koeffizienten über der Basis `(x₀ − x*, g₀, …, g_N)` (Gradientenverfahren mit Schritt `h`, Nesterovs FGM, OGM), der Aufbau der Interpolationszeilen für `μ ≥ 0`, das Kegelprogramm, das Lösen, die Gram-Matrix, die Dualvariablen, die Identität des Beweises; dazu `simulate` (dieselben Verfahren numerisch) und die Formeln der Literatur als Vergleichswerte.
- **Auswertung** (`kwc_evaluation.py`): Raten, Schrittlängen-Scan, Vergleich der Verfahren, μ-Tabelle, Handbeispiel, Beweis-Ansicht und die Verdikte.
- **Kein Eingriff in den Kern:** alles Neue steht in `kwc_*.py`; BLAS-Threads laufen mit einem Thread (`threadpoolctl`).

## Methodik

Fünf Experimente mit live berechnetem Verdikt: (1) das PEP für `N = 1` von Hand (drei Punkte, sechs Interpolationszeilen, die eindimensionale schlechteste Funktion `f₁`, die Zeilen an ihren Daten und am SDP-Optimum); (2) der Worst-Case-Wert des Gradientenverfahrens für `N = 1 … 12` gegen die Formel, mit Größe und Zeit des SDP; (3) die Schrittlänge `h ∈ (0, 2)`; (4) Gradientenverfahren, FGM und OGM im Vergleich (Steigungen, Verhältnis); (5) der Beweis (Heatmap der Multiplikatoren, Identität) und der μ-stark konvexe Fall. Regler: Verfahren, `N` (1 bis 12), Schrittlänge `h` (0,25 bis 1,9), `μ` (0 / 0,1 / 0,3 / 0,5); ein Schnellstart lädt sechs Szenarien, die Adresszeile spiegelt die Konfiguration. Alle Werte sind in Einheiten von `L R²` (`L = 1`, `‖x₀ − x*‖ ≤ 1`), die Toleranz des Lösers ist `10⁻⁹`.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py` (wo dort nur ein Band geprüft wird, wegen Rundung und Rechner, steht die gemessene Zahl trotzdem hier). Zeiten gelten für den Entwicklungsrechner und werden nur in weiten Bändern getestet.

| Frage | Befund |
|---|---|
| Gradientenverfahren, Schritt 1/L: PEP-Wert gegen `1/(4N+2)` | N = 1 / 2 / 3 / 4 / 5 / 6 / 7 / 8: **0,1666666662 / 0,0999999998 / 0,0714285714 / 0,0555555553 / 0,0454545448 / 0,0384615383 / 0,0333333333 / 0,0294117641**; Abweichung von `1/(4N+2)` höchstens 10⁻⁹ (bis `N = 12` im Test: 10⁻⁸). Status „optimal“, 13 bis 25 Iterationen, Gram-Matrix 3×3 bis 10×10 (6 bis 90 Interpolationszeilen), Primal- und Dualwert gleich auf 10⁻⁹ |
| Schrittlänge `h` (N = 4) | h = 0,25 / 0,5 / 0,75 / 1 / 1,25 / 1,5 / 1,75 / 1,9: **0,16666667 / 0,1 / 0,07142857 / 0,05555556 / 0,04545454 / 0,03846154 / 0,05005646 / 0,2152336** (Abweichung von `(1/2) max{1/(2Nh+1), (1−h)^{2N}}` höchstens 10⁻⁷; im Scan über 20 Schrittlängen 7,5·10⁻⁹). Bei `h = 1,75` und `1,9` bestimmt das Überschießen (`(1−h)^{2N}`) das Maximum. Das Minimum der 20 Schrittlängen liegt bei `h = 1,7` (0,03425) gegen 0,05556 bei `h = 1/L` |
| Beste Schrittlänge der Formel (`1/(2Nh+1) = (1−h)^{2N}`, Bisektion) | N = 1 / 2 / 3 / 4 / 6 / 8 / 12: **h_opt = 1,50 / 1,61 / 1,67 / 1,71 / 1,77 / 1,81 / 1,85**, Wert 0,125 / 0,06736 / 0,04536 / 0,03398 / 0,02246 / 0,01670 / 0,01100; das SDP trifft die Formel dort (Test für N = 1 und 4). Die Quelle nennt `3/2 ≤ h_opt < 2` und `h_opt → 2` |
| Beschleunigung (PEP-Werte, N = 4 / 8 / 12) | Gradientenverfahren **0,05556 / 0,02941 / 0,02000**, FGM **0,04683 / 0,01756 / 0,00915**, OGM **0,02558 / 0,00904 / 0,00463**. FGM gleich dem Gradientenverfahren für `N = 1` und `2`, danach besser, unter `2/(N+1)²` (N = 8: 0,0247). OGM stimmt auf **10⁻¹⁰** mit `1/(2θ_N²)` überein (`N = 1 … 12`), ist das **0,750fache** von FGM bei `N = 1` und das **0,507fache** bei `N = 12`; bei `N = 1` ist OGM gleich dem Gradientenverfahren mit `h = 1,5` (0,125) |
| Raten (Steigung von log(Wert) gegen log(N), N = 4 bis 12) | **−0,93 (Gradientenverfahren), −1,49 (FGM), −1,56 (OGM)**: die Grenzwerte −1, −2, −2 werden bei `N ≤ 12` nicht erreicht |
| μ-stark konvex (Gradientenverfahren, Schritt 1/L; N = 1 / 2 / 4) | μ = 0,1: 0,14944649 / 0,08010793 / 0,03513562; μ = 0,3: 0,11187215 / 0,0432909 / 0,00901082; μ = 0,5: 0,07142857 / 0,01612903 / 0,00097847. Je größer μ, desto kleiner der Wert; alle unter `1/(4N+2)` und unter `(1−μ)^{2N}/2`. Bei `μ = 0,5` und `N ≥ 6` liegt der Wert (6·10⁻⁵) unter der Toleranz des Lösers (Status „stalled“): die Rate endet dort bei `N = 5` |
| Das Handbeispiel (`N = 1`, `h = 1`) | Wert **1/6**; die Gram-Matrix des SDP ist `v vᵀ` mit `v = (1, 1/3, 1/3)` (Rang 1, auf 10⁻⁵ gleich der von Hand gebauten); die eindimensionale Funktion `f₁(x) = abs(x)/3 − 1/18` (für `abs(x) ≥ 1/3`) liefert `x₀ = 1`, `x₁ = 2/3`, `f(x₁) = 1/6`; die Daten sind zulässig (kleinste Zeile −3·10⁻¹⁷); von den sechs Interpolationszeilen sind vier aktiv (0), `(x₀, *)` hat den Wert 2/9 und `(x₁, *)` den Wert 1/9 |
| Rang der optimalen Gram-Matrix | **Rang 1** beim Gradientenverfahren (N = 3, 6; auch `h = 1,9` und `μ = 0,3`) und bei FGM (N = 4, 8): die schlechteste Funktion ist eindimensional. **Beim OGM nicht:** Rang **N + 1 von N + 2** für `N = 1 … 8`; die Menge der schlechtesten Funktionen ist groß, der Löser liefert den Mittelpunkt der optimalen Menge |
| Der Beweis (Gradientenverfahren, `h = 1`, `N = 4`) | Schranke (Multiplikator der Normierung) = Primalwert = 0,0555555556; die Identität `f_N = Schranke − Σ λ_r s_r − ⟨S, G⟩` gilt an zufälligen `(f, G)` auf **9·10⁻¹⁶** (Band 10⁻⁹); **25 von 30** Interpolationsmultiplikatoren sind positiv (Band 15 bis 30: der Löser liefert den **Mittelpunkt** der Menge optimaler Duallösungen, keine dünne Beweisskizze); `S` hat Rang 5 von 6 |
| Größe und Zeit des Programms (Gradientenverfahren, `h = 1`) | N = 2: Gram 4×4, 13 Variablen, 23 Zeilen, 15 Iterationen, rund 20 ms; N = 4: 6×6, 26, 52, 13, 20 ms; N = 12: 14×14, 118, 288, 22, rund 340 ms; N = 16 (nicht live): 18×18, 188, 478, 26, rund 1 s. Die Iterationszahl bleibt klein, die Zeit wächst mit dem dichten KKT-System |
| Gegen cvxpy | Ein **eigener** Aufbau des PEP in cvxpy (Variablen `G` mit `PSD=True` und `f`, eigene Zeilen aus den Abfragepunkten) gibt mit Clarabel und SCS dieselben Werte (10⁻⁶) für Gradientenverfahren (`h = 0,5; 1; 1,5`), FGM, OGM und `μ = 0,3` |

## Befunde und Korrekturen gegenüber dem Plan

- **OGM hat nicht Rang 1.** Der Plan nannte Rang 1 für alle Verfahren (Phase 0 hatte nur Gradientenverfahren und FGM gemessen). Beim OGM liefert der innere Punkt den **analytischen Mittelpunkt** der optimalen Menge: Rang `N + 1` von `N + 2` (das OGM hat viele schlechteste Funktionen); die Rang-1-Aussage steht jetzt nur bei Gradientenverfahren und FGM. Der Test prüft beides.
- **Für `h > 2` meldet der Löser keinen „unbeschränkten“ Status.** Der Plan sagte, das Programm sei dort unbeschränkt; tatsächlich ist es beschränkt, und der Wert wächst mit dem Überschießen (`h = 2,2`, `N = 3`: 1,49299 gleich `(1,2)⁶/2`; `h = 3`: 32). Der Status des Lösers ist dort „nan“, der Wert stimmt. Die App begrenzt den Regler trotzdem auf `h < 2` (die Formel der Quelle gilt für 0 ≤ h ≤ 2).
- **Der Status „nan“ oder „stalled“ heißt nicht „falsch“.** Bei `h = 0,25` (N = 4) und bei kleinen Werten stoppt der Löser, bevor die Toleranz `10⁻⁹` in allen Residuen erreicht ist, mit einem Wert, der auf 5·10⁻⁹ stimmt. Die App meldet den Status und beurteilt den Wert an der **relativen Lücke** zwischen Primal- und Dualwert (unter 10⁻⁴: Erfolg mit Hinweis, sonst Warnung).
- **Die Handzeilen des Plans waren unvollständig.** Der Plan nannte „die Paare mit `*` liefern 0 oder positive Werte“; gerechnet sind es `(x₀, *) = 2/9` und `(x₁, *) = 1/9` (positiv) und vier Nullzeilen `(*, x₀)`, `(*, x₁)`, `(x₀, x₁)`, `(x₁, x₀)`. Der Test hat die genauen Werte.
- **Die Schrittlängen-Beste der Formel liegt bei 1,61 für `N = 2`, nicht 1,60.** Phase 0 suchte auf einem Raster von 0,01 (Minimum bei 1,60); die Formel (Bisektion) gibt 1,6058. Das SDP-Minimum auf dem Raster (0,06757) liegt deshalb über dem Formelminimum (0,06736).
- **`N = 0` wird nicht gebaut** (der Plan ließ es offen): der Regler beginnt bei `N = 1`, das Programm lehnt `N < 1` ab.
- **Die Vergleichsformel für OGM ist aus der Quelle nicht nachgelesen.** Die SDP-Werte stimmen auf 10⁻¹⁰ mit `1/(2θ_N²)` (θ aus der Rekursion, im letzten Schritt mit 8) überein; Kim und Fessler selbst habe ich nicht gelesen. Die Formel steht als Vergleichswert, nicht als geprüfter Satz.
- **Quellen:** geprüft am 2026-11-03 per Suche und im **Volltext** (`pdftotext` der von WebFetch gespeicherten PDF): Taylor, Hendrickx, Glineur, *Smooth strongly convex interpolation and exact worst-case performance of first-order methods*, Mathematical Programming 161 (1–2), 2017, S. 307–345, DOI 10.1007/s10107-016-1009-3 (arXiv 1502.05666): **Satz 4** (Interpolation für `F_{μ,L}`), **Vermutung 1** (Gradientenverfahren, `0 ≤ h ≤ 2`; bewiesen für `h ≤ 1` bei Drori und Teboulle, „leaving 1 < h < 2 open“; getestet für `N ≤ 30`, `h = 0,05 … 1,95` mit relativem Fehler 10⁻⁷), die Aussage zu eindimensionalen schlechtesten Funktionen und `3/2 ≤ h_opt < 2`; Drori und Teboulle, Mathematical Programming 145 (1–2), 2014, S. 451–482, DOI 10.1007/s10107-013-0653-0 (Suche); Kim und Fessler, Mathematical Programming 159, 2016, DOI 10.1007/s10107-015-0949-3 (Suche: Schranke halb so groß wie bei FGM); die FGM-Schranke `2LR²/(N+1)²` steht dort als Verweis auf Beck und Teboulle (FISTA, 2009; nur über Taylor, Hendrickx, Glineur belegt). Der Abstract von Abbaszadehpeivasti, de Klerk, Zamani (arXiv 2104.05468) betrifft die Norm des Gradienten und wird nicht herangezogen. Beweise der Arbeiten nicht nachgelesen.
- **Port:** 9309 (`launch.json` und Netstat gelesen, frei).

## Ehrliche Grenzen

- **Die Formel für `1 < h < 2` ist eine Vermutung der Quelle.** Das SDP bestätigt sie hier numerisch (Scan über 20 Schrittlängen, Abweichung 10⁻⁸), ersetzt aber keinen Beweis; „bewiesen“ steht nur für `h ≤ 1`.
- **Nur Verfahren mit festen linearen Koeffizienten** (Gradientenverfahren, FGM, OGM); adaptive Schrittlängen, Liniensuche, proximale Schritte und Heavy-Ball brauchen ein anderes oder erweitertes Programm.
- **Nur das Ziel `f(x_N) − f*`** für L-glatte (μ-stark) konvexe Funktionen mit `L = 1`; kein Gradientennorm-Kriterium, kein ADMM (das PEP für ADMM ist eine eigene Aufgabe).
- **Der Beweis ist die Identität, nicht eine Skizze:** der Löser liefert den Mittelpunkt der optimalen Duallösungen (25 von 30 Multiplikatoren positiv); eine dünne Duallösung (Ecke), wie die Arbeiten sie zeigen, wird nicht gesucht.
- **Die Raten sind Messungen für `N ≤ 12`:** die Steigungen −0,93 / −1,49 / −1,56 sind nicht die asymptotischen; der Löser ist ein dichter innerer Punkt (rund 1 s bei `N = 16`).
- **Der Löser stockt bei kleinen Werten** (`μ = 0,5` und `N ≥ 6`, `h` nahe 0): Status „stalled“ oder „nan“, die letzten Stellen sind nicht verlässlich; die App begrenzt `N` bei `μ = 0,5` auf 5.
- **Zeiten hängen vom Rechner ab** und werden nur in weiten Bändern getestet.

## Verifikation

- **PEP** (`tests/test_pep.py`): das Handbeispiel von Hand (die Daten von `f₁` als zulässiger Punkt mit Zielwert 1/6, die sechs Zeilenwerte 2/9, 1/9 und vier Nullen, Zeile für Zeile, ohne den Löser), bekannte Werte (`N = 1` mit `h = 1` und `1,5`, `N = 2`, `N = 3`), **die Abfragepunkte stimmen mit den numerisch gerechneten Iterierten überein** (für alle drei Verfahren), der Wert ist eine **obere Schranke für Zufallsquadriken** (60 je Verfahren) und wird von `f₁` und `f₂` **von unten erreicht** (zwei Richtungen), die Dual-Identität, Monotonie in `N` und `μ`, `OGM ≤ FGM ≤ GD`, die Formeln und die beste Schrittlänge, abgelehnte Eingaben (`N = 0`, `μ ≥ 1`, unbekanntes Verfahren).
- **Löser** (`tests/test_solver.py`): die Hand-Programme des Kerns (Orthant, Lorentz, PSD) mit Primal- und Dualwert, der Dualpunkt liegt im Kegel.
- **Orakel** (`tests/test_oracle_kegel_worstcase.py`): ein **eigener** Aufbau des PEP in cvxpy gegen unseren Aufbau mit Clarabel und SCS; die Formeln der Literatur für fünf `N` und vier Schrittlängen.
- **Randfälle** (`tests/test_edge_cases.py`): `N = 1`, `h → 0`, `h = 2`, `h > 2`, großes `μ` mit Status „stalled“, alle Reglerecken für jedes Verfahren, **die zwischengespeicherten Ergebnisse sind mit pickle serialisierbar** (der Test hat einen Fehler des Handbeispiels gefunden, siehe unten), Toleranz des Lösers.
- **Verdikte** (`tests/test_verdicts.py`, Werte von Hand gesetzt), **Claims** (`tests/test_claims.py`: jede Zahl dieser README), **Oberfläche** (AppTest: jeder Button, jeder Regler an Minimum und Maximum, jedes Verfahren, jedes μ, die Extreme von `h` und `N`, PDF), Presets (die Zahlen der Hilfetexte), Permalink, Layout, Abbildungen, Footer, Laufzeit-Importe, Kern (Prüfsumme gegen das Schwester-Repo).
- **Ein Fehler des Baus:** das Handbeispiel gab die Funktion `f₁` (eine lokale Funktion) im zwischengespeicherten Ergebnis zurück; Streamlit speichert mit pickle und brach ab. Die App läuft jetzt mit Stützpunkten der Kurve; der Test `test_the_cached_results_can_be_pickled` sichert es.
- **Umfang:** 139 Tests (7 davon, die cvxpy-Orakel gegen Clarabel und SCS, überspringen sich ohne cvxpy; mit cvxpy laufen alle 12 Orakel-Tests grün), rund 2,5 Minuten (2026-11-03, Windows); kein Mutationstest (seit 2026-10-28 nicht mehr Teil des Ablaufs). cvxpy ist Gegenprobe im dev-Satz; ohne es überspringen sich die Orakel-Tests. Linux-CI: noch nicht gelaufen (Repo noch nicht veröffentlicht).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App (Skelett nach DEMO-PLAYBOOK §2) |
| `kegel_kern.py` | Gemeinsamer Kern der Reihe, wortgleich mit Stück 2 bis 10 |
| `kwc_pep.py` | Das PEP: Verfahren, Interpolationszeilen, Kegelprogramm, Lösen, Dual und Identität, Formeln der Literatur |
| `kwc_solver.py`, `kwc_cones.py` | Primal-dualer innerer Punkt (Mehrotra, Nesterov–Todd) mit Dualvariablen, aus Stück 6 |
| `kwc_constants.py` | Regler-Grenzen, feste Reihen, Farben, Presets |
| `kwc_evaluation.py` | Raten, Scan, Vergleich, μ-Tabelle, Handbeispiel, Beweis-Ansicht, Verdikte, Formatierung |
| `kwc_layout.py` | Fester Kopf (Stylesheet) |
| `kwc_pdf_export.py` | PDF-Export (fpdf2) |
| `kwc_presets.py` | Permalink, Presets (Schnellstart, aktives Preset hervorgehoben) |
| `kwc_visualization.py` | Plotly-Abbildungen (2D-Achsen gesperrt) |
| `requirements-dev.txt` | Laufzeit plus pytest, scipy und cvxpy (Gegenproben) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `tests/` | PEP, Löser, Orakel, Randfälle, Verdikte, Claims, Presets, Layout, Abbildungen, AppTest, Footer, Laufzeit-Importe, Kern |

## Bewusst nicht umgesetzt

- **Eigene Verfahren des Nutzers** (Eingabe von Schrittkoeffizienten `h_{i,k}`): würde die Eingabe prüfen müssen; Stück 16 nennt die Software PEPit/PESTO für beliebige Verfahren.
- **Andere Ziele** (Norm des Gradienten, Abstand `‖x_N − x*‖`) und **andere Funktionsklassen** (Lipschitz-stetig, proximal, ADMM): nicht gebaut.
- **Eine dünne Duallösung** (Beweisskizze wie in den Arbeiten): nicht gesucht.
- **Der Beweis der Vermutung für `1 < h < 2`:** nicht geführt, nur numerisch bestätigt.
- **Mutationstest:** nicht Teil des Ablaufs.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Streamlit Cloud: Python-Version in den App-Einstellungen prüfen (lokal Python 3.14, CI Python 3.12); der Code verwendet nur Merkmale, die in Python 3.12 laufen. <!-- lint-ok: dezimalpunkt -->

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy, pandas, fpdf2 und threadpoolctl.

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
