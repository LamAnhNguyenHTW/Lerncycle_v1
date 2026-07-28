# Review des Concept-Graph-Beitrags

Stand: 21.07.2026  
Grundlage: `concept_graph_comparison_fixed_size_final_v1_corrected_20260721.json`

Der Vergleich umfasst die 16 als `relational` oder `multi_hop` markierten
Fragen aus `ground-truth-final-v1`. Bewertet wurden (a) die Verfügbarkeit von
Graphkontext, (b) zusätzlich abgedeckte Ground-Truth-Seiten und (c) die
fachliche Verwendbarkeit der zurückgegebenen Beziehungen. Eine Seite gilt
nicht automatisch als fachlich korrekter Graphbeitrag, nur weil sie in der
Ground Truth liegt.

| query_id | Graphkontext | zusätzliche relevante Seiten | fachliche Prüfung der Beziehungen |
|---|---|---|---|
| `pilot-gpm-02` | ja | 10 | teilweise tragfähig; mehrere passende Prozessbeziehungen, aber auch inverse bzw. doppelte Kanten |
| `pilot-pwl-02` | ja | 18 | teilweise tragfähig; Zielkonflikt erkennbar, einzelne irrelevante Metadatenkante |
| `pilot-http-03` | nein | – | kein Graphbeitrag |
| `pilot-selfrag-03` | ja | 17, 18 | teilweise tragfähig; Trainingsobjekte gefunden, mehrere Relationen sind zu grob oder semantisch unklar |
| `pilot-selfrag-04` | ja | 6 | teilweise tragfähig; Critique-Tokens korrekt gruppiert, adaptive Logik nicht vollständig repräsentiert |
| `final-gpm-10` | ja | – | teilweise tragfähig; Sequenzfluss/Lanes erkannt, mindestens eine Hierarchierelation ist invertiert |
| `final-gpm-11` | ja | – | unzureichend; Simulation erkannt, aber die gefragte Experimentkette nicht abgebildet |
| `final-pwl-12` | ja | – | teilweise tragfähig; Kosten-/Nutzenaspekte erkannt, der methodische Kontrast bleibt unvollständig |
| `final-pwl-13` | ja | – | unzureichend; nur ein Teilaspekt der Abrechnung abgebildet |
| `final-http-10` | ja | – | teilweise tragfähig; Cookie-/Session-Begriffe erkannt, einzelne Kantenrichtungen sind fehlerhaft |
| `final-http-11` | ja | – | teilweise tragfähig; URL und Requestbestandteile erkannt, zusätzliche generische URL-Kanten sind nicht antwortrelevant |
| `final-se-08` | ja | – | teilweise tragfähig; Architektur und Anforderungen erkannt, mehrere Relationen sind invertiert |
| `final-se-09` | ja | – | unzureichend; keine Beziehung auf einer relevanten Ground-Truth-Seite |
| `final-selfrag-10` | ja | 3 | unzureichend; ISREL/ISSUP-Zusammenhang nicht korrekt ausgedrückt |
| `final-selfrag-11` | ja | 4 | unzureichend; relevante Begriffe gefunden, mehrere Subjekt-Objekt-Zuordnungen sind fachlich falsch |
| `final-selfrag-12` | ja | – | unzureichend; einzelne Ablations-/Metrikbegriffe, aber kein vollständiger Prüfzusammenhang |

## Zusammenfassung

- Graphkontext war bei 15 von 16 Fragen verfügbar (45 von 48 Beobachtungen).
- Bei 6 von 16 Fragen ergänzte der Graph mindestens eine relevante Seite; über
  drei Wiederholungen waren dies 18 von 48 Beobachtungen.
- Die zusätzliche Graphlatenz betrug im Mittel 15,0 ms, p50 11,9 ms und p95
  22,4 ms. Sie wurde getrennt von der Vektorlatenz gemessen.
- Keine der 16 Fragen erhielt einen durchgängig vollständigen und fehlerfreien
  relationalen Kontext. Neun Fragen wurden als teilweise tragfähig, sieben als
  unzureichend oder ohne Beitrag bewertet.
- Häufige Fehlerbilder sind invertierte Kanten, doppelte Beziehungen,
  irrelevante Metadatenbeziehungen und zu grobe Relationstypen.

Der Graph zeigt damit einen messbaren zusätzlichen Seitenzugriff, aber noch
keinen hinreichend zuverlässigen fachlichen Zusatznutzen für eine ungeprüfte
Standardaktivierung. Eine Antwortqualitätsmessung mit und ohne Graph wurde in
diesem Lauf nicht durchgeführt; Aussagen zu Faithfulness der final erzeugten
Antworten wären daher nicht zulässig.
