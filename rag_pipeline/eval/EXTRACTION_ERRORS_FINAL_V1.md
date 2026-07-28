# Extraktionsfehler der finalen Evaluation

Stand: 21.07.2026  
Ground Truth: `ground-truth-final-v1`  
Collection: `eval_fixed_size`  
Isolierter Eval-Benutzer: `eval-pilot-v1`

## Dokumentextraktion und Chunking

Die drei vorhandenen Eval-Collections wurden vor dem Lauf auf Vollständigkeit
und Benutzerzuordnung geprüft. Sie enthalten 264 Fixed-size-, 306 Docling- und
305 Docling-Semantic-Chunks. Beim erneuten Messlauf wurden die Collections
nicht verändert. Die bereits beim Aufbau dokumentierten fehlenden Docling-Seiten
6, 10, 15, 21, 25, 37 und 42 im Dokument Produktionswirtschaft/Logistik sind
Gliederungsseiten und werden von keiner Ground-Truth-Frage als relevant
referenziert. Seite 17 ist inhaltlich nahezu identisch zu Seite 16. Diese
bekannten Extraktionsabweichungen wurden nicht als Retrievaltreffer umgedeutet.

## Concept-Graph-Extraktion

Für die 264 Fixed-size-Chunks wurden höchstens drei Extraktionsversuche
durchgeführt. Erfolgreiche Chunks wurden bei Wiederholungen nicht erneut an den
LLM-Anbieter übertragen.

| Versuch | angefordert | erfolgreich | fehlgeschlagen | Dauer |
|---|---:|---:|---:|---:|
| initial | 264 | 182 | 82 | 577,76 s |
| Retry 1 | 82 | 38 | 44 | 340,98 s |
| Retry 2 | 44 | 14 | 30 | 330,71 s |
| **eindeutig erfolgreich** | **264** | **234** | **30** | **1.249,45 s gesamt** |

Damit wurden 88,6 % der Chunks erfolgreich in den isolierten Concept Graph
übernommen. Alle 30 endgültigen Fehler waren vom Typ `GraphExtractionError`.
Exception-Texte, Chunktexte, lokale Pfade und Zugangsdaten wurden nicht in den
Ergebnisdateien gespeichert.

Die verbliebenen Fehler verteilen sich wie folgt. Mehrfach genannte Seiten
enthalten mehrere fehlgeschlagene Chunks.

| Source-ID | fehlgeschlagene Chunks | PDF-Seiten |
|---|---:|---|
| `gpaa-geschaeftsprozessmanagement` | 2 | 6, 7 |
| `pwl-logistik-wise-2025` | 4 | 19, 36, 40, 41 |
| `se-prozess-uebersicht-2023` | 1 | 10 |
| `self-rag-iclr-2024` | 20 | 1, 3, 6 (2x), 7 (3x), 9, 10, 16, 17 (2x), 18, 19 (2x), 20 (2x), 23, 24, 26 |
| `verteilte-anwendungen-http` | 3 | 11, 15, 18 |

Diese Ausfälle betreffen ausschließlich den Concept Graph. Die Vektor- und
Sparse-Indizes enthalten weiterhin alle 264 Fixed-size-Chunks. Der
Graphvergleich ist deshalb als Messung eines unvollständigen Graphen zu
interpretieren.

## Fehler in der Graph-Abfragelogik

Der erste Graphvergleich lieferte nur für 15 von 48 Beobachtungen Kontext. Die
Ursache war eine fehlerhafte Zuordnung natürlicher Fragen zu Concept-Namen. Der
Lookup wurde anschließend quellenspezifisch gemacht, auf normalisierte
Query-Terme umgestellt und deterministisch gerankt. Der erste Bericht bleibt
als Diagnoseartefakt erhalten, wird aber nicht für die Ergebnisinterpretation
verwendet. Maßgeblich ist ausschließlich
`concept_graph_comparison_fixed_size_final_v1_corrected_20260721.json`.

## LLM-Reranking

Der separat freigegebene LLM-Reranking-Lauf wurde für 60 Fragen, höchstens 20
Kandidaten und drei Wiederholungen ausgeführt. Ein LLM-Aufruf innerhalb des
Laufs lieferte syntaktisch ungültiges JSON (`JSONDecodeError`). Die
Reranking-Implementierung fiel kontrolliert auf die ursprüngliche Dense-
Rangfolge zurück. Der Bericht markiert Fallbacks noch nicht gesondert; daher
lässt sich nicht nachträglich bestimmen, ob der Fehler im ausgeschlossenen
Warm-up oder in einer der 180 Messbeobachtungen auftrat. Der Lauf wurde nicht
wiederholt. Promptinhalte, Antworttexte und Exception-Texte wurden nicht in der
Ergebnisdatei gespeichert. Maßgeblich ist
`reranking_llm_dense_fixed_size_final_v1_20260721.json`.
