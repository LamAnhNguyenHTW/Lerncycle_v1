# Technische Evaluation

Dieses Verzeichnis enthält die für die Bachelorarbeit verwendeten Artefakte der technischen Evaluation des Retrieval-Augmented-Generation-Systems von LearnCycle.

## Evaluationsdatensatz

`questions.json` und `ground_truth.json` enthalten absichtlich denselben eingefrorenen Datensatz mit 60 Einträgen. Jeder Eintrag verbindet eine Evaluationsfrage mit ihren Relevanzmarkierungen und einer Referenzantwort. Die beiden Dateinamen werden bereitgestellt, damit Auswertungswerkzeuge mit unterschiedlichen Namenskonventionen verwendet werden können.

Die Metadaten zur Versionierung und Integritätsprüfung befinden sich in `configurations/ground_truth_freeze.json`. Der Datensatz trägt die Kennung `ground-truth-final-v1`. Inhaltliche Änderungen erfordern eine neue Kennung, eine erneute manuelle Prüfung und neue SHA-256-Prüfsummen.

## Evaluationskorpus

Das Korpus umfasst fünf Dokumente mit insgesamt 150 Seiten. `corpus_manifest.csv` dokumentiert für jedes Dokument eine stabile Quellenkennung, den Dateinamen, den Titel, die Sprache, den Quellentyp und die Seitenzahl.

Die PDF-Dateien selbst sind nicht Bestandteil dieses Abgabeverzeichnisses. Vor einer Reproduktion müssen die Originaldokumente über die in der Bachelorarbeit dokumentierten Quellen beziehungsweise Zugangswege beschafft werden. Dabei sind die jeweiligen Nutzungs- und Weitergaberechte zu beachten.

## Untersuchte Varianten

Die Evaluation vergleicht folgende Varianten:

- Dokumentverarbeitung und Chunking: `fixed_size`, `docling` und `docling_semantic`
- Retrieval: Dense Retrieval, Sparse Retrieval und hybrides Retrieval
- Reranking: ohne Reranking, Cross-Encoder-Reranking und LLM-basiertes Reranking
- ergänzende Untersuchung des Concept Graph

Alle finalen Vergleiche wurden mit drei Wiederholungen, einem nicht berücksichtigten Aufwärmdurchlauf, zyklisch variierter Ausführungsreihenfolge und deaktivierten Prozess-Caches durchgeführt. Die vollständige Versuchskonfiguration befindet sich in `configurations/final_v1.json`.

## Ergebnisdateien

`raw_results/` enthält die finalen maschinenlesbaren Berichte zum Chunking, Retrieval, Cross-Encoder-Reranking, LLM-basierten Reranking, Concept Graph und zur lokal abgeleiteten Robustheitsauswertung.

`summary_metrics.csv` fasst die Mittelwerte der wiederholten Messungen für die Robustheitsauswertung mit 45 Fragen zusammen. Davon sind 43 Fragen beantwortbar; zwei Fragen dienen als negative Kontrollfälle. Die negativen Kontrollfälle fließen nicht in die Relevanzmetriken ein.

Die ausgewiesenen Latenzen hängen von der verwendeten Hardware, der Netzwerkverbindung und den externen Diensten ab. Sie sind daher nicht als anbieterunabhängige Leistungswerte zu interpretieren.

## Metriken

Die Auswertung verwendet insbesondere Hit@1, Hit@3, Hit@5, Mean Reciprocal Rank (MRR), Precision@5, Recall@5 und nDCG@5. Zusätzlich werden Laufzeiten pro Frage sowie aggregierte Latenzwerte erfasst.

## Reproduktion

Die Befehle sind aus dem Wurzelverzeichnis des Repositorys in einer konfigurierten Python-Umgebung auszuführen. Die Dateien unter `scripts/` sind für die Abgabe erstellte Kopien der maßgeblichen Evaluationsmodule. Ihre Importe beziehen sich auf das Python-Paket `rag_pipeline` dieses Repositorys.

Beispiel für die erneute Ausführung des Retrieval-Vergleichs:

```bash
python -m rag_pipeline.eval.run_comparisons --phase retrieval \
  --queries research/technical_evaluation/ground_truth.json \
  --collection-strategy fixed_size --repetitions 3 \
  --out research/technical_evaluation/raw_results/reproduced_retrieval.json
```

Für einen tatsächlichen Retrieval-Durchlauf müssen OpenAI und Qdrant separat über Umgebungsvariablen konfiguriert werden. Zugangsdaten sind nicht Bestandteil dieses Verzeichnisses und dürfen nicht in Ergebnis- oder Konfigurationsdateien eingetragen werden.

LLM-basiertes Reranking und die Concept-Graph-Auswertung können Evaluationsinhalte an den jeweils konfigurierten Modellanbieter übertragen. Solche Durchläufe dürfen deshalb nur unter Beachtung der geltenden Datenschutz- und Nutzungsbedingungen ausgeführt werden.
