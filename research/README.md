# Forschungsartefakte

Dieses Verzeichnis enthält die Forschungsartefakte, die für die Dokumentation und Reproduktion der Evaluation von LearnCycle relevant sind. Dazu gehören die technische Evaluation, die in der Anwendung eingesetzten Prompts sowie die Unterlagen zur Befragung und zum Betatest.

## Verzeichnisstruktur

- `survey/`: Fragebögen, anonymisierte Antworten und Codebuch der Befragung. Diese Inhalte werden separat ergänzt.
- `beta_test/`: Fragebogen sowie anonymisierte Antworten und Rückmeldungen aus dem Betatest. Diese Inhalte werden separat ergänzt.
- `technical_evaluation/`: Fragen und Ground Truth, Metadaten des Evaluationskorpus, dokumentierte Konfigurationen, Evaluationsskripte, maschinenlesbare Rohresultate und eine Zusammenfassung der ermittelten Metriken.
- `prompts/`: Dokumentation der für Chat, Guided Learning, Feynman-Technik, Lernkarten und Mocktests eingesetzten Prompts einschließlich ihrer maßgeblichen Stellen im Quellcode.

## Reproduzierbarkeit

Der Datensatz der technischen Evaluation ist unter der Kennung `ground-truth-final-v1` eingefroren. Prüfsummen dokumentieren den verwendeten Stand und ermöglichen die Kontrolle der Datenintegrität. Die maßgebliche Implementierung des Evaluationsverfahrens befindet sich unter `rag_pipeline/eval/`. Die Dateien in diesem Verzeichnis bilden eine für die Abgabe zusammengestellte Kopie der relevanten Forschungsartefakte.

Das Korpusmanifest enthält ausschließlich Metadaten. Die zugrunde liegenden PDF-Dokumente werden hier nicht erneut bereitgestellt, da die jeweiligen Weitergaberechte zu beachten sind.

## Datenschutz und sensible Daten

Die technischen Rohresultate enthalten Evaluationskennungen und Messwerte, jedoch keine personenbezogenen Daten. Die Konfigurationsdateien enthalten bewusst keine API-Schlüssel, Datenbankzugänge, Service-Role-Schlüssel oder sonstigen Zugangsdaten.

Personenbezogene Angaben aus Befragung und Betatest müssen vor der Aufnahme in dieses Verzeichnis anonymisiert werden. Freitextantworten sind zusätzlich auf indirekte Identifikationsmerkmale zu prüfen.
