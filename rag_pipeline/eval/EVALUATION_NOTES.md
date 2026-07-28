# Manuelle Entscheidungen für die technische Evaluation

Stand: 21.07.2026

Diese Entscheidungen ergänzen die automatischen Coverage-Reports. Sie ändern
weder Ground-Truth-Seiten noch Retrievalmetriken.

## Produktionswirtschaft und Logistik

- Die in der Docling-Coverage als fehlend gemeldeten Seiten 6, 10, 15, 21, 25,
  37 und 42 sind Gliederungsseiten.
- Seite 17 entspricht inhaltlich und visuell weitgehend Seite 16; sie enthält
  lediglich eine kleine Änderung an der Abbildung.
- Keine dieser Seiten ist als relevante Seite einer Pilotfrage markiert.
- Die Coverage-Abweichungen werden weiterhin im Ergebnisreport ausgewiesen,
  gelten nach manueller Prüfung aber nicht als Ground-Truth-Verlust für das
  vorliegende Fragenset.

## SELF-RAG

- Für `pilot-selfrag-02` bleiben die relevanten PDF-Seiten 3 und 16 verbindlich.
- Abweichende Top-5-Seiten sind deshalb als Retrievalfehler zu werten und nicht
  durch eine nachträgliche Änderung der Ground Truth zu korrigieren.

## Geltungsbereich

Die Evaluation und ihre Auswahlentscheidungen gelten ausschließlich für die
fünf Korpusdokumente, die 15 Pilotfragen, die dokumentierten Modelle und
Parameter sowie die lokale technische Umgebung dieses Messlaufs.
