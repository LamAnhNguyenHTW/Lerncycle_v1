# Pilot Ground Truth – Manual Review

Diese Datei ist eine Prüfliste für die fachliche Freigabe des Pilotdatensatzes.
Eine Checkbox wird erst nach Sichtprüfung der angegebenen PDF-Seiten abgehakt.

## pilot-gpm-01

- **Frage:** Wodurch unterscheidet sich ein Geschäftsprozess von einem allgemeinen Prozess?
- **Dokument:** Geschäftsprozessoptimierung und -modellierung
- **Seiten:** 5
- **Fragetyp:** `fact`
- **Referenzantwort:** Ein Prozess besteht aus mehreren Einzelschritten, verarbeitet unter anderem Informationen als Input und führt zu einem gewünschten Output. Ein Geschäftsprozess verfolgt zusätzlich ein unternehmensbezogenes, aus der Unternehmensstrategie abgeleitetes Ziel und benötigt in der Regel Unterstützung durch Softwaresysteme und gegebenenfalls weitere Ressourcen.
- **extraction_risk:** `false`
- **manual_review_required:** `false`

- [ ] Ist die Frage eindeutig?
- [ ] Ist sie anhand des Korpus beantwortbar?
- [ ] Stimmen Quelle und Seiten?
- [ ] Ist die Referenzantwort korrekt?
- [ ] Ist der Fragetyp passend?
- [ ] Werden wirklich mehrere Chunks benötigt?
- [ ] Ist die Frage für den späteren Strategievergleich geeignet?

## pilot-gpm-02

- **Frage:** Wie führen Ist-Analyse, Soll-Konzeption sowie Entscheidung und Umsetzung zu einem neuen Überwachungszyklus der Geschäftsprozessoptimierung?
- **Dokument:** Geschäftsprozessoptimierung und -modellierung
- **Seiten:** 9, 10, 13, 14
- **Fragetyp:** `relational`
- **Referenzantwort:** Die Ist-Analyse erfasst den aktuellen Prozess, seine Abläufe, Systeme und Kennzahlen. Darauf aufbauend wird ein Soll-Prozess konzipiert und nach Kriterien wie Qualität, Zeit und Kosten bewertet. Nach der Entscheidung wird die gewählte Konzeption durch Prozesseinführung, Systemanpassungen und Schulungen umgesetzt. Die anschließende Prozessüberwachung vergleicht Ausführung und Soll; bei Abweichungen beginnt ein neuer Optimierungszyklus.
- **extraction_risk:** `false`
- **manual_review_required:** `false`

- [ ] Ist die Frage eindeutig?
- [ ] Ist sie anhand des Korpus beantwortbar?
- [ ] Stimmen Quelle und Seiten?
- [ ] Ist die Referenzantwort korrekt?
- [ ] Ist der Fragetyp passend?
- [ ] Werden wirklich mehrere Chunks benötigt?
- [ ] Ist die Frage für den späteren Strategievergleich geeignet?

## pilot-gpm-03

- **Frage:** Wie unterscheiden sich ein datenbasiertes exklusives Gateway und ein paralleles Gateway hinsichtlich Pfadauswahl und Zusammenführung?
- **Dokument:** Geschäftsprozessoptimierung und -modellierung
- **Seiten:** 24, 25
- **Fragetyp:** `table_or_figure`
- **Referenzantwort:** Beim datenbasierten exklusiven Gateway schließen sich die Alternativen gegenseitig aus; die Auswahl basiert auf bereits vorliegenden Daten. Das parallele Gateway aktiviert beziehungsweise vereinigt parallele Pfade und wartet beim schließenden Gateway, bis alle Pfade abgeschlossen sind.
- **extraction_risk:** `true`
- **manual_review_required:** `true`

- [x] Ist die Frage eindeutig?
- [x] Ist sie anhand des Korpus beantwortbar?
- [x] Stimmen Quelle und Seiten?
- [x] Ist die Referenzantwort korrekt?
- [x] Ist der Fragetyp passend?
- [x] Werden wirklich mehrere Chunks benötigt?
- [x] Ist die Frage für den späteren Strategievergleich geeignet?

## pilot-pwl-01

- **Frage:** Welche sechs 'R' beschreiben nach Jünnemann den logistischen Auftrag?
- **Dokument:** Produktionswirtschaft und Logistik – Vorlesung 1
- **Seiten:** 7
- **Fragetyp:** `terminology`
- **Referenzantwort:** Der logistische Auftrag stellt die richtige Menge der richtigen Objekte am richtigen Ort, zum richtigen Zeitpunkt, in der richtigen Qualität und zu den richtigen Kosten bereit.
- **extraction_risk:** `false`
- **manual_review_required:** `false`

- [ ] Ist die Frage eindeutig?
- [ ] Ist sie anhand des Korpus beantwortbar?
- [ ] Stimmen Quelle und Seiten?
- [ ] Ist die Referenzantwort korrekt?
- [ ] Ist der Fragetyp passend?
- [ ] Werden wirklich mehrere Chunks benötigt?
- [ ] Ist die Frage für den späteren Strategievergleich geeignet?

## pilot-pwl-02

- **Frage:** Warum entsteht zwischen hoher Lieferfähigkeit und niedrigen Logistikkosten ein Zielkonflikt, und welche Lösung nennt die Vorlesung?
- **Dokument:** Produktionswirtschaft und Logistik – Vorlesung 1
- **Seiten:** 16, 18, 22, 23, 24
- **Fragetyp:** `multi_hop`
- **Referenzantwort:** Die Logistik soll geforderte Leistungen wie Lieferfähigkeit und Lieferzuverlässigkeit einhalten und zugleich die dafür notwendigen Kosten minimieren. Hohe Lieferfähigkeit kann durch Bestände oder Flexibilität unterstützt werden, während Bestands-, Transport-, Verpackungs- und Prozesskosten gesenkt werden sollen. Die Vorlesung fordert deshalb abteilungsübergreifende Kompromisse zur Lösung dieser Zielkonflikte.
- **extraction_risk:** `false`
- **manual_review_required:** `false`

- [ ] Ist die Frage eindeutig?
- [ ] Ist sie anhand des Korpus beantwortbar?
- [ ] Stimmen Quelle und Seiten?
- [ ] Ist die Referenzantwort korrekt?
- [ ] Ist der Fragetyp passend?
- [ ] Werden wirklich mehrere Chunks benötigt?
- [ ] Ist die Frage für den späteren Strategievergleich geeignet?

## pilot-pwl-03

- **Frage:** Welche Bedeutung hat die Klasse AZ in einer kombinierten ABC/XYZ-Analyse?
- **Dokument:** Produktionswirtschaft und Logistik – Vorlesung 1
- **Seiten:** 30, 33, 35
- **Fragetyp:** `table_or_figure`
- **Referenzantwort:** A kennzeichnet einen hohen wertmäßigen Anteil am Gesamtwert, Z einen stark schwankenden beziehungsweise unregelmäßigen Bedarf. Ein AZ-Artikel verbindet daher einen hohen Wertanteil mit unregelmäßigem Bedarf.
- **extraction_risk:** `true`
- **manual_review_required:** `true`

- [ ] Ist die Frage eindeutig?
- [ ] Ist sie anhand des Korpus beantwortbar?
- [ ] Stimmen Quelle und Seiten?
- [ ] Ist die Referenzantwort korrekt?
- [ ] Ist der Fragetyp passend?
- [ ] Werden wirklich mehrere Chunks benötigt?
- [ ] Ist die Frage für den späteren Strategievergleich geeignet?

## pilot-http-01

- **Frage:** Wie unterscheiden sich POST, PUT und PATCH hinsichtlich ihrer beabsichtigten Wirkung auf eine Ressource?
- **Dokument:** Verteilte Anwendungen – HTTP
- **Seiten:** 11
- **Fragetyp:** `semantic_paraphrase`
- **Referenzantwort:** POST übermittelt eine Entität an die angegebene Ressource und verursacht häufig eine Zustandsänderung oder andere Seiteneffekte. PUT ersetzt alle aktuellen Repräsentationen der Zielressource durch den Request-Payload. PATCH nimmt dagegen nur teilweise Änderungen an einer Ressource vor.
- **extraction_risk:** `false`
- **manual_review_required:** `true`

- [ ] Ist die Frage eindeutig?
- [ ] Ist sie anhand des Korpus beantwortbar?
- [ ] Stimmen Quelle und Seiten?
- [ ] Ist die Referenzantwort korrekt?
- [ ] Ist der Fragetyp passend?
- [ ] Werden wirklich mehrere Chunks benötigt?
- [ ] Ist die Frage für den späteren Strategievergleich geeignet?

## pilot-http-02

- **Frage:** Welche der Methoden GET, POST, PUT und DELETE sind laut Übersicht sicher und welche sind idempotent?
- **Dokument:** Verteilte Anwendungen – HTTP
- **Seiten:** 21
- **Fragetyp:** `table_or_figure`
- **Referenzantwort:** GET ist sicher und idempotent. POST ist weder sicher noch idempotent. PUT und DELETE sind nicht sicher, aber idempotent.
- **extraction_risk:** `true`
- **manual_review_required:** `true`

- [ ] Ist die Frage eindeutig?
- [ ] Ist sie anhand des Korpus beantwortbar?
- [ ] Stimmen Quelle und Seiten?
- [ ] Ist die Referenzantwort korrekt?
- [ ] Ist der Fragetyp passend?
- [ ] Werden wirklich mehrere Chunks benötigt?
- [ ] Ist die Frage für den späteren Strategievergleich geeignet?

## pilot-http-03

- **Frage:** Welche HTTP-Methode ist für eine teilweise Änderung einer Ressource vorgesehen, und wie lassen sich danach erfolgreiche Antworten, Clientfehler und Serverfehler anhand des Statuscodes unterscheiden?
- **Dokument:** Verteilte Anwendungen – HTTP
- **Seiten:** 11, 16
- **Fragetyp:** `multi_hop`
- **Referenzantwort:** PATCH ist für teilweise Änderungen an einer Ressource vorgesehen. Statuscodes von 200 bis 299 kennzeichnen erfolgreiche Antworten, 400 bis 499 Clientfehler und 500 bis 599 Serverfehler.
- **extraction_risk:** `false`
- **manual_review_required:** `true`

- [ ] Ist die Frage eindeutig?
- [ ] Ist sie anhand des Korpus beantwortbar?
- [ ] Stimmen Quelle und Seiten?
- [ ] Ist die Referenzantwort korrekt?
- [ ] Ist der Fragetyp passend?
- [ ] Werden wirklich mehrere Chunks benötigt?
- [ ] Ist die Frage für den späteren Strategievergleich geeignet?

## pilot-se-01

- **Frage:** Welche fünf Hauptphasen des Softwareentwicklungsprozesses zeigt die Übersicht?
- **Dokument:** Übersicht zum Prozess der Softwareentwicklung
- **Seiten:** 2
- **Fragetyp:** `fact`
- **Referenzantwort:** Die Übersicht nennt Planung und Analyse, Anforderungsdefinition, Entwurf, Implementation sowie Abnahme und Einführung.
- **extraction_risk:** `false`
- **manual_review_required:** `false`

- [ ] Ist die Frage eindeutig?
- [ ] Ist sie anhand des Korpus beantwortbar?
- [ ] Stimmen Quelle und Seiten?
- [ ] Ist die Referenzantwort korrekt?
- [ ] Ist der Fragetyp passend?
- [ ] Werden wirklich mehrere Chunks benötigt?
- [ ] Ist die Frage für den späteren Strategievergleich geeignet?

## pilot-se-02

- **Frage:** Welche konkrete Programmiersprache wird für die Implementierung im Softwareentwicklungsprozess vorgeschrieben?
- **Dokument:** Übersicht zum Prozess der Softwareentwicklung (Negativbeispiel)
- **Seiten:** keine
- **Fragetyp:** `unanswerable`
- **Referenzantwort:** Im Evaluationskorpus wird für diesen Softwareentwicklungsprozess keine konkrete Programmiersprache vorgeschrieben.
- **extraction_risk:** `false`
- **manual_review_required:** `true`

- [ ] Ist die Frage eindeutig?
- [ ] Ist sie anhand des Korpus nicht beantwortbar?
- [ ] Sind Quelle, Seiten und Phrasen leer?
- [ ] Ist die Referenzantwort korrekt?
- [ ] Ist der Fragetyp passend?
- [ ] Ist `requires_multiple_chunks` korrekt auf `false` gesetzt?
- [ ] Ist die Frage für den späteren Strategievergleich geeignet?

## pilot-selfrag-01

- **Frage:** What limitations of conventional RAG motivate SELF-RAG?
- **Dokument:** SELF-RAG
- **Seiten:** 1, 2
- **Fragetyp:** `contextual`
- **Referenzantwort:** Conventional RAG often retrieves a fixed number of passages whether retrieval is needed or not, may introduce irrelevant or off-topic context, and does not guarantee that generated claims follow the retrieved evidence. SELF-RAG is motivated by the goal of retrieving on demand and evaluating passage relevance and generation support without reducing the model's versatility.
- **extraction_risk:** `false`
- **manual_review_required:** `false`

- [ ] Ist die Frage eindeutig?
- [ ] Ist sie anhand des Korpus beantwortbar?
- [ ] Stimmen Quelle und Seiten?
- [ ] Ist die Referenzantwort korrekt?
- [ ] Ist der Fragetyp passend?
- [ ] Werden wirklich mehrere Chunks benötigt?
- [ ] Ist die Frage für den späteren Strategievergleich geeignet?

## pilot-selfrag-02

- **Frage:** How do Retrieve, ISREL, ISSUP, and ISUSE differ in what they evaluate and in their output values?
- **Dokument:** SELF-RAG
- **Seiten:** 3, 16
- **Fragetyp:** `terminology`
- **Referenzantwort:** Retrieve decides whether retrieval is needed, with Yes, No, or Continue. ISREL judges whether a retrieved passage is relevant or irrelevant. ISSUP rates whether the output is fully supported, partially supported, or unsupported or contradictory with respect to evidence. ISUSE rates the response's perceived usefulness on a five-point scale from 1 to 5.
- **extraction_risk:** `true`
- **manual_review_required:** `true`

- [ ] Ist die Frage eindeutig?
- [ ] Ist sie anhand des Korpus beantwortbar?
- [ ] Stimmen Quelle und Seiten?
- [ ] Ist die Referenzantwort korrekt?
- [ ] Ist der Fragetyp passend?
- [ ] Werden wirklich mehrere Chunks benötigt?
- [ ] Ist die Frage für den späteren Strategievergleich geeignet?

## pilot-selfrag-03

- **Frage:** How are the SELF-RAG critic and generator trained, and how does the critic's output become generator training data?
- **Dokument:** SELF-RAG
- **Seiten:** 4, 5, 17, 18
- **Fragetyp:** `multi_hop`
- **Referenzantwort:** GPT-4 feedback is first collected as reflection-token supervision and used to fine-tune a critic with a next-token objective. The trained critic then predicts retrieval and critique tokens for the original input-output data; when retrieval is needed, passages and relevance, support, and utility judgments are added. These augmented examples form Dgen, on which the generator learns both the task output and reflection tokens with a standard next-token objective, while retrieved text is masked from the loss.
- **extraction_risk:** `true`
- **manual_review_required:** `true`

- [ ] Ist die Frage eindeutig?
- [ ] Ist sie anhand des Korpus beantwortbar?
- [ ] Stimmen Quelle und Seiten?
- [ ] Ist die Referenzantwort korrekt?
- [ ] Ist der Fragetyp passend?
- [ ] Werden wirklich mehrere Chunks benötigt?
- [ ] Ist die Frage für den späteren Strategievergleich geeignet?

## pilot-selfrag-04

- **Frage:** Why can always retrieving the top passage hurt SELF-RAG performance, and how do adaptive retrieval and critique weights provide an alternative?
- **Dokument:** SELF-RAG
- **Seiten:** 5, 6, 9
- **Fragetyp:** `relational`
- **Referenzantwort:** Always using the top retrieved passage can add evidence regardless of its relevance; the reported ablation lowers performance on PopQA and ASQA, while removing ISSUP also harms ASQA. SELF-RAG instead triggers retrieval from the predicted Retrieve token or a configurable threshold and ranks parallel continuations with weighted ISREL, ISSUP, and ISUSE scores. Those weights let practitioners emphasize evidence support or other behavior without retraining.
- **extraction_risk:** `false`
- **manual_review_required:** `true`

- [ ] Ist die Frage eindeutig?
- [ ] Ist sie anhand des Korpus beantwortbar?
- [ ] Stimmen Quelle und Seiten?
- [ ] Ist die Referenzantwort korrekt?
- [ ] Ist der Fragetyp passend?
- [ ] Werden wirklich mehrere Chunks benötigt?
- [ ] Ist die Frage für den späteren Strategievergleich geeignet?
