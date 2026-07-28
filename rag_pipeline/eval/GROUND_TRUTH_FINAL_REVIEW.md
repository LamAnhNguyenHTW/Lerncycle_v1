# Final Ground Truth ? Manual Review

Der vollst?ndige Datensatz wurde am 21.07.2026 manuell freigegeben und als `ground-truth-final-v1` eingefroren. ?nderungen erfordern eine neue Freeze-Version und neue Pr?fsummen.

## pilot-gpm-01

- **Frage:** Wodurch unterscheidet sich ein Geschäftsprozess von einem allgemeinen Prozess?
- **Dokument:** Geschäftsprozessoptimierung und -modellierung
- **Seiten:** 5
- **Fragetyp:** `fact`
- **Referenzantwort:** Ein Prozess besteht aus mehreren Einzelschritten, verarbeitet unter anderem Informationen als Input und führt zu einem gewünschten Output. Ein Geschäftsprozess verfolgt zusätzlich ein unternehmensbezogenes, aus der Unternehmensstrategie abgeleitetes Ziel und benötigt in der Regel Unterstützung durch Softwaresysteme und gegebenenfalls weitere Ressourcen.
- **extraction_risk:** `false`
- **manual_review_required:** `false`

- [x] Ist die Frage eindeutig?
- [x] Ist sie anhand des Korpus beantwortbar?
- [x] Stimmen Quelle und Seiten?
- [x] Ist die Referenzantwort korrekt?
- [x] Ist der Fragetyp passend?
- [x] Werden wirklich mehrere Chunks benötigt?
- [x] Ist die Frage für den späteren Strategievergleich geeignet?

## pilot-gpm-02

- **Frage:** Wie führen Ist-Analyse, Soll-Konzeption sowie Entscheidung und Umsetzung zu einem neuen Überwachungszyklus der Geschäftsprozessoptimierung?
- **Dokument:** Geschäftsprozessoptimierung und -modellierung
- **Seiten:** 9, 10, 13, 14
- **Fragetyp:** `relational`
- **Referenzantwort:** Die Ist-Analyse erfasst den aktuellen Prozess, seine Abläufe, Systeme und Kennzahlen. Darauf aufbauend wird ein Soll-Prozess konzipiert und nach Kriterien wie Qualität, Zeit und Kosten bewertet. Nach der Entscheidung wird die gewählte Konzeption durch Prozesseinführung, Systemanpassungen und Schulungen umgesetzt. Die anschließende Prozessüberwachung vergleicht Ausführung und Soll; bei Abweichungen beginnt ein neuer Optimierungszyklus.
- **extraction_risk:** `false`
- **manual_review_required:** `false`

- [x] Ist die Frage eindeutig?
- [x] Ist sie anhand des Korpus beantwortbar?
- [x] Stimmen Quelle und Seiten?
- [x] Ist die Referenzantwort korrekt?
- [x] Ist der Fragetyp passend?
- [x] Werden wirklich mehrere Chunks benötigt?
- [x] Ist die Frage für den späteren Strategievergleich geeignet?

## pilot-gpm-03

- **Frage:** Wie unterscheiden sich ein datenbasiertes exklusives Gateway und ein paralleles Gateway hinsichtlich Pfadauswahl und Zusammenführung?
- **Dokument:** Geschäftsprozessoptimierung und -modellierung
- **Seiten:** 24, 25
- **Fragetyp:** `table_or_figure`
- **Referenzantwort:** Beim datenbasierten exklusiven Gateway schließen sich die Alternativen gegenseitig aus; die Auswahl basiert auf bereits vorliegenden Daten. Das parallele Gateway aktiviert beziehungsweise vereinigt parallele Pfade und wartet beim schließenden Gateway, bis alle Pfade abgeschlossen sind.
- **extraction_risk:** `true`
- **manual_review_required:** `false`

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

- [x] Ist die Frage eindeutig?
- [x] Ist sie anhand des Korpus beantwortbar?
- [x] Stimmen Quelle und Seiten?
- [x] Ist die Referenzantwort korrekt?
- [x] Ist der Fragetyp passend?
- [x] Werden wirklich mehrere Chunks benötigt?
- [x] Ist die Frage für den späteren Strategievergleich geeignet?

## pilot-pwl-02

- **Frage:** Warum entsteht zwischen hoher Lieferfähigkeit und niedrigen Logistikkosten ein Zielkonflikt, und welche Lösung nennt die Vorlesung?
- **Dokument:** Produktionswirtschaft und Logistik – Vorlesung 1
- **Seiten:** 16, 18, 22, 23, 24
- **Fragetyp:** `multi_hop`
- **Referenzantwort:** Die Logistik soll geforderte Leistungen wie Lieferfähigkeit und Lieferzuverlässigkeit einhalten und zugleich die dafür notwendigen Kosten minimieren. Hohe Lieferfähigkeit kann durch Bestände oder Flexibilität unterstützt werden, während Bestands-, Transport-, Verpackungs- und Prozesskosten gesenkt werden sollen. Die Vorlesung fordert deshalb abteilungsübergreifende Kompromisse zur Lösung dieser Zielkonflikte.
- **extraction_risk:** `false`
- **manual_review_required:** `false`

- [x] Ist die Frage eindeutig?
- [x] Ist sie anhand des Korpus beantwortbar?
- [x] Stimmen Quelle und Seiten?
- [x] Ist die Referenzantwort korrekt?
- [x] Ist der Fragetyp passend?
- [x] Werden wirklich mehrere Chunks benötigt?
- [x] Ist die Frage für den späteren Strategievergleich geeignet?

## pilot-pwl-03

- **Frage:** Welche Bedeutung hat die Klasse AZ in einer kombinierten ABC/XYZ-Analyse?
- **Dokument:** Produktionswirtschaft und Logistik – Vorlesung 1
- **Seiten:** 30, 33, 35
- **Fragetyp:** `table_or_figure`
- **Referenzantwort:** A kennzeichnet einen hohen wertmäßigen Anteil am Gesamtwert, Z einen stark schwankenden beziehungsweise unregelmäßigen Bedarf. Ein AZ-Artikel verbindet daher einen hohen Wertanteil mit unregelmäßigem Bedarf.
- **extraction_risk:** `true`
- **manual_review_required:** `false`

- [x] Ist die Frage eindeutig?
- [x] Ist sie anhand des Korpus beantwortbar?
- [x] Stimmen Quelle und Seiten?
- [x] Ist die Referenzantwort korrekt?
- [x] Ist der Fragetyp passend?
- [x] Werden wirklich mehrere Chunks benötigt?
- [x] Ist die Frage für den späteren Strategievergleich geeignet?

## pilot-http-01

- **Frage:** Wie unterscheiden sich POST, PUT und PATCH hinsichtlich ihrer beabsichtigten Wirkung auf eine Ressource?
- **Dokument:** Verteilte Anwendungen – HTTP
- **Seiten:** 11
- **Fragetyp:** `semantic_paraphrase`
- **Referenzantwort:** POST übermittelt eine Entität an die angegebene Ressource und verursacht häufig eine Zustandsänderung oder andere Seiteneffekte. PUT ersetzt alle aktuellen Repräsentationen der Zielressource durch den Request-Payload. PATCH nimmt dagegen nur teilweise Änderungen an einer Ressource vor.
- **extraction_risk:** `false`
- **manual_review_required:** `false`

- [x] Ist die Frage eindeutig?
- [x] Ist sie anhand des Korpus beantwortbar?
- [x] Stimmen Quelle und Seiten?
- [x] Ist die Referenzantwort korrekt?
- [x] Ist der Fragetyp passend?
- [x] Werden wirklich mehrere Chunks benötigt?
- [x] Ist die Frage für den späteren Strategievergleich geeignet?

## pilot-http-02

- **Frage:** Welche der Methoden GET, POST, PUT und DELETE sind laut Übersicht sicher und welche sind idempotent?
- **Dokument:** Verteilte Anwendungen – HTTP
- **Seiten:** 21
- **Fragetyp:** `table_or_figure`
- **Referenzantwort:** GET ist sicher und idempotent. POST ist weder sicher noch idempotent. PUT und DELETE sind nicht sicher, aber idempotent.
- **extraction_risk:** `true`
- **manual_review_required:** `false`

- [x] Ist die Frage eindeutig?
- [x] Ist sie anhand des Korpus beantwortbar?
- [x] Stimmen Quelle und Seiten?
- [x] Ist die Referenzantwort korrekt?
- [x] Ist der Fragetyp passend?
- [x] Werden wirklich mehrere Chunks benötigt?
- [x] Ist die Frage für den späteren Strategievergleich geeignet?

## pilot-http-03

- **Frage:** Welche HTTP-Methode ist für eine teilweise Änderung einer Ressource vorgesehen, und wie lassen sich danach erfolgreiche Antworten, Clientfehler und Serverfehler anhand des Statuscodes unterscheiden?
- **Dokument:** Verteilte Anwendungen – HTTP
- **Seiten:** 11, 16
- **Fragetyp:** `multi_hop`
- **Referenzantwort:** PATCH ist für teilweise Änderungen an einer Ressource vorgesehen. Statuscodes von 200 bis 299 kennzeichnen erfolgreiche Antworten, 400 bis 499 Clientfehler und 500 bis 599 Serverfehler.
- **extraction_risk:** `false`
- **manual_review_required:** `false`

- [x] Ist die Frage eindeutig?
- [x] Ist sie anhand des Korpus beantwortbar?
- [x] Stimmen Quelle und Seiten?
- [x] Ist die Referenzantwort korrekt?
- [x] Ist der Fragetyp passend?
- [x] Werden wirklich mehrere Chunks benötigt?
- [x] Ist die Frage für den späteren Strategievergleich geeignet?

## pilot-se-01

- **Frage:** Welche fünf Hauptphasen des Softwareentwicklungsprozesses zeigt die Übersicht?
- **Dokument:** Übersicht zum Prozess der Softwareentwicklung
- **Seiten:** 2
- **Fragetyp:** `fact`
- **Referenzantwort:** Die Übersicht nennt Planung und Analyse, Anforderungsdefinition, Entwurf, Implementation sowie Abnahme und Einführung.
- **extraction_risk:** `false`
- **manual_review_required:** `false`

- [x] Ist die Frage eindeutig?
- [x] Ist sie anhand des Korpus beantwortbar?
- [x] Stimmen Quelle und Seiten?
- [x] Ist die Referenzantwort korrekt?
- [x] Ist der Fragetyp passend?
- [x] Werden wirklich mehrere Chunks benötigt?
- [x] Ist die Frage für den späteren Strategievergleich geeignet?

## pilot-se-02

- **Frage:** Welche konkrete Programmiersprache wird für die Implementierung im Softwareentwicklungsprozess vorgeschrieben?
- **Dokument:** Übersicht zum Prozess der Softwareentwicklung (Negativbeispiel)
- **Seiten:** keine
- **Fragetyp:** `unanswerable`
- **Referenzantwort:** Im Evaluationskorpus wird für diesen Softwareentwicklungsprozess keine konkrete Programmiersprache vorgeschrieben.
- **extraction_risk:** `false`
- **manual_review_required:** `false`

- [x] Ist die Frage eindeutig?
- [x] Ist sie anhand des Korpus nicht beantwortbar?
- [x] Sind Quelle, Seiten und Phrasen leer?
- [x] Ist die Referenzantwort korrekt?
- [x] Ist der Fragetyp passend?
- [x] Ist `requires_multiple_chunks` korrekt auf `false` gesetzt?
- [x] Ist die Frage für den späteren Strategievergleich geeignet?

## pilot-selfrag-01

- **Frage:** What limitations of conventional RAG motivate SELF-RAG?
- **Dokument:** SELF-RAG
- **Seiten:** 1, 2
- **Fragetyp:** `contextual`
- **Referenzantwort:** Conventional RAG often retrieves a fixed number of passages whether retrieval is needed or not, may introduce irrelevant or off-topic context, and does not guarantee that generated claims follow the retrieved evidence. SELF-RAG is motivated by the goal of retrieving on demand and evaluating passage relevance and generation support without reducing the model's versatility.
- **extraction_risk:** `false`
- **manual_review_required:** `false`

- [x] Ist die Frage eindeutig?
- [x] Ist sie anhand des Korpus beantwortbar?
- [x] Stimmen Quelle und Seiten?
- [x] Ist die Referenzantwort korrekt?
- [x] Ist der Fragetyp passend?
- [x] Werden wirklich mehrere Chunks benötigt?
- [x] Ist die Frage für den späteren Strategievergleich geeignet?

## pilot-selfrag-02

- **Frage:** How do Retrieve, ISREL, ISSUP, and ISUSE differ in what they evaluate and in their output values?
- **Dokument:** SELF-RAG
- **Seiten:** 3, 16
- **Fragetyp:** `terminology`
- **Referenzantwort:** Retrieve decides whether retrieval is needed, with Yes, No, or Continue. ISREL judges whether a retrieved passage is relevant or irrelevant. ISSUP rates whether the output is fully supported, partially supported, or unsupported or contradictory with respect to evidence. ISUSE rates the response's perceived usefulness on a five-point scale from 1 to 5.
- **extraction_risk:** `true`
- **manual_review_required:** `false`

- [x] Ist die Frage eindeutig?
- [x] Ist sie anhand des Korpus beantwortbar?
- [x] Stimmen Quelle und Seiten?
- [x] Ist die Referenzantwort korrekt?
- [x] Ist der Fragetyp passend?
- [x] Werden wirklich mehrere Chunks benötigt?
- [x] Ist die Frage für den späteren Strategievergleich geeignet?

## pilot-selfrag-03

- **Frage:** How are the SELF-RAG critic and generator trained, and how does the critic's output become generator training data?
- **Dokument:** SELF-RAG
- **Seiten:** 4, 5, 17, 18
- **Fragetyp:** `multi_hop`
- **Referenzantwort:** GPT-4 feedback is first collected as reflection-token supervision and used to fine-tune a critic with a next-token objective. The trained critic then predicts retrieval and critique tokens for the original input-output data; when retrieval is needed, passages and relevance, support, and utility judgments are added. These augmented examples form Dgen, on which the generator learns both the task output and reflection tokens with a standard next-token objective, while retrieved text is masked from the loss.
- **extraction_risk:** `true`
- **manual_review_required:** `false`

- [x] Ist die Frage eindeutig?
- [x] Ist sie anhand des Korpus beantwortbar?
- [x] Stimmen Quelle und Seiten?
- [x] Ist die Referenzantwort korrekt?
- [x] Ist der Fragetyp passend?
- [x] Werden wirklich mehrere Chunks benötigt?
- [x] Ist die Frage für den späteren Strategievergleich geeignet?

## pilot-selfrag-04

- **Frage:** Why can always retrieving the top passage hurt SELF-RAG performance, and how do adaptive retrieval and critique weights provide an alternative?
- **Dokument:** SELF-RAG
- **Seiten:** 5, 6, 9
- **Fragetyp:** `relational`
- **Referenzantwort:** Always using the top retrieved passage can add evidence regardless of its relevance; the reported ablation lowers performance on PopQA and ASQA, while removing ISSUP also harms ASQA. SELF-RAG instead triggers retrieval from the predicted Retrieve token or a configurable threshold and ranks parallel continuations with weighted ISREL, ISSUP, and ISUSE scores. Those weights let practitioners emphasize evidence support or other behavior without retraining.
- **extraction_risk:** `false`
- **manual_review_required:** `false`

- [x] Ist die Frage eindeutig?
- [x] Ist sie anhand des Korpus beantwortbar?
- [x] Stimmen Quelle und Seiten?
- [x] Ist die Referenzantwort korrekt?
- [x] Ist der Fragetyp passend?
- [x] Werden wirklich mehrere Chunks benötigt?
- [x] Ist die Frage für den späteren Strategievergleich geeignet?

## final-gpm-04

- **Frage:** Welche Aufgaben umfasst die strategische Ebene des integrierten Geschäftsprozessmanagements?
- **Dokument:** Gesch?ftsprozessoptimierung und -modellierung
- **Seiten:** 2
- **Fragetyp:** `fact`
- **Referenzantwort:** Die strategische Ebene umfasst Strategieentwicklung und -steuerung, die Betrachtung der Geschäftsfelder einschließlich kritischer Erfolgsfaktoren, die Identifikation, Planung und Umsetzung zentraler Unternehmensprozesse sowie strategisches Prozesscontrolling anhand von Kennzahlen und gegebenenfalls Korrekturmaßnahmen.
- **extraction_risk:** `false`
- **manual_review_required:** `false`

- [x] Ist die Frage eindeutig?
- [x] Ist sie anhand des Korpus beantwortbar?
- [x] Stimmen Quelle und Seiten?
- [x] Ist die Referenzantwort korrekt?
- [x] Ist der Fragetyp passend?
- [x] Werden wirklich mehrere Chunks ben?tigt?
- [x] Ist die Frage f?r den sp?teren Strategievergleich geeignet?

## final-gpm-05

- **Frage:** Welche typischen Schwachstellen können bei der Analyse eines Ist-Prozesses sichtbar werden?
- **Dokument:** Gesch?ftsprozessoptimierung und -modellierung
- **Seiten:** 12
- **Fragetyp:** `fact`
- **Referenzantwort:** Als typische Schwachstellen nennt die Unterlage Doppelarbeiten aufgrund unklarer Zuständigkeiten, lange Wartezeiten etwa wegen Rückfragen, häufige Bearbeiterwechsel sowie Medienbrüche und falsche Dateneingaben.
- **extraction_risk:** `false`
- **manual_review_required:** `false`

- [x] Ist die Frage eindeutig?
- [x] Ist sie anhand des Korpus beantwortbar?
- [x] Stimmen Quelle und Seiten?
- [x] Ist die Referenzantwort korrekt?
- [x] Ist der Fragetyp passend?
- [x] Werden wirklich mehrere Chunks ben?tigt?
- [x] Ist die Frage f?r den sp?teren Strategievergleich geeignet?

## final-gpm-06

- **Frage:** Warum werden sowohl Ist- als auch Soll-Modelle von Geschäftsprozessen erstellt?
- **Dokument:** Gesch?ftsprozessoptimierung und -modellierung
- **Seiten:** 7
- **Fragetyp:** `semantic_paraphrase`
- **Referenzantwort:** Ist-Modelle bilden den Ausgangspunkt für die Analyse vorhandener Schwachstellen. Soll-Modelle beschreiben die angestrebte Gestaltung und dienen als Ausgangspunkt für eine Restrukturierung; Prozessmodelle können außerdem Anforderungen an IT-Unterstützung und automatisierbare Workflows ableiten.
- **extraction_risk:** `false`
- **manual_review_required:** `false`

- [x] Ist die Frage eindeutig?
- [x] Ist sie anhand des Korpus beantwortbar?
- [x] Stimmen Quelle und Seiten?
- [x] Ist die Referenzantwort korrekt?
- [x] Ist der Fragetyp passend?
- [x] Werden wirklich mehrere Chunks ben?tigt?
- [x] Ist die Frage f?r den sp?teren Strategievergleich geeignet?

## final-gpm-07

- **Frage:** Wie unterscheiden sich ein BPMN-Prozessdiagramm und ein Kollaborationsdiagramm hinsichtlich Fokus und Pool-Nutzung?
- **Dokument:** Gesch?ftsprozessoptimierung und -modellierung
- **Seiten:** 18
- **Fragetyp:** `semantic_paraphrase`
- **Referenzantwort:** Ein Prozessdiagramm fokussiert einen unternehmensinternen Geschäftsprozess und benötigt höchstens einen Pool, der mehrere Lanes enthalten kann. Ein Kollaborationsdiagramm stellt dagegen die Zusammenarbeit unabhängiger Prozesse in verschiedenen Pools dar, beispielsweise zwischen Kunde und Lieferant.
- **extraction_risk:** `false`
- **manual_review_required:** `false`

- [x] Ist die Frage eindeutig?
- [x] Ist sie anhand des Korpus beantwortbar?
- [x] Stimmen Quelle und Seiten?
- [x] Ist die Referenzantwort korrekt?
- [x] Ist der Fragetyp passend?
- [x] Werden wirklich mehrere Chunks ben?tigt?
- [x] Ist die Frage f?r den sp?teren Strategievergleich geeignet?

## final-gpm-08

- **Frage:** Was ist in BPMN ein Gateway, und welche Funktion hat es ausdrücklich nicht?
- **Dokument:** Gesch?ftsprozessoptimierung und -modellierung
- **Seiten:** 17
- **Fragetyp:** `terminology`
- **Referenzantwort:** Ein Gateway bildet die Logik für Verzweigungen oder Zusammenführungen von Sequenzflüssen ab. Es führt selbst keine Tätigkeit aus.
- **extraction_risk:** `false`
- **manual_review_required:** `false`

- [x] Ist die Frage eindeutig?
- [x] Ist sie anhand des Korpus beantwortbar?
- [x] Stimmen Quelle und Seiten?
- [x] Ist die Referenzantwort korrekt?
- [x] Ist der Fragetyp passend?
- [x] Werden wirklich mehrere Chunks ben?tigt?
- [x] Ist die Frage f?r den sp?teren Strategievergleich geeignet?

## final-gpm-09

- **Frage:** Wie beschreibt die Unterlage die wechselseitige Beziehung zwischen Geschäftsprozessen, IT-Systemen und Unternehmensstrategie?
- **Dokument:** Gesch?ftsprozessoptimierung und -modellierung
- **Seiten:** 6
- **Fragetyp:** `contextual`
- **Referenzantwort:** IT-Systeme werden anhand der Geschäftsprozesse gestaltet, während die IT-Strategie an der Geschäftsstrategie ausgerichtet und ihr untergeordnet ist. Umgekehrt können IT-Systeme Geschäftsprozesse verändern und Innovationen ermöglichen.
- **extraction_risk:** `false`
- **manual_review_required:** `false`

- [x] Ist die Frage eindeutig?
- [x] Ist sie anhand des Korpus beantwortbar?
- [x] Stimmen Quelle und Seiten?
- [x] Ist die Referenzantwort korrekt?
- [x] Ist der Fragetyp passend?
- [x] Werden wirklich mehrere Chunks ben?tigt?
- [x] Ist die Frage f?r den sp?teren Strategievergleich geeignet?

## final-gpm-10

- **Frage:** Warum darf ein Sequenzfluss zwischen Lanes verlaufen, ein Nachrichtenfluss dagegen nur zwischen Pools?
- **Dokument:** Gesch?ftsprozessoptimierung und -modellierung
- **Seiten:** 18, 19
- **Fragetyp:** `relational`
- **Referenzantwort:** Lanes untergliedern Zuständigkeiten innerhalb desselben Prozess-Pools, deshalb kann der interne Sequenzfluss mehrere Lanes durchlaufen. Ein Nachrichtenfluss stellt dagegen die Informationsübermittlung zwischen unabhängigen Teilnehmern dar und muss deshalb eine Poolgrenze überschreiten; innerhalb eines Pools wird er nicht verwendet.
- **extraction_risk:** `false`
- **manual_review_required:** `false`

- [x] Ist die Frage eindeutig?
- [x] Ist sie anhand des Korpus beantwortbar?
- [x] Stimmen Quelle und Seiten?
- [x] Ist die Referenzantwort korrekt?
- [x] Ist der Fragetyp passend?
- [x] Werden wirklich mehrere Chunks ben?tigt?
- [x] Ist die Frage f?r den sp?teren Strategievergleich geeignet?

## final-gpm-11

- **Frage:** Wie wird aus einem Prozessmodell ein Simulationsexperiment, und welche Ergebnisgrößen werden in der Demonstration sichtbar gemacht?
- **Dokument:** Gesch?ftsprozessoptimierung und -modellierung
- **Seiten:** 28, 29, 30
- **Fragetyp:** `multi_hop`
- **Referenzantwort:** Zunächst werden Zielsetzung und benötigte Informationen wie Durchführungszeiten und Kapazitäten festgelegt. Danach wird das Modell mit Bearbeitungsreihenfolgen gebildet, in einem Simulator implementiert und auf Übereinstimmung mit der Realität validiert. Es folgen Versuchsreihen und Ergebnisanalyse. In der Demonstration werden unter anderem Zeiten und Kosten je Aufgabe, Wahrscheinlichkeiten an Gateway-Sequenzflüssen sowie Wartezeiten hinterlegt und ausgewertet.
- **extraction_risk:** `false`
- **manual_review_required:** `false`

- [x] Ist die Frage eindeutig?
- [x] Ist sie anhand des Korpus beantwortbar?
- [x] Stimmen Quelle und Seiten?
- [x] Ist die Referenzantwort korrekt?
- [x] Ist der Fragetyp passend?
- [x] Werden wirklich mehrere Chunks ben?tigt?
- [x] Ist die Frage f?r den sp?teren Strategievergleich geeignet?

## final-gpm-12

- **Frage:** Welches konkrete Simulationssoftware-Produkt schreibt die Unterlage für Geschäftsprozesse verbindlich vor?
- **Dokument:** Gesch?ftsprozessoptimierung und -modellierung
- **Seiten:** keine
- **Fragetyp:** `unanswerable`
- **Referenzantwort:** Im Evaluationskorpus wird kein konkretes Simulationssoftware-Produkt für Geschäftsprozesse verbindlich vorgeschrieben.
- **extraction_risk:** `false`
- **manual_review_required:** `false`

- [x] Ist die Frage eindeutig?
- [x] Ist sie anhand des Korpus beantwortbar?
- [x] Stimmen Quelle und Seiten?
- [x] Ist die Referenzantwort korrekt?
- [x] Ist der Fragetyp passend?
- [x] Werden wirklich mehrere Chunks ben?tigt?
- [x] Ist die Frage f?r den sp?teren Strategievergleich geeignet?

## final-pwl-04

- **Frage:** Welche klassischen Bereiche der Logistik nennt die Vorlesung?
- **Dokument:** Produktionswirtschaft und Logistik ? Vorlesung 1
- **Seiten:** 11
- **Fragetyp:** `fact`
- **Referenzantwort:** Genannt werden Beschaffungslogistik, Produktionslogistik und Distributionslogistik sowie als vierter Bereich die Entsorgungslogistik.
- **extraction_risk:** `false`
- **manual_review_required:** `false`

- [x] Ist die Frage eindeutig?
- [x] Ist sie anhand des Korpus beantwortbar?
- [x] Stimmen Quelle und Seiten?
- [x] Ist die Referenzantwort korrekt?
- [x] Ist der Fragetyp passend?
- [x] Werden wirklich mehrere Chunks ben?tigt?
- [x] Ist die Frage f?r den sp?teren Strategievergleich geeignet?

## final-pwl-05

- **Frage:** Wann gelten Logistikleistungen laut Fazit als optimal?
- **Dokument:** Produktionswirtschaft und Logistik ? Vorlesung 1
- **Seiten:** 20
- **Fragetyp:** `fact`
- **Referenzantwort:** Logistikleistungen gelten als optimal, wenn alle angebotenen Leistungen verfügbar sind, Schwierigkeiten geräuschlos beseitigt werden, die Logistik über alle Informationen verfügt und dadurch alles unbemerkt funktioniert.
- **extraction_risk:** `false`
- **manual_review_required:** `false`

- [x] Ist die Frage eindeutig?
- [x] Ist sie anhand des Korpus beantwortbar?
- [x] Stimmen Quelle und Seiten?
- [x] Ist die Referenzantwort korrekt?
- [x] Ist der Fragetyp passend?
- [x] Werden wirklich mehrere Chunks ben?tigt?
- [x] Ist die Frage f?r den sp?teren Strategievergleich geeignet?

## final-pwl-06

- **Frage:** How do the modern logistics definitions connect material flows, information flows, and the customer?
- **Dokument:** Produktionswirtschaft und Logistik ? Vorlesung 1
- **Seiten:** 8, 9
- **Fragetyp:** `semantic_paraphrase`
- **Referenzantwort:** The definitions describe logistics as planning, controlling, executing, and monitoring material and information flows within and between companies. These flows and storage activities extend from suppliers to receiving points according to customer requirements, so logistics is treated as a holistic function across the whole chain.
- **extraction_risk:** `false`
- **manual_review_required:** `false`

- [x] Ist die Frage eindeutig?
- [x] Ist sie anhand des Korpus beantwortbar?
- [x] Stimmen Quelle und Seiten?
- [x] Ist die Referenzantwort korrekt?
- [x] Ist der Fragetyp passend?
- [x] Werden wirklich mehrere Chunks ben?tigt?
- [x] Ist die Frage f?r den sp?teren Strategievergleich geeignet?

## final-pwl-07

- **Frage:** Weshalb können C-Artikel trotz ihres niedrigen Einzelwerts erhebliche Beschaffungskosten verursachen?
- **Dokument:** Produktionswirtschaft und Logistik ? Vorlesung 1
- **Seiten:** 38
- **Fragetyp:** `semantic_paraphrase`
- **Referenzantwort:** C-Artikel besitzen zwar einen sehr niedrigen Einzelwert, werden aber häufig bestellt und verursachen dabei hohen administrativen Aufwand. Dieser Prozessaufwand kann zu hohen Kosten und Wettbewerbsnachteilen führen.
- **extraction_risk:** `false`
- **manual_review_required:** `false`

- [x] Ist die Frage eindeutig?
- [x] Ist sie anhand des Korpus beantwortbar?
- [x] Stimmen Quelle und Seiten?
- [x] Ist die Referenzantwort korrekt?
- [x] Ist der Fragetyp passend?
- [x] Werden wirklich mehrere Chunks ben?tigt?
- [x] Ist die Frage f?r den sp?teren Strategievergleich geeignet?

## final-pwl-08

- **Frage:** Wie grenzt die Vorlesung Supply Chain Management von den klassischen Logistikbereichen ab?
- **Dokument:** Produktionswirtschaft und Logistik ? Vorlesung 1
- **Seiten:** 14
- **Fragetyp:** `terminology`
- **Referenzantwort:** Die klassischen Logistikbereiche fokussieren vor allem innerbetriebliche Material- und Informationsflüsse. Supply Chain Management ergänzt diese Perspektive um die überbetriebliche Zusammenarbeit mehrerer Unternehmen zur Optimierung der gesamten Versorgungskette.
- **extraction_risk:** `false`
- **manual_review_required:** `false`

- [x] Ist die Frage eindeutig?
- [x] Ist sie anhand des Korpus beantwortbar?
- [x] Stimmen Quelle und Seiten?
- [x] Ist die Referenzantwort korrekt?
- [x] Ist der Fragetyp passend?
- [x] Werden wirklich mehrere Chunks ben?tigt?
- [x] Ist die Frage f?r den sp?teren Strategievergleich geeignet?

## final-pwl-09

- **Frage:** Nach welchem Kriterium klassifiziert die XYZ-Analyse Materialien als X-, Y- oder Z-Teile?
- **Dokument:** Produktionswirtschaft und Logistik ? Vorlesung 1
- **Seiten:** 33
- **Fragetyp:** `terminology`
- **Referenzantwort:** Die XYZ-Analyse unterscheidet nach der Vorhersagegenauigkeit des Verbrauchs oder Absatzes, die aus vergangenen Verbrauchsschwankungen abgeleitet wird. X steht für gleichmäßigen, Y für stärker schwankenden und Z für stark schwankenden beziehungsweise unregelmäßigen Verbrauch.
- **extraction_risk:** `false`
- **manual_review_required:** `false`

- [x] Ist die Frage eindeutig?
- [x] Ist sie anhand des Korpus beantwortbar?
- [x] Stimmen Quelle und Seiten?
- [x] Ist die Referenzantwort korrekt?
- [x] Ist der Fragetyp passend?
- [x] Werden wirklich mehrere Chunks ben?tigt?
- [x] Ist die Frage f?r den sp?teren Strategievergleich geeignet?

## final-pwl-10

- **Frage:** Welche gemeinsame Kernaussage leitet die Vorlesung aus den verschiedenen Logistikdefinitionen ab?
- **Dokument:** Produktionswirtschaft und Logistik ? Vorlesung 1
- **Seiten:** 7, 8, 9
- **Fragetyp:** `contextual`
- **Referenzantwort:** Über die unterschiedlichen Definitionen hinweg kümmert sich Logistik um Flüsse und Prozesse, umfasst sowohl Güter als auch Informationen und betrachtet die gesamte Kette als ganzheitliche Funktion.
- **extraction_risk:** `false`
- **manual_review_required:** `false`

- [x] Ist die Frage eindeutig?
- [x] Ist sie anhand des Korpus beantwortbar?
- [x] Stimmen Quelle und Seiten?
- [x] Ist die Referenzantwort korrekt?
- [x] Ist der Fragetyp passend?
- [x] Werden wirklich mehrere Chunks ben?tigt?
- [x] Ist die Frage f?r den sp?teren Strategievergleich geeignet?

## final-pwl-11

- **Frage:** Welche Ziele verfolgt das C-Artikel-Management, und mit welchen strategischen Ansätzen sollen sie erreicht werden?
- **Dokument:** Produktionswirtschaft und Logistik ? Vorlesung 1
- **Seiten:** 39
- **Fragetyp:** `contextual`
- **Referenzantwort:** Ziele sind unter anderem niedrigere Prozesskosten, kürzere Durchlaufzeiten, schlankere Beschaffungsprozesse, günstigere Einkaufspreise durch Bedarfsbündelung und gesicherte Artikelverfügbarkeit. Als Strategien nennt die Unterlage unter anderem dezentrale Beschaffung und Budgetverantwortung, Materialgruppenmanagement, die Bündelung auf weniger Lieferanten, Outsourcing und E-Procurement.
- **extraction_risk:** `false`
- **manual_review_required:** `false`

- [x] Ist die Frage eindeutig?
- [x] Ist sie anhand des Korpus beantwortbar?
- [x] Stimmen Quelle und Seiten?
- [x] Ist die Referenzantwort korrekt?
- [x] Ist der Fragetyp passend?
- [x] Werden wirklich mehrere Chunks ben?tigt?
- [x] Ist die Frage f?r den sp?teren Strategievergleich geeignet?

## final-pwl-12

- **Frage:** Wie unterscheiden sich Wertanalyse und Target Costing hinsichtlich Ausgangspunkt und Ziel der Kostenbeeinflussung?
- **Dokument:** Produktionswirtschaft und Logistik ? Vorlesung 1
- **Seiten:** 43, 44
- **Fragetyp:** `relational`
- **Referenzantwort:** Die Wertanalyse untersucht Funktionen, Kosten und Nutzen eines Produkts und sucht Lösungen, die benötigte Funktionen zu möglichst geringen Kosten realisieren. Target Costing beginnt dagegen beim Markt und beim Kunden, um zulässige Produktkosten und die Produktgestaltung festzulegen und so die Wettbewerbsfähigkeit zu erhöhen.
- **extraction_risk:** `false`
- **manual_review_required:** `false`

- [x] Ist die Frage eindeutig?
- [x] Ist sie anhand des Korpus beantwortbar?
- [x] Stimmen Quelle und Seiten?
- [x] Ist die Referenzantwort korrekt?
- [x] Ist der Fragetyp passend?
- [x] Werden wirklich mehrere Chunks ben?tigt?
- [x] Ist die Frage f?r den sp?teren Strategievergleich geeignet?

## final-pwl-13

- **Frage:** Wie wird die Beschaffung von C-Artikeln von der Sortimentsanalyse bis zur Abrechnung organisatorisch verschlankt?
- **Dokument:** Produktionswirtschaft und Logistik ? Vorlesung 1
- **Seiten:** 40, 41
- **Fragetyp:** `multi_hop`
- **Referenzantwort:** C-Artikel werden in einem separaten Ablauf analysiert und zu Materialgruppen und Bedarfen gebündelt. Danach werden wenige leistungsfähige Lieferanten ausgewählt, Beschaffungs-, Budget- und teilweise Prozessverantwortung dezentralisiert und Rahmenverträge geschlossen. Webbasierte E-Procurement-Lösungen oder Vollsortimenter vereinfachen Bestellungen; die Abrechnung kann gebündelt über periodische Sammelrechnungen erfolgen.
- **extraction_risk:** `false`
- **manual_review_required:** `false`

- [x] Ist die Frage eindeutig?
- [x] Ist sie anhand des Korpus beantwortbar?
- [x] Stimmen Quelle und Seiten?
- [x] Ist die Referenzantwort korrekt?
- [x] Ist der Fragetyp passend?
- [x] Werden wirklich mehrere Chunks ben?tigt?
- [x] Ist die Frage f?r den sp?teren Strategievergleich geeignet?

## final-pwl-14

- **Frage:** Welche Gewinnwirkung zeigt das Rechenbeispiel für zehn Prozent mehr Umsatz im Vergleich zu zehn Prozent geringeren Materialkosten?
- **Dokument:** Produktionswirtschaft und Logistik ? Vorlesung 1
- **Seiten:** 3
- **Fragetyp:** `table_or_figure`
- **Referenzantwort:** Im Beispiel steigt der Gewinn bei zehn Prozent mehr Umsatz von 10.000 auf 11.000 und damit um zehn Prozent. Werden die Materialkosten um zehn Prozent reduziert, steigt der Gewinn von 10.000 auf 15.000 und damit um fünfzig Prozent.
- **extraction_risk:** `false`
- **manual_review_required:** `false`

- [x] Ist die Frage eindeutig?
- [x] Ist sie anhand des Korpus beantwortbar?
- [x] Stimmen Quelle und Seiten?
- [x] Ist die Referenzantwort korrekt?
- [x] Ist der Fragetyp passend?
- [x] Werden wirklich mehrere Chunks ben?tigt?
- [x] Ist die Frage f?r den sp?teren Strategievergleich geeignet?

## final-pwl-15

- **Frage:** Welches konkrete Softwareprodukt schreibt die Vorlesung zur Durchführung einer ABC-Analyse vor?
- **Dokument:** Produktionswirtschaft und Logistik ? Vorlesung 1
- **Seiten:** keine
- **Fragetyp:** `unanswerable`
- **Referenzantwort:** Im Evaluationskorpus wird kein konkretes Softwareprodukt zur Durchführung einer ABC-Analyse vorgeschrieben.
- **extraction_risk:** `false`
- **manual_review_required:** `false`

- [x] Ist die Frage eindeutig?
- [x] Ist sie anhand des Korpus beantwortbar?
- [x] Stimmen Quelle und Seiten?
- [x] Ist die Referenzantwort korrekt?
- [x] Ist der Fragetyp passend?
- [x] Werden wirklich mehrere Chunks ben?tigt?
- [x] Ist die Frage f?r den sp?teren Strategievergleich geeignet?

## final-http-04

- **Frage:** Welche Besonderheit hat die Antwort auf eine HEAD-Anfrage im Vergleich zu GET?
- **Dokument:** Verteilte Anwendungen ? HTTP
- **Seiten:** 12
- **Fragetyp:** `fact`
- **Referenzantwort:** HEAD fordert eine Antwort an, die einer GET-Antwort entspricht, aber keinen Response Body enthält.
- **extraction_risk:** `false`
- **manual_review_required:** `false`

- [x] Ist die Frage eindeutig?
- [x] Ist sie anhand des Korpus beantwortbar?
- [x] Stimmen Quelle und Seiten?
- [x] Ist die Referenzantwort korrekt?
- [x] Ist der Fragetyp passend?
- [x] Werden wirklich mehrere Chunks ben?tigt?
- [x] Ist die Frage f?r den sp?teren Strategievergleich geeignet?

## final-http-05

- **Frage:** Welche vier Cookie-Kategorien nennt die HTTP-Unterlage?
- **Dokument:** Verteilte Anwendungen ? HTTP
- **Seiten:** 22
- **Fragetyp:** `fact`
- **Referenzantwort:** Die Unterlage nennt Session Cookies, Persistent Cookies, First-Party Cookies und Third-Party Cookies.
- **extraction_risk:** `false`
- **manual_review_required:** `false`

- [x] Ist die Frage eindeutig?
- [x] Ist sie anhand des Korpus beantwortbar?
- [x] Stimmen Quelle und Seiten?
- [x] Ist die Referenzantwort korrekt?
- [x] Ist der Fragetyp passend?
- [x] Werden wirklich mehrere Chunks ben?tigt?
- [x] Ist die Frage f?r den sp?teren Strategievergleich geeignet?

## final-http-06

- **Frage:** How can HTTP be stateless while still supporting stateful user sessions?
- **Dokument:** Verteilte Anwendungen ? HTTP
- **Seiten:** 22, 23, 24
- **Fragetyp:** `semantic_paraphrase`
- **Referenzantwort:** HTTP itself is stateless because it does not inherently link two requests on the same connection. A server can nevertheless create a session by sending a cookie, the browser storing it, and the browser returning that cookie with later requests so the server can associate them with stored state.
- **extraction_risk:** `false`
- **manual_review_required:** `false`

- [x] Ist die Frage eindeutig?
- [x] Ist sie anhand des Korpus beantwortbar?
- [x] Stimmen Quelle und Seiten?
- [x] Ist die Referenzantwort korrekt?
- [x] Ist der Fragetyp passend?
- [x] Werden wirklich mehrere Chunks ben?tigt?
- [x] Ist die Frage f?r den sp?teren Strategievergleich geeignet?

## final-http-07

- **Frage:** Wie unterscheiden sich URN und URL als Formen eines URI?
- **Dokument:** Verteilte Anwendungen ? HTTP
- **Seiten:** 13
- **Fragetyp:** `semantic_paraphrase`
- **Referenzantwort:** Ein URN benennt eine Ressource über einen Namensraum und einen namensraumspezifischen Teil, etwa eine ISBN. Ein URL lokalisiert beziehungsweise adressiert eine Ressource über ein Schema und einen schemaspezifischen Teil, etwa mit http, mailto oder file. Beide werden als URI-Formen dargestellt.
- **extraction_risk:** `false`
- **manual_review_required:** `false`

- [x] Ist die Frage eindeutig?
- [x] Ist sie anhand des Korpus beantwortbar?
- [x] Stimmen Quelle und Seiten?
- [x] Ist die Referenzantwort korrekt?
- [x] Ist der Fragetyp passend?
- [x] Werden wirklich mehrere Chunks ben?tigt?
- [x] Ist die Frage f?r den sp?teren Strategievergleich geeignet?

## final-http-08

- **Frage:** Wie ist ein MIME-Typ aufgebaut, und welches Beispiel zeigt einen zusätzlichen Parameter?
- **Dokument:** Verteilte Anwendungen ? HTTP
- **Seiten:** 19
- **Fragetyp:** `terminology`
- **Referenzantwort:** Ein MIME-Typ folgt der Struktur Typ/Untertyp und kann durch einen Parameter mit Wert ergänzt werden. Als Beispiel nennt die Unterlage text/plain;charset=UTF-8.
- **extraction_risk:** `false`
- **manual_review_required:** `false`

- [x] Ist die Frage eindeutig?
- [x] Ist sie anhand des Korpus beantwortbar?
- [x] Stimmen Quelle und Seiten?
- [x] Ist die Referenzantwort korrekt?
- [x] Ist der Fragetyp passend?
- [x] Werden wirklich mehrere Chunks ben?tigt?
- [x] Ist die Frage f?r den sp?teren Strategievergleich geeignet?

## final-http-09

- **Frage:** Wie wählt der Server bei servergesteuerter Content Negotiation eine passende Repräsentation aus?
- **Dokument:** Verteilte Anwendungen ? HTTP
- **Seiten:** 17, 18
- **Fragetyp:** `contextual`
- **Referenzantwort:** Der Client teilt im Request über Accept, Accept-Language und Accept-Encoding seine Präferenzen für Medientyp, Sprache und Kodierung mit. Der Server wählt daraus eine verfügbare Repräsentation und beschreibt sie in der Antwort unter anderem mit Content-Type, Content-Language und Content-Encoding.
- **extraction_risk:** `false`
- **manual_review_required:** `false`

- [x] Ist die Frage eindeutig?
- [x] Ist sie anhand des Korpus beantwortbar?
- [x] Stimmen Quelle und Seiten?
- [x] Ist die Referenzantwort korrekt?
- [x] Ist der Fragetyp passend?
- [x] Werden wirklich mehrere Chunks ben?tigt?
- [x] Ist die Frage f?r den sp?teren Strategievergleich geeignet?

## final-http-10

- **Frage:** Welche Beziehung besteht zwischen Set-Cookie im Login-Response und Cookie in späteren Requests?
- **Dokument:** Verteilte Anwendungen ? HTTP
- **Seiten:** 22, 23
- **Fragetyp:** `relational`
- **Referenzantwort:** Der Server setzt im Login-Response mit Set-Cookie eine Sitzungskennung. Der Browser speichert sie lokal und sendet sie in späteren Requests im Cookie-Header zurück. Der Server liest die Kennung und kann dadurch mehrere ansonsten unabhängige HTTP-Anfragen einer Sitzung zuordnen.
- **extraction_risk:** `false`
- **manual_review_required:** `false`

- [x] Ist die Frage eindeutig?
- [x] Ist sie anhand des Korpus beantwortbar?
- [x] Stimmen Quelle und Seiten?
- [x] Ist die Referenzantwort korrekt?
- [x] Ist der Fragetyp passend?
- [x] Werden wirklich mehrere Chunks ben?tigt?
- [x] Ist die Frage f?r den sp?teren Strategievergleich geeignet?

## final-http-11

- **Frage:** Wie wird aus der URL `http://server.example.org:8080/index.html?param1=A&param2=B` ein HTTP-Request-Pfad, und welche Zeichen trennen die Query-Parameter?
- **Dokument:** Verteilte Anwendungen ? HTTP
- **Seiten:** 14, 15
- **Fragetyp:** `multi_hop`
- **Referenzantwort:** Im Request steht als Pfad `/index.html?param1=A&param2=B`, während Host und Port als `server.example.org:8080` im Host-Header erscheinen. Das Fragezeichen beginnt den Query String, das Gleichheitszeichen verbindet Parameternamen und Wert, und das kaufmännische Und trennt mehrere Parameter.
- **extraction_risk:** `false`
- **manual_review_required:** `false`

- [x] Ist die Frage eindeutig?
- [x] Ist sie anhand des Korpus beantwortbar?
- [x] Stimmen Quelle und Seiten?
- [x] Ist die Referenzantwort korrekt?
- [x] Ist der Fragetyp passend?
- [x] Werden wirklich mehrere Chunks ben?tigt?
- [x] Ist die Frage f?r den sp?teren Strategievergleich geeignet?

## final-http-12

- **Frage:** Welchen TCP/IP-Schichten ordnet die Übersicht HTTP, TLS, TCP und IP zu?
- **Dokument:** Verteilte Anwendungen ? HTTP
- **Seiten:** 6
- **Fragetyp:** `table_or_figure`
- **Referenzantwort:** HTTP und HTTPS sind der Application-Schicht zugeordnet. TLS erscheint bei den OSI-Darstellungs- und Sitzungsschichten innerhalb des Application-Bereichs, TCP gehört zur Transport-Schicht und IP zur Internet-Schicht.
- **extraction_risk:** `false`
- **manual_review_required:** `false`

- [x] Ist die Frage eindeutig?
- [x] Ist sie anhand des Korpus beantwortbar?
- [x] Stimmen Quelle und Seiten?
- [x] Ist die Referenzantwort korrekt?
- [x] Ist der Fragetyp passend?
- [x] Werden wirklich mehrere Chunks ben?tigt?
- [x] Ist die Frage f?r den sp?teren Strategievergleich geeignet?

## final-se-03

- **Frage:** Welche vier wesentlichen Aktivitäten gehören zur Anforderungsdefinition?
- **Dokument:** ?bersicht zum Prozess der Softwareentwicklung
- **Seiten:** 3
- **Fragetyp:** `fact`
- **Referenzantwort:** Zur Anforderungsdefinition gehören das Ermitteln, Dokumentieren, Validieren und Abstimmen sowie das Verwalten von Anforderungen.
- **extraction_risk:** `false`
- **manual_review_required:** `false`

- [x] Ist die Frage eindeutig?
- [x] Ist sie anhand des Korpus beantwortbar?
- [x] Stimmen Quelle und Seiten?
- [x] Ist die Referenzantwort korrekt?
- [x] Ist der Fragetyp passend?
- [x] Werden wirklich mehrere Chunks ben?tigt?
- [x] Ist die Frage f?r den sp?teren Strategievergleich geeignet?

## final-se-04

- **Frage:** Wie unterscheiden sich Kundenanforderungen, Anwendungsfälle und spezifizierte Produktanforderungen als Ergebnisse der Anforderungsdefinition?
- **Dokument:** ?bersicht zum Prozess der Softwareentwicklung
- **Seiten:** 4
- **Fragetyp:** `semantic_paraphrase`
- **Referenzantwort:** Kundenanforderungen halten Wünsche etwa in User Stories oder Interviewprotokollen fest. Anwendungsfälle und Akteure beschreiben die Leistungen des Systems aus externer Sicht. Spezifizierte funktionale Produktanforderungen beschreiben diese Leistungen detaillierter, unter anderem durch Anwenderszenarien.
- **extraction_risk:** `false`
- **manual_review_required:** `false`

- [x] Ist die Frage eindeutig?
- [x] Ist sie anhand des Korpus beantwortbar?
- [x] Stimmen Quelle und Seiten?
- [x] Ist die Referenzantwort korrekt?
- [x] Ist der Fragetyp passend?
- [x] Werden wirklich mehrere Chunks ben?tigt?
- [x] Ist die Frage f?r den sp?teren Strategievergleich geeignet?

## final-se-05

- **Frage:** How do component, integration, system, and acceptance tests differ in scope and environment?
- **Dokument:** ?bersicht zum Prozess der Softwareentwicklung
- **Seiten:** 10, 11
- **Fragetyp:** `semantic_paraphrase`
- **Referenzantwort:** A component test examines individual classes or components separately. An integration test checks the interaction and interfaces of multiple components. A system test verifies the complete system against the requirements specification and includes non-functional tests. An acceptance test is performed in the target environment under real user conditions and focuses on usability and external quality before handover.
- **extraction_risk:** `false`
- **manual_review_required:** `false`

- [x] Ist die Frage eindeutig?
- [x] Ist sie anhand des Korpus beantwortbar?
- [x] Stimmen Quelle und Seiten?
- [x] Ist die Referenzantwort korrekt?
- [x] Ist der Fragetyp passend?
- [x] Werden wirklich mehrere Chunks ben?tigt?
- [x] Ist die Frage f?r den sp?teren Strategievergleich geeignet?

## final-se-06

- **Frage:** Was versteht die Unterlage unter einer Software-Architektur?
- **Dokument:** ?bersicht zum Prozess der Softwareentwicklung
- **Seiten:** 6
- **Fragetyp:** `terminology`
- **Referenzantwort:** Die Software-Architektur ist die strukturierte und hierarchische Anordnung der Systemkomponenten und ihrer Beziehungen untereinander.
- **extraction_risk:** `false`
- **manual_review_required:** `false`

- [x] Ist die Frage eindeutig?
- [x] Ist sie anhand des Korpus beantwortbar?
- [x] Stimmen Quelle und Seiten?
- [x] Ist die Referenzantwort korrekt?
- [x] Ist der Fragetyp passend?
- [x] Werden wirklich mehrere Chunks ben?tigt?
- [x] Ist die Frage f?r den sp?teren Strategievergleich geeignet?

## final-se-07

- **Frage:** Welche nicht-funktionalen Qualitätsziele soll die Software-Architektur berücksichtigen?
- **Dokument:** ?bersicht zum Prozess der Softwareentwicklung
- **Seiten:** 7, 8
- **Fragetyp:** `contextual`
- **Referenzantwort:** Die Architektur soll nicht-funktionale Anforderungen wie Zuverlässigkeit und Fehlertoleranz, Leistung und Effizienz einschließlich Antwortzeiten und Ressourcenbedarf sowie Sicherheit mit Datenintegrität, Verfügbarkeit und Vertraulichkeit berücksichtigen. Auch Bedienarten und Technologievorgaben werden genannt.
- **extraction_risk:** `false`
- **manual_review_required:** `false`

- [x] Ist die Frage eindeutig?
- [x] Ist sie anhand des Korpus beantwortbar?
- [x] Stimmen Quelle und Seiten?
- [x] Ist die Referenzantwort korrekt?
- [x] Ist der Fragetyp passend?
- [x] Werden wirklich mehrere Chunks ben?tigt?
- [x] Ist die Frage f?r den sp?teren Strategievergleich geeignet?

## final-se-08

- **Frage:** Wie verbindet die Software-Architektur Anforderungen mit der späteren Implementierung?
- **Dokument:** ?bersicht zum Prozess der Softwareentwicklung
- **Seiten:** 5, 7, 8, 9
- **Fragetyp:** `relational`
- **Referenzantwort:** Im Entwurf wird festgelegt, wie funktionale und nicht-funktionale Anforderungen durch Komponenten, Schnittstellen, technische Infrastruktur und Design umgesetzt werden. Die daraus entstehende Architektur dient als fachlicher und technischer Leitfaden. In der Implementierung werden diese Entwurfsergebnisse anschließend programmtechnisch in Konstrukte der verwendeten Programmiersprache übertragen.
- **extraction_risk:** `false`
- **manual_review_required:** `false`

- [x] Ist die Frage eindeutig?
- [x] Ist sie anhand des Korpus beantwortbar?
- [x] Stimmen Quelle und Seiten?
- [x] Ist die Referenzantwort korrekt?
- [x] Ist der Fragetyp passend?
- [x] Werden wirklich mehrere Chunks ben?tigt?
- [x] Ist die Frage f?r den sp?teren Strategievergleich geeignet?

## final-se-09

- **Frage:** Wie reicht die Qualitätssicherung von der Anforderungsermittlung über die Implementierung bis zum Systemtest?
- **Dokument:** ?bersicht zum Prozess der Softwareentwicklung
- **Seiten:** 3, 9, 10
- **Fragetyp:** `multi_hop`
- **Referenzantwort:** Anforderungen werden zunächst anhand ihrer Dokumentation, Spezifikation und Qualitätskriterien validiert. Während der Implementierung werden Test und Verifikation einschließlich Testplanung und Testfallerstellung durchgeführt. Beim späteren Systemtest wird das vollständige System gegen die Anforderungsspezifikation geprüft; die dafür verwendeten Testfälle entstehen bereits während der Anforderungsermittlung.
- **extraction_risk:** `false`
- **manual_review_required:** `false`

- [x] Ist die Frage eindeutig?
- [x] Ist sie anhand des Korpus beantwortbar?
- [x] Stimmen Quelle und Seiten?
- [x] Ist die Referenzantwort korrekt?
- [x] Ist der Fragetyp passend?
- [x] Werden wirklich mehrere Chunks ben?tigt?
- [x] Ist die Frage f?r den sp?teren Strategievergleich geeignet?

## final-selfrag-05

- **Frage:** What hardware and principal optimization settings were used to train the reported SELF-RAG models?
- **Dokument:** SELF-RAG
- **Seiten:** 19
- **Fragetyp:** `fact`
- **Referenzantwort:** The models were trained on four Nvidia A100 GPUs with 80 GB memory for three epochs, using a batch size of 128, a peak learning rate of 2e-5, three percent warmup, and linear decay. The setup used DeepSpeed stage 3, Bfloat16 precision, and FlashAttention.
- **extraction_risk:** `false`
- **manual_review_required:** `false`

- [x] Ist die Frage eindeutig?
- [x] Ist sie anhand des Korpus beantwortbar?
- [x] Stimmen Quelle und Seiten?
- [x] Ist die Referenzantwort korrekt?
- [x] Ist der Fragetyp passend?
- [x] Werden wirklich mehrere Chunks ben?tigt?
- [x] Ist die Frage f?r den sp?teren Strategievergleich geeignet?

## final-selfrag-06

- **Frage:** What does the paper's memory analysis suggest about SELF-RAG's reliance on evidence compared with instruction-tuned baselines?
- **Dokument:** SELF-RAG
- **Seiten:** 20
- **Fragetyp:** `semantic_paraphrase`
- **Referenzantwort:** Among correctly answered sampled questions, SELF-RAG produced answers absent from the retrieved evidence much less often than the instruction-tuned baselines: the paper reports two percent for SELF-RAG versus higher shares for the compared Alpaca and Llama2-chat models. When passages are irrelevant, SELF-RAG can also emit ISREL=Irrelevant instead of continuing with an apparently grounded answer.
- **extraction_risk:** `false`
- **manual_review_required:** `false`

- [x] Ist die Frage eindeutig?
- [x] Ist sie anhand des Korpus beantwortbar?
- [x] Stimmen Quelle und Seiten?
- [x] Ist die Referenzantwort korrekt?
- [x] Ist der Fragetyp passend?
- [x] Werden wirklich mehrere Chunks ben?tigt?
- [x] Ist die Frage f?r den sp?teren Strategievergleich geeignet?

## final-selfrag-07

- **Frage:** How is a SELF-RAG continuation scored during critique-guided tree decoding?
- **Dokument:** SELF-RAG
- **Seiten:** 6, 19
- **Fragetyp:** `terminology`
- **Referenzantwort:** A continuation combines its generator probability with a critique score. That critique score is a weighted sum of normalized probabilities derived from the desirable ISREL, ISSUP, and ISUSE outcomes. The weights are inference-time hyperparameters, so the decoder can emphasize aspects such as evidential support; the appendix specifies the normalization and utility weights.
- **extraction_risk:** `false`
- **manual_review_required:** `false`

- [x] Ist die Frage eindeutig?
- [x] Ist sie anhand des Korpus beantwortbar?
- [x] Stimmen Quelle und Seiten?
- [x] Ist die Referenzantwort korrekt?
- [x] Ist der Fragetyp passend?
- [x] Werden wirklich mehrere Chunks ben?tigt?
- [x] Ist die Frage f?r den sp?teren Strategievergleich geeignet?

## final-selfrag-08

- **Frage:** What ethical limitation remains even after SELF-RAG improves factuality and citation accuracy?
- **Dokument:** SELF-RAG
- **Seiten:** 10
- **Fragetyp:** `contextual`
- **Referenzantwort:** SELF-RAG can still generate outputs that are not fully supported by their citations. The authors therefore present explicit self-reflection and fine-grained attribution as aids for users to verify possible factual errors, not as a guarantee that errors are eliminated.
- **extraction_risk:** `false`
- **manual_review_required:** `false`

- [x] Ist die Frage eindeutig?
- [x] Ist sie anhand des Korpus beantwortbar?
- [x] Stimmen Quelle und Seiten?
- [x] Ist die Referenzantwort korrekt?
- [x] Ist der Fragetyp passend?
- [x] Werden wirklich mehrere Chunks ben?tigt?
- [x] Ist die Frage f?r den sp?teren Strategievergleich geeignet?

## final-selfrag-09

- **Frage:** Which two categories of retrieval baselines does the paper compare against, and how do they differ?
- **Dokument:** SELF-RAG
- **Seiten:** 7
- **Fragetyp:** `contextual`
- **Referenzantwort:** One category augments an existing language model with retrieved documents at test time, as in the standard RAG and retrieval-augmented chat baselines. The other category consists of methods trained with retrieved text or retrieval-related calls, including SAIL and Toolformer.
- **extraction_risk:** `false`
- **manual_review_required:** `false`

- [x] Ist die Frage eindeutig?
- [x] Ist sie anhand des Korpus beantwortbar?
- [x] Stimmen Quelle und Seiten?
- [x] Ist die Referenzantwort korrekt?
- [x] Ist der Fragetyp passend?
- [x] Werden wirklich mehrere Chunks ben?tigt?
- [x] Ist die Frage f?r den sp?teren Strategievergleich geeignet?

## final-selfrag-10

- **Frage:** Why are ISREL and ISSUP complementary rather than interchangeable judgments?
- **Dokument:** SELF-RAG
- **Seiten:** 3, 16
- **Fragetyp:** `relational`
- **Referenzantwort:** ISREL asks whether a retrieved passage is useful for answering the input. ISSUP instead asks whether the verification-worthy claims in a particular generated segment are entailed by that evidence, with fully, partially, or no support outcomes. A passage may therefore be relevant to the topic without fully supporting every generated claim.
- **extraction_risk:** `false`
- **manual_review_required:** `false`

- [x] Ist die Frage eindeutig?
- [x] Ist sie anhand des Korpus beantwortbar?
- [x] Stimmen Quelle und Seiten?
- [x] Ist die Referenzantwort korrekt?
- [x] Ist der Fragetyp passend?
- [x] Werden wirklich mehrere Chunks ben?tigt?
- [x] Ist die Frage f?r den sp?teren Strategievergleich geeignet?

## final-selfrag-11

- **Frage:** Why does SELF-RAG train the generator to predict reflection tokens while masking retrieved passage text from the loss?
- **Dokument:** SELF-RAG
- **Seiten:** 4, 5
- **Fragetyp:** `relational`
- **Referenzantwort:** The generator must learn to produce both task outputs and reflection decisions itself so that the critic is not required at inference time. Retrieved passages are conditioning evidence rather than target text, so their tokens are masked from the next-token loss while the output and reflection tokens remain training targets.
- **extraction_risk:** `false`
- **manual_review_required:** `false`

- [x] Ist die Frage eindeutig?
- [x] Ist sie anhand des Korpus beantwortbar?
- [x] Stimmen Quelle und Seiten?
- [x] Ist die Referenzantwort korrekt?
- [x] Ist der Fragetyp passend?
- [x] Werden wirklich mehrere Chunks ben?tigt?
- [x] Ist die Frage f?r den sp?teren Strategievergleich geeignet?

## final-selfrag-12

- **Frage:** How do the task-specific metrics and the ablation variants together test whether SELF-RAG's gains come from retrieval and self-reflection?
- **Dokument:** SELF-RAG
- **Seiten:** 6, 8, 9
- **Fragetyp:** `multi_hop`
- **Referenzantwort:** The evaluation uses task-appropriate outcomes: accuracy for closed-set and short-form tasks, and measures such as FactScore, exact match, ROUGE, MAUVE, and citation precision and recall for long-form generation. The ablations then remove training-time retrieval, critic supervision, inference-time retrieval, adaptive triggering, or the ISSUP score. Comparing those variants with full SELF-RAG tests whether improvements persist when retrieval or self-reflection components are absent.
- **extraction_risk:** `false`
- **manual_review_required:** `false`

- [x] Ist die Frage eindeutig?
- [x] Ist sie anhand des Korpus beantwortbar?
- [x] Stimmen Quelle und Seiten?
- [x] Ist die Referenzantwort korrekt?
- [x] Ist der Fragetyp passend?
- [x] Werden wirklich mehrere Chunks ben?tigt?
- [x] Ist die Frage f?r den sp?teren Strategievergleich geeignet?
