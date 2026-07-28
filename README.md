# LearnCycle

LearnCycle ist eine webbasierte Lernplattform, die klassische Materialverwaltung mit KI-gestützten Lernmethoden verbindet. Studierende können ihre Unterlagen nach Kursen und Ordnern organisieren, PDF-Dokumente bearbeiten und anschließend gezielt mit den eigenen Materialien lernen.

Die Anwendung entstand im Rahmen einer Bachelorarbeit. Im Mittelpunkt steht die Frage, wie ein KI-gestütztes Lernsystem entwickelt werden kann, das Active Learning fördert. Retrieval-Augmented Generation (RAG) dient dabei als technischer Ansatz, um die Lernunterstützung nachvollziehbar auf den bereitgestellten Materialien aufzubauen.

## Funktionen

- Kurse, Ordner und PDF-Lernmaterialien verwalten
- PDF-Dokumente lesen, Textstellen markieren und kommentieren
- strukturierte Notizen mit automatischer Speicherung erstellen
- Fragen an die eigenen Materialien stellen und Quellen nachvollziehen
- Themen im Modus „Guided Learning“ schrittweise erarbeiten
- eigene Erklärungen mit der Feynman-Technik überprüfen
- Lernkarten und Multiple-Choice-Mocktests generieren
- Lerninhalte als hierarchische Mindmap darstellen
- Lernkarten mit dem SM-2-Verfahren zeitlich wiederholen

KI-generierte Antworten werden aus den ausgewählten PDFs, Notizen und Annotationen abgeleitet. Supabase bleibt dabei die maßgebliche Datenquelle; Qdrant dient ausschließlich als Retrieval-Index.

## Architektur

LearnCycle besteht aus drei zentralen Bereichen:

```text
Browser
  │
  ▼
Next.js-Anwendung ── Supabase (Authentifizierung, Datenbank, PDF-Speicher)
  │
  ▼
Python RAG API / Worker ── Qdrant (hybride Suche)
                       └── Neo4j (optionale Lern- und Wissensgraphen)
```

Die Webanwendung übernimmt Benutzeroberfläche, Authentifizierung und serverseitige Zugriffsprüfung. Der Python-Dienst verarbeitet Dokumente, erzeugt Embeddings und beantwortet Lernfragen. Die Kommunikation mit dem RAG-Dienst erfolgt serverseitig über einen gemeinsamen internen API-Schlüssel.

Verwendete Technologien:

- Next.js 16, React 19 und TypeScript
- Tailwind CSS, TipTap und React PDF Viewer
- Supabase für Authentifizierung, PostgreSQL und Dateispeicher
- Python, FastAPI und Docling für die RAG-Pipeline
- OpenAI Embeddings und Sprachmodelle
- Qdrant für Dense-, Sparse- und Hybrid-Retrieval
- optional Neo4j für GraphRAG und Lernstrukturen

## Lokale Einrichtung

Vorausgesetzt werden Node.js, Python 3.12, Docker sowie ein konfiguriertes Supabase-Projekt.

1. Repository klonen und Abhängigkeiten installieren:

   ```bash
   npm install
   python -m pip install -r rag_pipeline/requirements.txt
   ```

2. Umgebungsdateien anlegen:

   ```bash
   cp .env.example .env.local
   cp rag_pipeline/.env.example rag_pipeline/.env
   ```

   Unter PowerShell kann stattdessen `Copy-Item` verwendet werden. Mindestens erforderlich sind die Supabase-Zugangsdaten, `OPENAI_API_KEY`, `RAG_INTERNAL_API_KEY` und die Qdrant-Konfiguration. Der interne API-Schlüssel muss in beiden Umgebungsdateien identisch sein.

3. Infrastruktur und RAG-Dienste starten:

   ```bash
   docker compose up -d qdrant neo4j rag-api rag-worker
   ```

4. Webanwendung starten:

   ```bash
   npm run dev
   ```

Die Anwendung ist anschließend unter [http://localhost:3000](http://localhost:3000) erreichbar. Die RAG API läuft standardmäßig auf Port `8001`.

Alternativ können API und Worker ohne Docker gestartet werden:

```bash
uvicorn rag_pipeline.api:app --host 0.0.0.0 --port 8001
python -m rag_pipeline.worker --loop
```

Die SQL-Dateien unter `supabase/migrations/` müssen vor der Nutzung in der richtigen Reihenfolge auf das Supabase-Projekt angewendet werden. Zugangsdaten und lokale `.env`-Dateien dürfen nicht in das Repository eingecheckt werden.

## Qualitätssicherung

```bash
npx tsc --noEmit
pytest rag_pipeline/tests
npm run lint
```

Die Python-Tests prüfen unter anderem Retrieval, Reranking, Dokumentverarbeitung, Lernmodi, Revision, Sicherheitsgrenzen und den RAG-Worker.

## Forschungsartefakte

Die für die Bachelorarbeit aufbereiteten Evaluationsdaten befinden sich unter [`research/`](./research/README.md). Dort liegen der eingefrorene Ground-Truth-Datensatz, das Korpusmanifest, Evaluationsskripte, Konfigurationen, Rohresultate, zusammengefasste Metriken und die dokumentierten Prompts.

Unterlagen zur Befragung und zum Betatest werden ausschließlich anonymisiert in den dafür vorgesehenen Verzeichnissen ergänzt.

## Weitere Dokumentation

- [`PROJECT.md`](./PROJECT.md): ausführlicher Implementierungsstand und Architekturentscheidungen
- [`rag_pipeline/README.md`](./rag_pipeline/README.md): technische Beschreibung der RAG-Pipeline
- [`research/README.md`](./research/README.md): Forschungs- und Evaluationsartefakte
- [`conductor/tracks.md`](./conductor/tracks.md): Entwicklungs- und Implementierungstracks

## Hinweis

Dieses Repository dokumentiert einen Forschungsprototyp. KI-generierte Inhalte können trotz Quellenbindung fehlerhaft sein und sollten bei fachlich wichtigen Entscheidungen anhand der angegebenen Originalquellen überprüft werden.
