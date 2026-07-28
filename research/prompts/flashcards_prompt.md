# Flashcards generation prompt

Canonical source: `rag_pipeline/revision/generator.py`
(`_FLASHCARD_SYSTEM_PROMPT_DE`, `_FLASHCARD_SYSTEM_PROMPT_EN`, and
`_build_user_prompt`).

German production instruction:

> Du erstellst Lernkarten für eine Lernanwendung. Gib NUR gültiges JSON mit
> `{"cards":[{"front":str,"back":str,"source_chunk_ids":[str,...]}]}` zurück.
> Keine Erklärungen, kein Markdown und kein Code-Fence. Jede Karte muss aus dem
> bereitgestellten Material kommen. Die Vorderseite ist eine kurze Frage oder
> ein Begriff, die Rückseite eine präzise Antwort in 1–3 Sätzen.
> `source_chunk_ids` verweist ausschließlich auf Chunk-IDs aus dem Kontext.

The English variant has the same constraints. At runtime the user prompt asks
for the selected number of cards and supplies bounded chunk snippets with
`chunk_id`, title, source type, and text.

