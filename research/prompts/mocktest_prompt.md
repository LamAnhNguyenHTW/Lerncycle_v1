# Mock-test generation prompt

Canonical source: `rag_pipeline/revision/generator.py`
(`_MOCKTEST_SYSTEM_PROMPT_DE`, `_MOCKTEST_SYSTEM_PROMPT_EN`, and
`_build_user_prompt`).

German production instruction:

> Du erstellst Multiple-Choice-Fragen mit genau vier Optionen und einer
> korrekten Antwort. Gib NUR gültiges JSON mit
> `{"questions":[{"prompt":str,"choices":[str,str,str,str],"correct_index":int,"explanation":str,"source_chunk_ids":[str,...]}]}`
> zurück. Keine Erklärung außerhalb des JSON, kein Markdown und kein
> Code-Fence. Jede Frage muss aus dem bereitgestellten Material kommen und die
> korrekte Antwort muss eindeutig sein.

The English variant has the same constraints. The runtime user prompt requests
the chosen number of questions and includes bounded, source-scoped chunks.

