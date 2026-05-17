# RAG Service Deployment

Run the Python RAG API as a persistent process, for example on Fly.io, Railway,
or a dedicated Docker container.

Do not deploy it as a per-request serverless function. The latency work in this
track depends on warm OpenAI, Qdrant, FastEmbed, and Neo4j clients; cold starts
would negate those gains.
