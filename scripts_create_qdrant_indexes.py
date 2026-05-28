import os
from qdrant_client import QdrantClient, models

url = os.environ["QDRANT_URL"]
api_key = os.environ.get("QDRANT_API_KEY")
collection = os.environ.get("QDRANT_COLLECTION", "learncycle_chunks")

client = QdrantClient(url=url, api_key=api_key)

for field in ["user_id", "source_type", "pdf_id", "source_id"]:
    try:
        client.create_payload_index(
            collection_name=collection,
            field_name=field,
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        print(f"created index: {field}")
    except Exception as exc:
        print(f"skip {field}: {exc}")

print("done")
