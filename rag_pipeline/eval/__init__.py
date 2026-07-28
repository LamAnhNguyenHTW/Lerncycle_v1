"""Offline retrieval/chunking evaluation harness.

This package is evaluation tooling only. Nothing here is imported by the
production worker, RAG API, or Next.js app. It builds separate Qdrant
collections, computes retrieval metrics, and writes reproducible JSON reports.
"""
