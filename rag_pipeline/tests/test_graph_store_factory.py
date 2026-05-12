from __future__ import annotations

from types import SimpleNamespace

import pytest

from rag_pipeline.graph_store_factory import create_graph_store


class FakeStoreDriver:
    def session(self, database=None):
        class Session:
            def __enter__(self):
                return self

            def __exit__(self, *_):
                return None

            def run(self, *_):
                return []

        return Session()


def _config(**overrides):
    data = {
        "graph_enabled": False,
        "graph_extraction_enabled": False,
        "graph_retrieval_enabled": False,
        "graph_store_provider": "neo4j",
        "neo4j_uri": "bolt://localhost:7687",
        "neo4j_user": "neo4j",
        "neo4j_password": "password",
        "neo4j_database": "neo4j",
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def test_create_graph_store_returns_none_when_graph_disabled() -> None:
    assert create_graph_store(_config(), driver=FakeStoreDriver()) is None


def test_create_graph_store_returns_neo4j_store() -> None:
    store = create_graph_store(_config(graph_enabled=True), driver=FakeStoreDriver())

    assert store is not None


def test_create_graph_store_requires_credentials() -> None:
    with pytest.raises(RuntimeError, match="NEO4J_URI"):
        create_graph_store(_config(graph_enabled=True, neo4j_uri=None), driver=FakeStoreDriver())


def test_create_graph_store_calls_ensure_constraints() -> None:
    class Driver(FakeStoreDriver):
        def __init__(self):
            self.calls = []

        def session(self, database=None):
            driver = self

            class Session:
                def __enter__(self):
                    return self

                def __exit__(self, *_):
                    return None

                def run(self, statement, parameters=None):
                    driver.calls.append(statement)
                    return []

            return Session()

    driver = Driver()
    create_graph_store(_config(graph_retrieval_enabled=True), driver=driver)

    assert any("CREATE CONSTRAINT" in statement for statement in driver.calls)
