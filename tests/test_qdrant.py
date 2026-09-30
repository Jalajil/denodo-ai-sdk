"""Exercise Qdrant through the SDK using a real, isolated in-memory store."""

import json

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from utils.uniformVectorStore import UniformVectorStore


class ThemeEmbeddings(Embeddings):
    def embed_query(self, text):
        return [float("customers" in text), float("orders" in text), 0.1]

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]


class FlatEmbeddings(Embeddings):
    def embed_query(self, text):
        return [1.0, 0.0, 0.0]

    def embed_documents(self, texts):
        return [[1.0, 0.01 * i, 0.0] for i, _ in enumerate(texts)]


@pytest.fixture
def make_store(monkeypatch):
    monkeypatch.setenv("QDRANT_LOCATION", ":memory:")
    monkeypatch.setenv("QDRANT_HYBRID", "0")
    stores = []

    def create(documents=(), embeddings=None, name="test_catalog"):
        store = UniformVectorStore("qdrant", embeddings or ThemeEmbeddings(), name)
        stores.append(store)
        if documents:
            store.client.add_documents(documents, ids=[doc.id for doc in documents])
        return store

    yield create
    for store in stores:
        store.client.client.close()


@pytest.fixture
def sparse_model(monkeypatch):
    from fastembed.sparse.bm25 import Bm25
    from utils import qdrant_store

    # Exercise the real encoder without downloading model files during tests.
    try:
        model = Bm25("Qdrant/bm25", language="english", local_files_only=True)
    except (FileNotFoundError, ValueError) as error:
        pytest.skip(f"BM25 model is not cached locally: {error}")
    monkeypatch.setattr(qdrant_store, "_SPARSE_MODEL", model)
    return model


def document(sdk_id, name, database="fin", content="customers", **metadata):
    return Document(id=sdk_id, page_content=content, metadata={
        "view_id": sdk_id.split("_")[0], "document_id": sdk_id,
        "view_name": name, "database_name": database, **metadata,
    })


def test_chunk_ids_and_json_content_survive_storage(make_store):
    payload = '{"sample_rows":[["Jeddah, Saudi Arabia",null],["",0]]}'
    store = make_store([document("42_0", "fin.customers", content=payload)])

    found = store.get_views(["42"])

    assert len(found) == 1
    assert found[0].id == "42_0"
    assert found[0].metadata["document_id"] == "42_0"
    assert json.loads(found[0].page_content) == {
        "sample_rows": [["Jeddah, Saudi Arabia", None], ["", 0]]
    }


def test_repeated_sdk_id_overwrites_then_can_be_deleted(make_store):
    store = make_store([document("42_0", "fin.customers"), document("7", "fin.orders")])
    store.client.add_documents([document("42_0", "fin.customers", content="updated")], ids=["42_0"])

    found = store.search_by_vector([1.0, 0.0, 0.1], k=10, view_ids=["42"])
    assert len(found) == 1
    assert found[0].page_content == "updated"

    store.delete(["42_0"])
    assert store.check_existence(["42"]) is False
    assert store.check_existence(["7"]) is True


def test_metadata_filters_cannot_relax_view_permissions(make_store):
    store = make_store([
        document("1", "fin.customers", tag_bank="1"),
        document("2", "ops.orders", database="ops", content="orders"),
        document("3", "ops.customers", database="ops", tag_bank="1"),
    ])

    found = store.search("customers", k=10, view_ids=["1", "2"],
                         database_names=["ops"], tag_names=["bank"])
    assert {doc.id for doc in found} == {"1", "2"}
    assert store.search("customers", view_ids=[]) == []
    assert store.search_by_vector([1.0, 0.0, 0.1], view_ids=["missing"]) == []
    assert store.get_view_ids(["ops.orders"]) == ["2"]


def test_sync_metadata_round_trips_and_overwrites(make_store):
    store = make_store()
    store.update_last_update({"DATABASE": {"fin": 100}}, {"partial_tags_by_db": {"fin": ["bank"]}})
    assert store.get_sync_metadata() == (
        {"DATABASE": {"fin": 100}}, {"partial_tags_by_db": {"fin": ["bank"]}}
    )

    store.update_last_update({"DATABASE": {"fin": 200}}, {})
    assert store.get_last_update("DATABASE", "fin") == 200
    assert len(store.search_by_vector([1.0, 0.0, 0.1], k=10, view_ids=["last_update"])) == 1


def test_batched_search_selects_highest_similarity(make_store):
    store = make_store([
        document("1", "fin.orders", content="orders"),
        document("30001", "fin.customers", content="customers"),
    ])
    permissions = [str(i) for i in range(1, 30002)]

    found = store.search_batched(k=1, view_ids=permissions, vector=[1.0, 0.0, 0.1], scores=True)

    assert found[0][0].id == "30001"
    assert found[0][1] > 0.99


def test_hybrid_search_matches_specific_term_and_keeps_permissions(make_store, monkeypatch, sparse_model):
    monkeypatch.setenv("QDRANT_HYBRID", "1")
    monkeypatch.setenv("QDRANT_SPARSE_LANGUAGE", "english")
    rows = [
        document("1", "fin.orders", content="orders purchases suppliers"),
        document("2", "fin.payroll", content="payroll monthly salaries"),
        document("3", "fin.murabaha", content="murabaha contracts installments"),
        document("4", "fin.deposits", content="deposits savings accounts"),
        document("5", "fin.customers", content="customers personal details"),
    ]
    store = make_store(rows, FlatEmbeddings())
    vector = [1.0, 0.0, 0.0]

    dense = store.search_by_vector(vector, k=3, view_ids=["1", "2", "3", "4", "5"])
    hybrid = store.search_batched(k=3, query="murabaha contracts", vector=vector,
                                  view_ids=["1", "2", "3", "4", "5"])

    assert dense[0].id != "3"
    assert hybrid[0].id == "3"
    assert [doc.id for doc in store.search("murabaha contracts", k=5, view_ids=["2"])] == ["2"]


def test_hybrid_ranks_all_permission_ids_together(make_store, monkeypatch, sparse_model):
    monkeypatch.setenv("QDRANT_HYBRID", "1")
    store = make_store([
        document("1", "fin.orders", content="orders murabaha"),
        document("30001", "fin.customers", content="customers murabaha"),
    ])
    permissions = [str(i) for i in range(1, 30002)]
    found = store.search_batched(k=1, view_ids=permissions, query="customers murabaha",
                                 vector=[1.0, 0.0, 0.1], scores=True)

    assert found[0][0].id == "30001"


def test_local_collections_share_a_path_and_keep_data_after_reopening(make_store, monkeypatch, tmp_path):
    monkeypatch.delenv("QDRANT_LOCATION")
    monkeypatch.setenv("QDRANT_PATH", str(tmp_path / "catalog"))
    schema = make_store([document("1", "fin.customers")], name="schema")
    # Both spellings address the same storage directory.
    monkeypatch.setenv("QDRANT_PATH", str(tmp_path / "catalog") + "/.")
    samples = make_store([document("1_tuple_0", "fin.customers", content="sample")], name="samples")

    assert schema.get_views(["1"])[0].page_content == "customers"
    assert samples.get_views(["1"])[0].page_content == "sample"

    schema.client.client.close()
    reopened_schema = make_store(name="schema")
    reopened_samples = make_store(name="samples")
    assert reopened_schema.get_views(["1"])[0].page_content == "customers"
    assert reopened_samples.get_views(["1"])[0].page_content == "sample"


def test_existing_dense_collection_supports_hybrid_after_resync(make_store, monkeypatch, tmp_path, sparse_model):
    monkeypatch.delenv("QDRANT_LOCATION")
    monkeypatch.setenv("QDRANT_PATH", str(tmp_path / "catalog"))
    rows = [
        document("1", "fin.orders", content="orders purchases suppliers"),
        document("2", "fin.payroll", content="payroll monthly salaries"),
        document("3", "fin.murabaha", content="murabaha contracts installments"),
    ]
    dense = make_store(rows, FlatEmbeddings())
    dense.client.client.close()
    monkeypatch.setenv("QDRANT_HYBRID", "1")
    hybrid = make_store(rows, FlatEmbeddings())

    found = hybrid.search("murabaha contracts", k=1, view_ids=["1", "2", "3"])
    assert found[0].id == "3"
