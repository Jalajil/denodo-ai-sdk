"""
 Copyright (c) 2026. DENODO Technologies.
 http://www.denodo.com
 All rights reserved.
"""

"""Qdrant backend for UniformVectorStore.

Qdrant is worth a dedicated adapter rather than a thin langchain wrapper because
it can do something the other supported stores cannot: serve a sparse lexical
index alongside the dense one and fuse the two server-side. That matters here
specifically. Dense retrieval has to carry the whole burden of matching a
question against a description written in the same language, and it is weakest
on exactly the terms a finance user is most likely to type - a branch name, a
product name, a city. A BM25 index catches those by surface form.

The reason this was not worth doing on Chroma or pgvector is that a useful BM25
over a non-Latin language needs a real analyzer - stemming, stopwords - and
neither provides one. FastEmbed's BM25 does, via snowball, including Arabic.

What this adapter adds beyond parity:

  - hybrid dense + sparse retrieval fused with Reciprocal Rank Fusion, computed
    by Qdrant rather than stitched together client-side
  - keyword payload indexes on the fields the SDK filters by, so filtered
    vector search stays on the fast path instead of degrading to a scan
  - HNSW parameters tuned for recall rather than Qdrant's throughput-oriented
    defaults, plus an exact-search escape hatch for small catalogs
  - a deterministic id mapping, because Qdrant point ids must be UUIDs or
    integers and the SDK uses strings such as "42_0" and "last_update"
"""

import logging
import os
import threading
import uuid
import weakref

from utils.text_utils import normalize_for_lexical_index

# Stable namespace for deriving Qdrant point ids from the SDK's string ids.
# Fixed forever: changing it orphans every previously written point.
SDK_ID_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")

# Bumped whenever the text fed to the BM25 encoder changes. Sparse vectors are
# written at index time, so a bump means the stored vectors were built under
# different rules and the collection needs a full metadata re-synchronization.
#   1 - raw text, as fastembed received it
#   2 - normalize_for_lexical_index applied to documents, queries and stopwords
SPARSE_TEXT_VERSION = 2

# The sync bookkeeping document the SDK already writes after every metadata
# synchronization. Stamping the version onto its payload avoids adding a point
# of our own to the collection.
SYNC_DOCUMENT_ID = "last_update"

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"

# Embedded Qdrant locks its directory. Both SDK collections must use the same
# client. Weak references let Qdrant close it once its last store is released.
_LOCAL_CLIENTS = weakref.WeakValueDictionary()
_LOCAL_CLIENTS_LOCK = threading.Lock()

# The metadata fields the SDK filters on. Without a payload index Qdrant cannot
# use its filterable-HNSW path and a filtered search degrades badly once a
# catalog is more than a few thousand views.
INDEXED_PAYLOAD_FIELDS = (
    "metadata.view_id",
    "metadata.view_name",
    "metadata.database_name",
    "metadata.document_id",
)


def point_id(sdk_id):
    """Map an SDK document id onto a valid Qdrant point id, deterministically.

    Deterministic so that re-syncing a view overwrites its point instead of
    duplicating it, and so deletes can address a point without a lookup table.
    """
    return str(uuid.uuid5(SDK_ID_NAMESPACE, str(sdk_id)))


def hybrid_enabled():
    return os.getenv("QDRANT_HYBRID", "1") == "1"


def sparse_normalization_enabled():
    r"""Whether to fold orthographic variants before BM25 sees the text.

    On by default. Without it, fastembed's tokenizer - `re.sub(r"[^\w]", " ",
    text.lower())` - treats Arabic diacritics and bidi marks as word separators,
    so a vocalized word is shattered into single letters, and the snowball
    stemmer never gets the chance to fold anything. Turn it off only to compare
    against the previous behaviour, and re-synchronize when you do.
    """
    return os.getenv("QDRANT_SPARSE_NORMALIZE", "1") == "1"


def sparse_language():
    """Language for BM25 stemming and stopwords.

    Defaults to the response language, since the catalog descriptions and the
    questions are normally in the same language, then to English.
    """
    configured = os.getenv("QDRANT_SPARSE_LANGUAGE", "").strip()
    if configured:
        return configured.lower()
    return (os.getenv("RESPONSE_LANGUAGE", "english") or "english").strip().lower()


_SPARSE_MODEL = None


def get_sparse_model():
    """Load the BM25 encoder once, or return None if it is unavailable.

    Sparse retrieval is an enhancement, not a dependency: if fastembed is
    missing or the model cannot be fetched, retrieval falls back to dense only
    rather than failing.
    """
    global _SPARSE_MODEL
    if _SPARSE_MODEL is not None:
        return _SPARSE_MODEL or None

    try:
        from fastembed.sparse.bm25 import Bm25
    except ImportError:
        logging.warning(
            "QDRANT_HYBRID is on but fastembed is not installed; using dense retrieval only. "
            "Install fastembed and snowballstemmer to enable sparse retrieval."
        )
        _SPARSE_MODEL = False
        return None

    language = sparse_language()
    try:
        _SPARSE_MODEL = Bm25(os.getenv("QDRANT_SPARSE_MODEL", "Qdrant/bm25"), language=language)
        logging.info(f"Qdrant sparse retrieval enabled (BM25, language='{language}').")
    except Exception as error:
        # An unsupported language is the likely cause; retry with the default
        # rather than dropping sparse retrieval entirely.
        logging.warning(f"BM25 with language='{language}' failed ({error}); retrying with English.")
        try:
            _SPARSE_MODEL = Bm25(os.getenv("QDRANT_SPARSE_MODEL", "Qdrant/bm25"))
        except Exception as fallback_error:
            logging.warning(f"Sparse retrieval unavailable ({fallback_error}); using dense only.")
            _SPARSE_MODEL = False
            return None

    if sparse_normalization_enabled():
        _normalize_stopwords(_SPARSE_MODEL)
        logging.info("Qdrant sparse text normalization enabled (QDRANT_SPARSE_NORMALIZE=1).")

    return _SPARSE_MODEL


def _normalize_stopwords(model):
    """Fold the shipped stopword list the same way the indexed text is folded.

    fastembed checks membership BEFORE stemming (`Bm25._stem`), against the raw
    list from the model repository. Normalizing the text but not the list makes
    most stopwords stop matching: of the 754 Arabic stopwords in Qdrant/bm25, 305
    change under this fold and only 49 of those folded forms are already present,
    so ~256 would silently start being indexed as ordinary terms.
    """
    stopwords = getattr(model, "stopwords", None)
    if not stopwords:
        return

    try:
        model.stopwords = {normalize_for_lexical_index(word) for word in stopwords}
    except Exception as error:
        # A fastembed change could rename or retype this; losing stopword
        # filtering costs precision, not correctness, so do not fail the request.
        logging.warning(f"Could not normalize BM25 stopwords ({error}); leaving them as shipped.")


def encode_sparse(text, is_query=False):
    """Return a Qdrant SparseVector for `text`, or None if sparse is unavailable.

    Queries and documents use different BM25 code paths: document encoding
    applies term saturation and length normalisation, query encoding does not.
    """
    model = get_sparse_model()
    if not model or not text or not str(text).strip():
        return None

    from qdrant_client import models

    if sparse_normalization_enabled():
        # Symmetric by construction: the same fold runs over documents at index
        # time and over the question at query time. Folding one side only would
        # be worse than folding neither.
        text = normalize_for_lexical_index(text)
        if not text:
            return None

    try:
        embedder = model.query_embed(text) if is_query else model.embed([text])
        embedding = next(iter(embedder))
    except Exception as error:
        logging.warning(f"Sparse encoding failed ({error}); this request uses dense retrieval only.")
        return None

    indices = [int(i) for i in embedding.indices]
    if not indices:
        return None

    return models.SparseVector(indices=indices, values=[float(v) for v in embedding.values])


def build_client():
    """Connect to Qdrant. ':memory:' and a local path are both supported."""
    from qdrant_client import QdrantClient

    location = os.getenv("QDRANT_LOCATION", "").strip()
    url = os.getenv("QDRANT_URL", "").strip()
    path = os.getenv("QDRANT_PATH", "").strip()
    api_key = os.getenv("QDRANT_API_KEY", "").strip() or None
    timeout = int(os.getenv("QDRANT_TIMEOUT", "60"))

    if location:
        return QdrantClient(location=location, timeout=timeout)
    if path:
        path = os.path.normcase(os.path.realpath(os.path.expanduser(path)))
        with _LOCAL_CLIENTS_LOCK:
            client = _LOCAL_CLIENTS.get(path)
            if client is not None:
                try:
                    # A caller may have explicitly closed the shared client.
                    # This is a local operation and does not contact a server.
                    client.get_collections()
                except RuntimeError as error:
                    if "closed" not in str(error).lower():
                        raise
                else:
                    return client
            client = QdrantClient(path=path, timeout=timeout)
            _LOCAL_CLIENTS[path] = client
            return client
    if not url:
        url = "http://localhost:6333"

    return QdrantClient(
        url=url,
        api_key=api_key,
        timeout=timeout,
        prefer_grpc=os.getenv("QDRANT_PREFER_GRPC", "0") == "1",
        https=os.getenv("QDRANT_HTTPS", "0") == "1" or url.startswith("https://"),
    )


def ensure_collection(client, collection_name, dimensions, with_sparse):
    """Create the collection if absent, with recall-oriented HNSW parameters.

    Qdrant's defaults (m=16, ef_construct=100) favour build time and memory.
    A schema catalog is small by vector-database standards - thousands of views,
    not millions - so the cost of a denser graph is negligible and the recall is
    worth more, which is the same trade the Chroma configuration already makes.
    """
    from qdrant_client import models

    if client.collection_exists(collection_name):
        if with_sparse:
            params = client.get_collection(collection_name).config.params
            if SPARSE_VECTOR_NAME not in (params.sparse_vectors or {}):
                client.create_vector_name(
                    collection_name=collection_name,
                    vector_name=SPARSE_VECTOR_NAME,
                    vector_name_config=models.SparseVectorNameConfig(
                        sparse=models.SparseVectorConfig(modifier=models.Modifier.IDF)
                    ),
                    wait=True,
                )
                logging.warning(
                    f"Enabled sparse vectors for existing Qdrant collection '{collection_name}'. "
                    "Re-synchronize all metadata and sampled data to populate the sparse index."
                )
        return False

    hnsw = models.HnswConfigDiff(
        m=int(os.getenv("QDRANT_HNSW_M", "32")),
        ef_construct=int(os.getenv("QDRANT_HNSW_EF_CONSTRUCT", "256")),
    )
    vectors_config = {
        DENSE_VECTOR_NAME: models.VectorParams(
            size=dimensions,
            distance=models.Distance.COSINE,
            hnsw_config=hnsw,
            on_disk=os.getenv("QDRANT_ON_DISK", "0") == "1",
        )
    }
    sparse_vectors_config = None
    if with_sparse:
        sparse_vectors_config = {
            SPARSE_VECTOR_NAME: models.SparseVectorParams(
                modifier=models.Modifier.IDF,  # Qdrant computes IDF across the collection
            )
        }

    client.create_collection(
        collection_name=collection_name,
        vectors_config=vectors_config,
        sparse_vectors_config=sparse_vectors_config,
    )
    logging.info(
        f"Created Qdrant collection '{collection_name}' "
        f"(dim={dimensions}, hnsw m={hnsw.m}/ef_construct={hnsw.ef_construct}, "
        f"sparse={'on' if with_sparse else 'off'})"
    )
    return True


def ensure_payload_indexes(client, collection_name):
    """Index the fields the SDK filters on. Idempotent."""
    from qdrant_client import models

    for field in INDEXED_PAYLOAD_FIELDS:
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
                wait=True,
            )
        except Exception:
            # Already present, or the server rejected a duplicate. Not fatal:
            # a missing index costs speed, not correctness.
            pass


def search_params():
    """Per-query search tuning.

    hnsw_ef trades latency for recall at query time. `exact` bypasses the graph
    entirely for a brute-force scan - viable, and worth it, on a catalog of a few
    thousand views where the whole point is not to miss the right one.
    """
    from qdrant_client import models

    return models.SearchParams(
        hnsw_ef=int(os.getenv("QDRANT_HNSW_EF", "256")),
        exact=os.getenv("QDRANT_EXACT_SEARCH", "0") == "1",
    )


def build_filter(view_ids=None, database_names=None, tag_names=None, view_names=None):
    """Translate the SDK's filter shape into a Qdrant Filter.

    Semantics match the other backends: view_ids is a hard AND (it carries the
    user's row-level permissions and must never be relaxed), while database,
    tag and view-name conditions OR together.
    """
    from qdrant_client import models

    must = []
    should = []

    if view_ids:
        must.append(models.FieldCondition(
            key="metadata.view_id",
            match=models.MatchAny(any=[str(v) for v in view_ids]),
        ))

    for database_name in database_names or []:
        should.append(models.FieldCondition(
            key="metadata.database_name", match=models.MatchValue(value=database_name)))

    for tag_name in tag_names or []:
        # Tags are stored as one boolean-ish payload key per tag, so the key
        # itself is dynamic and cannot be covered by a static payload index.
        should.append(models.FieldCondition(
            key=f"metadata.tag_{tag_name}", match=models.MatchValue(value="1")))

    for view_name in view_names or []:
        should.append(models.FieldCondition(
            key="metadata.view_name", match=models.MatchValue(value=view_name)))

    if should:
        if must:
            # An OR group nested inside the AND, so permissions still bind.
            must.append(models.Filter(should=should))
        else:
            return models.Filter(should=should)

    if not must:
        return None

    return models.Filter(must=must)


class QdrantStore:
    """Presents the slice of the langchain VectorStore surface UniformVectorStore
    uses, and adds a Qdrant-native hybrid search path."""

    def __init__(self, collection_name, embeddings, dimensions):
        from langchain_qdrant import QdrantVectorStore, RetrievalMode

        self.collection_name = collection_name
        self.embeddings = embeddings
        self.dimensions = dimensions
        self.client = build_client()

        self.hybrid = hybrid_enabled() and get_sparse_model() is not None
        ensure_collection(self.client, collection_name, dimensions, with_sparse=self.hybrid)
        ensure_payload_indexes(self.client, collection_name)

        # If the collection predates hybrid being switched on it has no sparse
        # vector configured; detect that rather than failing every write.
        if self.hybrid and not self._collection_has_sparse():
            logging.warning(
                f"Collection '{collection_name}' has no sparse vector configured. "
                f"Hybrid retrieval is disabled for it; re-synchronize the metadata to enable it."
            )
            self.hybrid = False

        if self.hybrid:
            self._warn_on_stale_sparse_text()

        self.store = QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embedding=embeddings,
            vector_name=DENSE_VECTOR_NAME,
            retrieval_mode=RetrievalMode.DENSE,
            validate_collection_config=False,
        )

    def _collection_has_sparse(self):
        try:
            config = self.client.get_collection(self.collection_name).config.params
            return SPARSE_VECTOR_NAME in (getattr(config, "sparse_vectors", None) or {})
        except Exception:
            return False

    def _stored_sparse_text_version(self):
        """Version stamped by whoever last synchronized this collection, or None."""
        try:
            points = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[point_id(SYNC_DOCUMENT_ID)],
                with_payload=True,
            )
        except Exception:
            return None

        if not points:
            return None
        return (points[0].payload or {}).get("sparse_text_version")

    def _warn_on_stale_sparse_text(self):
        """Say so, loudly, when the stored sparse vectors predate the current rules.

        Querying a normalized question against an un-normalized index is worse
        than normalizing neither, so this must not fail quietly. The stamp rides
        on the sync bookkeeping document the SDK already writes at the end of a
        metadata synchronization; a collection that has never been synchronized
        has no stamp and is left alone.
        """
        stored = self._stored_sparse_text_version()
        if stored is None or stored == SPARSE_TEXT_VERSION:
            return

        logging.warning(
            f"Collection '{self.collection_name}' holds sparse vectors built with "
            f"SPARSE_TEXT_VERSION={stored}, but this build writes and queries version "
            f"{SPARSE_TEXT_VERSION}. Lexical (BM25) matching will underperform until the "
            f"metadata is re-synchronized. Run getMetadata over every database/tag, or set "
            f"QDRANT_SPARSE_NORMALIZE=0 to keep the old behaviour."
        )

    # --- writes ------------------------------------------------------------

    def add_documents(self, documents, ids=None, **kwargs):
        """Upsert documents, attaching a sparse vector when hybrid is enabled."""
        from qdrant_client import models

        ids = ids or [doc.id for doc in documents]
        points = []
        dense_vectors = self.embeddings.embed_documents([doc.page_content for doc in documents])

        for document, sdk_id, dense in zip(documents, ids, dense_vectors):
            vector = {DENSE_VECTOR_NAME: dense}
            if self.hybrid:
                sparse = encode_sparse(document.page_content, is_query=False)
                if sparse is not None:
                    vector[SPARSE_VECTOR_NAME] = sparse

            payload = {
                "page_content": document.page_content,
                "metadata": document.metadata,
                # Kept so a point can be traced back to the id the SDK uses.
                "sdk_id": str(sdk_id),
            }

            # The sync bookkeeping document is rewritten at the end of every
            # metadata synchronization, which is exactly when the sparse vectors
            # in this collection were (re)built - so it is the honest place to
            # record which rules built them.
            if str(sdk_id) == SYNC_DOCUMENT_ID and self.hybrid:
                payload["sparse_text_version"] = SPARSE_TEXT_VERSION

            points.append(models.PointStruct(
                id=point_id(sdk_id),
                vector=vector,
                payload=payload,
            ))

        self.client.upsert(collection_name=self.collection_name, points=points, wait=True)
        return list(ids)

    def delete(self, ids=None, **kwargs):
        from qdrant_client import models

        if not ids:
            return
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(points=[point_id(i) for i in ids]),
            wait=True,
        )

    # --- reads -------------------------------------------------------------

    def _to_documents(self, points, with_scores):
        from langchain_core.documents import Document

        results = []
        for point in points:
            payload = point.payload or {}
            document = Document(
                id=payload.get("sdk_id"),
                page_content=payload.get("page_content", ""),
                metadata=payload.get("metadata", {}) or {},
            )
            results.append((document, point.score) if with_scores else document)
        return results

    def _query(self, dense_vector, k, query_filter, query_text=None, with_scores=False):
        from qdrant_client import models

        sparse = encode_sparse(query_text, is_query=True) if (self.hybrid and query_text) else None

        if sparse is not None:
            # Fuse dense and sparse rankings server-side with RRF. Each branch
            # over-fetches so the fusion has something to work with; a term that
            # ranks 30th on the dense side can still win after fusion.
            prefetch_limit = max(k * int(os.getenv("QDRANT_PREFETCH_FACTOR", "4")), k)
            response = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    models.Prefetch(query=dense_vector, using=DENSE_VECTOR_NAME,
                                    limit=prefetch_limit, filter=query_filter,
                                    params=search_params()),
                    models.Prefetch(query=sparse, using=SPARSE_VECTOR_NAME,
                                    limit=prefetch_limit, filter=query_filter),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=k,
                with_payload=True,
            )
        else:
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=dense_vector,
                using=DENSE_VECTOR_NAME,
                limit=k,
                query_filter=query_filter,
                search_params=search_params(),
                with_payload=True,
            )

        return self._to_documents(response.points, with_scores)

    def similarity_search_by_vector(self, embedding, k=4, filter=None, query_text=None, **kwargs):
        return self._query(embedding, k, filter, query_text, with_scores=False)

    def similarity_search_with_score_by_vector(self, embedding, k=4, filter=None, query_text=None, **kwargs):
        return self._query(embedding, k, filter, query_text, with_scores=True)

    def similarity_search(self, query, k=4, filter=None, **kwargs):
        return self._query(self.embeddings.embed_query(query), k, filter, query, with_scores=False)

    def similarity_search_with_score(self, query, k=4, filter=None, **kwargs):
        return self._query(self.embeddings.embed_query(query), k, filter, query, with_scores=True)
