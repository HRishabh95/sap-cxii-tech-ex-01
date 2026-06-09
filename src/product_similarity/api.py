import logging
import warnings
from contextlib import asynccontextmanager
from enum import Enum
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from .similarity import load_dataset, build_feature_matrix, find_similar_products
from .image_models import load_cached_embeddings, find_similar_multimodal
from .text_models import build_tfidf_matrix, find_similar_by_tfidf
from .text_models import build_word2vec_index, find_similar_by_word2vec
from .text_models import build_text_index, find_similar_by_text, _SBERT_MODEL

logger = logging.getLogger(__name__)
_state: dict = {}


class Model(str, Enum):
    structured = "structured"
    tfidf = "tfidf"
    word2vec = "word2vec"
    semantic = "semantic"
    multimodal = "multimodal"


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["df"] = load_dataset()

    _state["feature_matrix"] = build_feature_matrix(_state["df"])[0]
    _state["tfidf_matrix"] = build_tfidf_matrix(_state["df"])[0]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        _state["w2v_index"], _state["w2v_embeddings"], _state["w2v_model"] = build_word2vec_index(_state["df"])
    _state["text_index"], _state["text_embeddings"], _state["sbert_model"] = build_text_index(_state["df"])

    image_data = load_cached_embeddings()
    if image_data:
        _state["image_embeddings"], _state["has_image"] = image_data
        logger.info("multimodal enabled")
    else:
        _state["image_embeddings"] = _state["has_image"] = None

    yield
    _state.clear()


app = FastAPI(title="Product Similarity API", version="1.0.0", lifespan=lifespan)


class ProductResult(BaseModel):
    product_id: str
    product_name: str
    brand: Optional[str]
    price: Optional[str]
    rating: Optional[str]
    category: Optional[str]


def _resolve(query: str) -> str:
    """Accept a product_id or free text. Returns a product_id either way."""
    df = _state["df"]
    if (df["uniq_id"] == query).any():
        return query
    vec = _state["sbert_model"].encode([query], convert_to_numpy=True)
    vec = (vec / np.linalg.norm(vec, axis=1, keepdims=True).clip(min=1e-9)).astype("float32")
    _, ids = _state["text_index"].search(vec, 1)
    return str(df.iloc[ids[0][0]]["uniq_id"])


def _hydrate(ids: List[str]) -> List[ProductResult]:
    df = _state["df"]
    rows = df.set_index("uniq_id").loc[ids]
    return [
        ProductResult(
            product_id=pid,
            product_name=str(row.get("product_name", "")),
            brand=str(row.get("brand", "")) or None,
            price=str(row.get("sales_price", "")) or None,
            rating=str(row.get("rating", "")) or None,
            category=str(list(row.get("parent___child_category__all") or {}).pop(0) if isinstance(row.get("parent___child_category__all"), dict) else "") or None,
        )
        for pid, (_, row) in zip(ids, rows.iterrows())
    ]


@app.get("/find_similar_products", response_model=List[ProductResult])
def get_similar_products(
    product_id: str = Query(..., description="Product ID or free text e.g. 'red cotton kurta'"),
    num_similar: int = Query(5, ge=1, le=100),
    model: Model = Query(Model.semantic),
    alpha: float = Query(0.6, ge=0.0, le=1.0, description="Text/image blend — only used when model=multimodal"),
):
    df = _state["df"]
    try:
        product_id = _resolve(product_id)

        if model == Model.structured:
            ids = find_similar_products(product_id, num_similar, df, _state["feature_matrix"])
        elif model == Model.tfidf:
            ids = find_similar_by_tfidf(product_id, num_similar, df, _state["tfidf_matrix"])
        elif model == Model.word2vec:
            ids = find_similar_by_word2vec(product_id, num_similar, df, _state["w2v_index"], _state["w2v_embeddings"])
        elif model == Model.semantic:
            ids = find_similar_by_text(product_id, num_similar, df, _state["text_index"], _state["text_embeddings"])
        elif model == Model.multimodal:
            if _state["image_embeddings"] is None:
                raise HTTPException(status_code=503, detail="Run: uv run python -m product_similarity.image_models")
            ids = find_similar_multimodal(product_id, num_similar, df, _state["text_embeddings"], _state["image_embeddings"], _state["has_image"], alpha=alpha)

        return _hydrate(ids)

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


