import logging
from contextlib import asynccontextmanager
from enum import Enum
from typing import List

from fastapi import FastAPI, HTTPException, Query

from .similarity import load_dataset, build_feature_matrix, find_similar_products
from .text_models import build_tfidf_matrix, find_similar_by_tfidf
from .text_models import build_word2vec_index, find_similar_by_word2vec
from .text_models import build_text_index, find_similar_by_text
from .image_models import load_cached_embeddings, find_similar_multimodal

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

    _state["w2v_index"], _state["w2v_embeddings"], _ = build_word2vec_index(_state["df"])
    _state["text_index"], _state["text_embeddings"] = build_text_index(_state["df"])

    image_data = load_cached_embeddings()
    if image_data:
        _state["image_embeddings"], _state["has_image"] = image_data
        logger.info("multimodal enabled")
    else:
        _state["image_embeddings"] = _state["has_image"] = None
        logger.warning("no image cache — run: uv run python -m product_similarity.image_models")

    yield
    _state.clear()


app = FastAPI(title="Product Similarity API", version="1.0.0", lifespan=lifespan)


@app.get("/find_similar_products", response_model=List[str])
def get_similar_products(
    product_id: str = Query(...),
    num_similar: int = Query(5, ge=1, le=100),
    model: Model = Query(Model.semantic),
    alpha: float = Query(0.6, ge=0.0, le=1.0, description="Text/image blend — only used when model=multimodal"),
):
    df = _state["df"]
    try:
        if model == Model.structured:
            return find_similar_products(product_id, num_similar, df, _state["feature_matrix"])

        if model == Model.tfidf:
            return find_similar_by_tfidf(product_id, num_similar, df, _state["tfidf_matrix"])

        if model == Model.word2vec:
            return find_similar_by_word2vec(product_id, num_similar, df, _state["w2v_index"], _state["w2v_embeddings"])

        if model == Model.semantic:
            return find_similar_by_text(product_id, num_similar, df, _state["text_index"], _state["text_embeddings"])

        if model == Model.multimodal:
            if _state["image_embeddings"] is None:
                raise HTTPException(status_code=503, detail="Run: uv run python -m product_similarity.image_models")
            return find_similar_multimodal(product_id, num_similar, df, _state["text_embeddings"], _state["image_embeddings"], _state["has_image"], alpha=alpha)

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
