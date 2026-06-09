import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import requests
import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/image_cache")
EMBEDDINGS_FILE = CACHE_DIR / "image_embeddings.npy"
VALID_MASK_FILE = CACHE_DIR / "valid_mask.npy"

_TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _first_url(cell) -> Optional[str]:
    if pd.isna(cell):
        return None
    url = str(cell).split("|")[0].strip()
    return url if url.startswith("http") else None


_ADAPTER = requests.adapters.HTTPAdapter(pool_connections=100, pool_maxsize=100)
_SESSION = requests.Session()
_SESSION.headers["User-Agent"] = "Mozilla/5.0"
_SESSION.mount("http://", _ADAPTER)
_SESSION.mount("https://", _ADAPTER)


def _fetch(url: str) -> Optional[Image.Image]:
    try:
        r = _SESSION.get(url, timeout=3, stream=True)
        r.raise_for_status()
        data = b"".join(r.iter_content(65536))
        return Image.open(BytesIO(data)).convert("RGB")
    except Exception:
        return None


def build_image_embeddings(df: pd.DataFrame, batch_size: int = 128, max_workers: int = 64, force_rebuild: bool = False):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if not force_rebuild and EMBEDDINGS_FILE.exists() and VALID_MASK_FILE.exists():
        return np.load(EMBEDDINGS_FILE), np.load(VALID_MASK_FILE)

    urls = df["medium"].apply(_first_url).tolist()
    images: List[Optional[Image.Image]] = [None] * len(urls)

    valid_urls = [(i, url) for i, url in enumerate(urls) if url]
    logger.info("downloading %d images with %d workers...", len(valid_urls), max_workers)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch, url): i for i, url in valid_urls}
        done = 0
        for fut in as_completed(futures):
            images[futures[fut]] = fut.result()
            done += 1
            if done % 1000 == 0:
                logger.info("  %d/%d done", done, len(valid_urls))

    has_image = np.array([img is not None for img in images], dtype=bool)
    valid_idx = np.where(has_image)[0]
    logger.info("%d/%d images downloaded successfully", has_image.sum(), len(urls))

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    resnet.fc = torch.nn.Identity()
    resnet.eval()
    resnet.to(device)
    logger.info("running ResNet-18 inference on %s", device)

    embeddings = np.zeros((len(urls), 512), dtype="float32")
    with torch.no_grad():
        for start in range(0, len(valid_idx), batch_size):
            pos = valid_idx[start: start + batch_size]
            tensors = torch.stack([_TRANSFORM(images[i]) for i in pos]).to(device)
            feats = resnet(tensors).cpu().numpy()
            embeddings[pos] = feats / np.linalg.norm(feats, axis=1, keepdims=True).clip(min=1e-9)
            if start % 2000 == 0:
                logger.info("  inference %d/%d", start, len(valid_idx))

    np.save(EMBEDDINGS_FILE, embeddings)
    np.save(VALID_MASK_FILE, has_image)
    return embeddings, has_image


def load_cached_embeddings():
    if not EMBEDDINGS_FILE.exists() or not VALID_MASK_FILE.exists():
        return None
    return np.load(EMBEDDINGS_FILE), np.load(VALID_MASK_FILE)


def find_similar_multimodal(
    product_id: str,
    num_similar: int,
    df: pd.DataFrame,
    text_embeddings: np.ndarray,
    image_embeddings: np.ndarray,
    has_image: np.ndarray,
    alpha: float = 0.6,
) -> List[str]:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")

    hits = df.index[df["uniq_id"] == product_id]
    if len(hits) == 0:
        raise ValueError(f"product_id '{product_id}' not found")

    idx = int(hits[0])
    text_scores = text_embeddings @ text_embeddings[idx]

    if has_image[idx]:
        image_scores = image_embeddings @ image_embeddings[idx]
        combined = np.where(has_image, alpha * text_scores + (1.0 - alpha) * image_scores, text_scores)
    else:
        combined = text_scores

    combined[idx] = -2.0
    top_n = np.argsort(combined)[::-1][:num_similar]
    return df.iloc[top_n]["uniq_id"].tolist()


if __name__ == "__main__":
    from product_similarity.similarity import load_dataset
    logging.basicConfig(level=logging.INFO)
    build_image_embeddings(load_dataset(), force_rebuild=True)
