import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import List

DATA_PATH = Path("data/marketing_sample_for_amazon_com-amazon_fashion_products__20200201_20200430__30k_data.ldjson")
ZIP_PATH = Path("data/archive.zip")


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    if not Path(path).exists():
        if ZIP_PATH.exists():
            with zipfile.ZipFile(ZIP_PATH) as z:
                z.extractall(ZIP_PATH.parent)
        else:
            raise FileNotFoundError(f"Dataset not found at {path} and no archive.zip to extract from")
    return pd.read_json(path, lines=True).reset_index(drop=True)


def _parse_price(val) -> float:
    if pd.isna(val):
        return np.nan
    cleaned = re.sub(r"[^\d.]", "", str(val))
    try:
        return float(cleaned)
    except ValueError:
        return np.nan


def _top_category(cat_dict) -> str:
    if not isinstance(cat_dict, dict) or not cat_dict:
        return "unknown"
    return list(cat_dict.keys())[0]


def _yn_to_int(val) -> int:
    return 1 if str(val).strip().upper() == "Y" else 0


def build_feature_matrix(df: pd.DataFrame):
    # dataset has no 'color' column — color info lives in product name and meta_keywords
    # image URLs aren't numeric; using amazon_prime + best_seller as proxy signals
    work = pd.DataFrame(index=df.index)

    work["price"] = df["sales_price"].apply(_parse_price)
    work["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    # weight uses 999999999 as sentinel for missing
    raw_w = pd.to_numeric(df["weight"], errors="coerce")
    work["weight"] = raw_w.where(raw_w < 999_999_999, other=np.nan)

    work["is_prime"] = df["amazon_prime__y_or_n"].apply(_yn_to_int)
    work["is_bestseller"] = df["best_seller_tag__y_or_n"].apply(_yn_to_int)

    num_cols = ["price", "rating", "weight", "is_prime", "is_bestseller"]
    for col in num_cols:
        med = work[col].median()
        work[col] = work[col].fillna(0.0 if np.isnan(med) else med)

    le_brand = LabelEncoder()
    work["brand_enc"] = le_brand.fit_transform(df["brand"].fillna("unknown").astype(str))

    le_cat = LabelEncoder()
    work["category_enc"] = le_cat.fit_transform(df["parent___child_category__all"].apply(_top_category))

    all_cols = num_cols + ["brand_enc", "category_enc"]
    scaler = StandardScaler()
    matrix = scaler.fit_transform(work[all_cols]).astype("float32")
    return np.nan_to_num(matrix, nan=0.0), all_cols


def find_similar_products(
    product_id: str,
    num_similar: int,
    df: pd.DataFrame,
    feature_matrix: np.ndarray,
) -> List[str]:
    hits = df.index[df["uniq_id"] == product_id]
    if len(hits) == 0:
        raise ValueError(f"product_id '{product_id}' not found")

    idx = int(hits[0])
    scores = cosine_similarity(feature_matrix[idx: idx + 1], feature_matrix)[0]
    scores[idx] = -2.0

    rating = pd.to_numeric(df["rating"], errors="coerce").fillna(0).to_numpy()
    price = pd.to_numeric(df["sales_price"], errors="coerce").fillna(0).to_numpy()

    # lexsort reads right-to-left: primary=score desc, then rating desc, then price asc
    order = np.lexsort((price, -rating, -scores))
    return df.iloc[order[:num_similar]]["uniq_id"].tolist()
