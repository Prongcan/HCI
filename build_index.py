"""
build_index.py
Extract image embeddings from the GroceryStoreDataset using
Alibaba Cloud Bailian (tongyi-embedding-vision-plus-2026-03-06)
and upsert them into Upstash Vector index.

Model: tongyi-embedding-vision-plus-2026-03-06 (512-dim)
Dataset: GroceryStoreDataset (https://github.com/marcusklasson/GroceryStoreDataset)
"""

import os
import time
import base64
import json
from pathlib import Path

import requests
from upstash_vector import Index, Vector

# ── Bailian API config ────────────────────────────────────────────
BAILIAN_API_KEY = "sk-fd02acce73634ecbbab21757c1cada10"
BAILIAN_EMBEDDING_URL = (
    "https://dashscope.aliyuncs.com/api/v1/services/"
    "embeddings/multimodal-embedding/multimodal-embedding"
)
EMBEDDING_MODEL = "tongyi-embedding-vision-plus-2026-03-06"
EMBEDDING_DIM = 512

# ── Upstash Vector config ─────────────────────────────────────────
UPSTASH_URL = "https://lasting-raven-85217-us1-vector.upstash.io"
UPSTASH_TOKEN = "ABcFMGxhc3RpbmctcmF2ZW4tODUyMTctdXMxYWRtaW5aVFEzWXpSaVltTXROREkwTUMwMFkyRXlMV0l5TlRjdE5tWTNNakl5TTJGbE5qTXc="

index = Index(url=UPSTASH_URL, token=UPSTASH_TOKEN)

# ── Dataset paths ─────────────────────────────────────────────────
DATASET_ROOT = Path(__file__).parent / "dataset" / "dataset"
TRAIN_DIR = DATASET_ROOT / "train"
TEST_DIR = DATASET_ROOT / "test"


def collect_images():
    """Collect all images from train + test with class metadata.

    Directory structure varies:
      train/Fruit/Apple/Golden-Delicious/Golden-Delicious_001.jpg  (5 levels)
      train/Vegetables/Asparagus/Asparagus_001.jpg                 (4 levels)
    So we use rglob to find images recursively.
    """
    images = []
    for split_dir in [TRAIN_DIR, TEST_DIR]:
        if not split_dir.exists():
            continue
        split = split_dir.name
        for img_path in sorted(split_dir.rglob("*.jpg")):
            rel_path = img_path.relative_to(DATASET_ROOT)
            # Extract class info from path parts
            # rel_path = train/Fruit/Apple/Golden-Delicious/file.jpg
            #   or     = test/Vegetables/Asparagus/file.jpg
            parts = rel_path.parts  # ('train', 'Fruit', 'Apple', ...)
            coarse_class = parts[1] if len(parts) > 2 else ""
            fine_class = parts[2] if len(parts) > 2 else ""
            images.append({
                "path": str(img_path),
                "rel_path": str(rel_path),
                "split": split,
                "coarse_class": coarse_class,
                "fine_class": fine_class,
                "filename": img_path.name,
            })
    return images


def get_image_embeddings_batch(image_paths: list[str]) -> list[list[float]]:
    """Get embeddings for a batch of images (up to 10) via Bailian API."""
    contents = []
    for p in image_paths:
        with open(p, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        contents.append({"image": f"data:image/jpeg;base64,{img_b64}"})

    data = {
        "model": EMBEDDING_MODEL,
        "input": {"contents": contents},
        "parameters": {"dimension": EMBEDDING_DIM},
    }
    headers = {
        "Authorization": f"Bearer {BAILIAN_API_KEY}",
        "Content-Type": "application/json",
    }

    for attempt in range(3):
        try:
            resp = requests.post(
                BAILIAN_EMBEDDING_URL, headers=headers, json=data, timeout=60
            )
            resp.raise_for_status()
            result = resp.json()
            embeddings = []
            for item in result["output"]["embeddings"]:
                embeddings.append(item["embedding"])
            return embeddings
        except Exception as e:
            print(f"    [Retry {attempt+1}] API error: {e}")
            time.sleep(2 * (attempt + 1))

    return None


def build_and_upload(api_batch=10, upstash_batch=50):
    """Extract embeddings via Bailian API and upload to Upstash Vector."""
    images = collect_images()
    print(f"Total images found: {len(images)}")

    vectors = []
    uploaded = 0
    failed = 0

    for i in range(0, len(images), api_batch):
        batch = images[i : i + api_batch]
        batch_paths = [img["path"] for img in batch]

        print(f"  Extracting embeddings {i+1}-{min(i+api_batch, len(images))}/{len(images)} ...", end=" ", flush=True)
        embeddings = get_image_embeddings_batch(batch_paths)

        if embeddings is None or len(embeddings) != len(batch):
            print("FAILED")
            failed += len(batch)
            continue

        print("OK")

        for img_info, emb in zip(batch, embeddings):
            img_id = img_info["rel_path"].replace("/", "_").replace("\\", "_")
            metadata = {
                "path": img_info["path"],
                "rel_path": img_info["rel_path"],
                "split": img_info["split"],
                "coarse_class": img_info["coarse_class"],
                "fine_class": img_info["fine_class"],
                "filename": img_info["filename"],
            }
            vectors.append(Vector(id=img_id, vector=emb, metadata=metadata))

        if len(vectors) >= upstash_batch:
            print(f"  -> Uploading {len(vectors)} vectors to Upstash ...")
            index.upsert(vectors)
            uploaded += len(vectors)
            vectors = []
            time.sleep(0.3)

    # Upload remaining
    if vectors:
        print(f"  -> Uploading final {len(vectors)} vectors ...")
        index.upsert(vectors)
        uploaded += len(vectors)

    print(f"\nDone! Uploaded: {uploaded}, Failed: {failed}, Total: {len(images)}")


if __name__ == "__main__":
    build_and_upload(api_batch=10, upstash_batch=50)
