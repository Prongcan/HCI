"""
app.py
Image Search Interface based on Five-Stage Search Framework.
Uses Bailian multimodal embedding API + Upstash Vector for semantic image retrieval.

Five Stages implemented:
  1. Formulation  - text input, image upload, live query preview
  2. Initiation   - explicit "Search" button
  3. Review       - result count summary
  4. Refinement   - top-K slider to adjust result count
  5. Use          - download / add to favorites
"""

import json
import datetime
import base64
import io
from pathlib import Path

import requests
import gradio as gr
from PIL import Image
from upstash_vector import Index

# ════════════════════════════════════════════════════════════════════
# Configuration
# ════════════════════════════════════════════════════════════════════

# Bailian API
BAILIAN_API_KEY = "sk-fd02acce73634ecbbab21757c1cada10"
BAILIAN_EMBEDDING_URL = (
    "https://dashscope.aliyuncs.com/api/v1/services/"
    "embeddings/multimodal-embedding/multimodal-embedding"
)
EMBEDDING_MODEL = "tongyi-embedding-vision-plus-2026-03-06"
EMBEDDING_DIM = 512

# Upstash Vector
UPSTASH_URL = "https://lasting-raven-85217-us1-vector.upstash.io"
UPSTASH_TOKEN = "ABcFMGxhc3RpbmctcmF2ZW4tODUyMTctdXMxYWRtaW5aVFEzWXpSaVltTXROREkwTUMwMFkyRXlMV0l5TlRjdE5tWTNNakl5TTJGbE5qTXc="

FAVORITES_FILE = Path(__file__).parent / "favorites.json"

# ════════════════════════════════════════════════════════════════════
# Upstash Vector client
# ════════════════════════════════════════════════════════════════════
print("Connecting to Upstash Vector ...")
index = Index(url=UPSTASH_URL, token=UPSTASH_TOKEN)
info = index.info()
print(f"Connected. Index has {info.vector_count} vectors ({info.dimension}d, {info.similarity_function})")


# ════════════════════════════════════════════════════════════════════
# Embedding functions via Bailian API
# ════════════════════════════════════════════════════════════════════

def _call_bailian(contents: list[dict]) -> list[list[float]]:
    """Call Bailian multimodal embedding API with given contents."""
    data = {
        "model": EMBEDDING_MODEL,
        "input": {"contents": contents},
        "parameters": {"dimension": EMBEDDING_DIM},
    }
    headers = {
        "Authorization": f"Bearer {BAILIAN_API_KEY}",
        "Content-Type": "application/json",
    }
    resp = requests.post(BAILIAN_EMBEDDING_URL, headers=headers, json=data, timeout=60)
    resp.raise_for_status()
    result = resp.json()
    return [item["embedding"] for item in result["output"]["embeddings"]]


def text_to_embedding(text: str) -> list[float]:
    """Convert text query to embedding via Bailian API."""
    embs = _call_bailian([{"text": text}])
    return embs[0]


def image_to_embedding(image) -> list[float]:
    """Convert PIL image to embedding via Bailian API."""
    if image is None:
        return None
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    image = image.convert("RGB")

    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    embs = _call_bailian([{"image": f"data:image/jpeg;base64,{img_b64}"}])
    return embs[0]


# ── Favorites persistence ──────────────────────────────────────────

def load_favorites():
    if FAVORITES_FILE.exists():
        return json.loads(FAVORITES_FILE.read_text())
    return []


def save_favorites(favs):
    FAVORITES_FILE.write_text(json.dumps(favs, indent=2, ensure_ascii=False))


# ════════════════════════════════════════════════════════════════════
# Search logic
# ════════════════════════════════════════════════════════════════════

# Global state for current result paths (for favorites)
_current_result_paths = []


def do_search(text_query, image_query, top_k):
    """Perform search based on text and/or image query."""
    global _current_result_paths

    emb = None
    search_mode = ""

    if text_query and text_query.strip():
        emb = text_to_embedding(text_query.strip())
        search_mode = f'text: "{text_query.strip()}"'

    if image_query is not None:
        img_emb = image_to_embedding(image_query)
        if emb is not None:
            # Combine text + image embeddings (equal weight average)
            emb = [(a + b) / 2 for a, b in zip(emb, img_emb)]
            # Re-normalize
            norm = sum(x ** 2 for x in emb) ** 0.5
            emb = [x / norm for x in emb]
            search_mode += " + image"
        else:
            emb = img_emb
            search_mode = "image query"

    if emb is None:
        return (
            [],
            "Please enter a text description or upload an image to search.",
            "",
        )

    results = index.query(vector=emb, top_k=top_k, include_metadata=True)

    gallery_items = []
    _current_result_paths = []
    for r in results:
        meta = r.metadata
        img_path = meta.get("path", "")
        score = r.score
        coarse = meta.get("coarse_class", "")
        fine = meta.get("fine_class", "")
        caption = f"{fine} ({coarse}) \u2014 {score:.3f}"
        gallery_items.append((img_path, caption))
        _current_result_paths.append(img_path)

    summary = f"Found **{len(results)}** results (mode: {search_mode}, top-{top_k})"
    return gallery_items, summary, search_mode


def add_selected_to_favorites(evt: gr.SelectData):
    """Add clicked image to favorites."""
    global _current_result_paths
    if evt.index is not None and evt.index < len(_current_result_paths):
        path = _current_result_paths[evt.index]
        favs = load_favorites()
        if not any(f["path"] == path for f in favs):
            favs.append({"path": path, "added_at": datetime.datetime.now().isoformat()})
            save_favorites(favs)
        fav_count = len(favs)
        fav_summary = f"You have **{fav_count}** favorite(s).\n\n"
        fav_summary += "\n".join(f"- `{f['path']}`" for f in favs[-20:])
        return fav_summary
    return "No image selected."


def add_all_to_favorites():
    """Add all current results to favorites."""
    global _current_result_paths
    favs = load_favorites()
    added = 0
    for path in _current_result_paths:
        if not any(f["path"] == path for f in favs):
            favs.append({"path": path, "added_at": datetime.datetime.now().isoformat()})
            added += 1
    save_favorites(favs)
    return f"Added {added} new image(s). Total favorites: **{len(favs)}**"


def download_favorites():
    """Return the favorites file for download."""
    if FAVORITES_FILE.exists():
        return str(FAVORITES_FILE)
    return None


def clear_favorites():
    """Clear all favorites."""
    save_favorites([])
    return "Favorites cleared."


# ════════════════════════════════════════════════════════════════════
# Gradio UI — Five-Stage Search Framework
# ════════════════════════════════════════════════════════════════════

with gr.Blocks(title="Image Search - Five-Stage Framework") as demo:

    # ── Header ─────────────────────────────────────────────────────
    gr.Markdown(
        """
        # Grocery Store Image Search
        Search grocery product images using **text descriptions** or **reference images**.
        Built with Bailian Multimodal Embedding + Upstash Vector, following the **Five-Stage Search Framework**.
        """
    )

    # ── Stage 1: Formulation ───────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Formulation \u2014 Enter Your Query")
            text_input = gr.Textbox(
                label="Text Query",
                placeholder='e.g. "a red apple", "fresh banana", "milk carton"',
                lines=2,
            )
            image_input = gr.Image(
                label="Image Query (upload a reference image)",
                type="pil",
                height=200,
            )

        with gr.Column(scale=1):
            gr.Markdown("### Query Preview")
            preview_text = gr.Textbox(
                label="Current Text Query",
                interactive=False,
                lines=2,
            )
            preview_image = gr.Image(
                label="Current Image Query",
                type="pil",
                interactive=False,
                height=200,
            )

    # Live preview updates (Formulation stage)
    text_input.change(fn=lambda x: x, inputs=text_input, outputs=preview_text)
    image_input.change(fn=lambda x: x, inputs=image_input, outputs=preview_image)

    # ── Stage 2: Initiation ────────────────────────────────────────
    gr.Markdown("---")
    with gr.Row():
        search_btn = gr.Button("\U0001f50d Search", variant="primary", size="lg")
        clear_btn = gr.Button("\U0001f5d1\ufe0f Clear", variant="secondary", size="lg")

    # ── Stage 4: Refinement ────────────────────────────────────────
    with gr.Row():
        top_k_slider = gr.Slider(
            minimum=4,
            maximum=50,
            value=12,
            step=1,
            label="Number of Results (Top-K) \u2014 Refinement",
            info="Adjust to show more or fewer results",
        )

    # ── Stage 3: Review ────────────────────────────────────────────
    gr.Markdown("---")
    gr.Markdown("### Search Results")
    result_summary = gr.Markdown(
        "No search performed yet. Enter a query and click **Search**."
    )

    # ── Results Gallery ────────────────────────────────────────────
    gallery = gr.Gallery(
        label="Results (click any image to add to favorites)",
        show_label=True,
        columns=4,
        height="auto",
        object_fit="contain",
        allow_preview=True,
    )

    # ── Stage 5: Use ──────────────────────────────────────────────
    gr.Markdown("---")
    gr.Markdown("### Use \u2014 Actions on Results")
    with gr.Row():
        fav_info = gr.Markdown(
            "Click on any image in the gallery to add it to favorites."
        )
        add_all_btn = gr.Button("\u2b50 Add All to Favorites")
    with gr.Row():
        download_btn = gr.Button("\U0001f4e5 Download Favorites File")
        clear_fav_btn = gr.Button("\U0001f5d1\ufe0f Clear All Favorites")

    fav_display = gr.Markdown("")
    fav_file = gr.File(label="Downloaded Favorites", visible=False)

    # ── Event wiring ──────────────────────────────────────────────
    search_outputs = [gallery, result_summary, preview_text]

    search_btn.click(
        fn=do_search,
        inputs=[text_input, image_input, top_k_slider],
        outputs=search_outputs,
    )

    # Click on gallery image -> add to favorites
    gallery.select(
        fn=add_selected_to_favorites,
        outputs=fav_display,
    )

    add_all_btn.click(fn=add_all_to_favorites, outputs=fav_display)
    download_btn.click(fn=download_favorites, outputs=fav_file)
    clear_fav_btn.click(fn=clear_favorites, outputs=fav_display)

    clear_btn.click(
        fn=lambda: (
            "",
            None,
            "",
            None,
            [],
            "Query cleared.",
            "",
        ),
        outputs=[
            text_input,
            image_input,
            preview_text,
            preview_image,
            gallery,
            result_summary,
            fav_display,
        ],
    )


# ════════════════════════════════════════════════════════════════════
# Launch
# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
    )
