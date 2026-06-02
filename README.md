# Image Search Interface — Five-Stage Search Framework

An image search web application built with **Gradio + Bailian Multimodal Embedding + Upstash Vector**, following the Five-Stage Search Framework. The dataset used is the [GroceryStoreDataset](https://github.com/marcusklasson/GroceryStoreDataset).

> **Fully cloud-based**: Embedding model (Bailian API) and vector database (Upstash Vector) are both cloud services. No local model deployment or database installation required.

## Quick Start

```bash
# 1. Create conda environment
conda create -n genai-env python=3.12 -y
conda activate genai-env

# 2. Install dependencies
pip install gradio upstash-vector requests pillow

# 3. Clone the dataset into project root
git clone https://github.com/marcusklasson/GroceryStoreDataset.git dataset

# 4. Build vector index (run once)
python build_index.py

# 5. Launch the app
python app.py
```

Open http://127.0.0.1:7861 in your browser.

## Tech Stack

| Component | Technology | Cloud/Local |
|-----------|-----------|-------------|
| Embedding Model | Bailian `tongyi-embedding-vision-plus-2026-03-06` (512-dim) | Cloud API |
| Vector Database | Upstash Vector (512-dim, Cosine similarity) | Cloud Service |
| Frontend | Gradio | Local |
| Dataset | GroceryStoreDataset (5,125 images, 43 classes) | Local files |

## Features (Five-Stage Search Framework)

| Stage | Feature | Description |
|-------|---------|-------------|
| **Formulation** | Text input + Image upload | Users can enter text descriptions or upload reference images |
| **Formulation** | Query preview | Live preview of the current text/image query before searching |
| **Initiation** | Search button | Explicit "Search" button to trigger the query |
| **Review** | Result summary | Shows total number of results and search mode |
| **Refinement** | Top-K slider | Adjustable slider to control the number of displayed results (4–50) |
| **Use** | Favorites & Download | Click images to add to favorites; download favorites as JSON file |

## Project Structure

```
HCI/
├── app.py              # Gradio web interface
├── build_index.py      # Data ingestion & embedding extraction
├── README.md           # This file
├── .gitignore
├── favorites.json      # Saved favorites (auto-generated, gitignored)
└── dataset/            # GroceryStoreDataset (cloned separately, gitignored)
    └── dataset/
        ├── train/      # 2,640 training images
        ├── test/       # 2,485 test images
        └── classes.csv # 43 fine-grained classes
```
