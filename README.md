# Image Search Interface — Five-Stage Search Framework

An image search web application built with **Gradio + Bailian Multimodal Embedding + Upstash Vector**, following the Five-Stage Search Framework. The dataset used is the [GroceryStoreDataset](https://github.com/marcusklasson/GroceryStoreDataset).

## Requirements

- Python 3.10+
- Conda environment `genai-env`

## Setup

### 1. Create and activate the environment

```bash
conda create -n genai-env python=3.12 -y
conda activate genai-env
```

### 2. Install dependencies

```bash
pip install gradio upstash-vector requests pillow openai
```

### 3. Clone the dataset

```bash
git clone https://github.com/marcusklasson/GroceryStoreDataset.git dataset
```

The `dataset/` folder should be at the same level as `app.py` and `build_index.py`.

## Usage

### Step 1: Build the Vector Index

This step extracts image embeddings from all dataset images using the Bailian multimodal embedding API (`tongyi-embedding-vision-plus-2026-03-06`, 512-dim) and uploads them to the Upstash Vector index. **Run this once** before starting the app.

```bash
conda activate genai-env
python build_index.py
```

This will:
- Scan all images in `dataset/dataset/train/` and `dataset/dataset/test/` (5,502 images total)
- Extract 512-dim image embeddings via Bailian API (batched, 10 images per API call)
- Upload vectors to Upstash Vector in batches

> Note: This step requires internet access to reach the Bailian API and Upstash API.

### Step 2: Launch the Gradio App

```bash
conda activate genai-env
python app.py
```

Then open your browser and navigate to:

```
http://127.0.0.1:7860
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Embedding Model | Bailian `tongyi-embedding-vision-plus-2026-03-06` (512-dim) |
| Vector Database | Upstash Vector (512-dim, Cosine similarity) |
| Frontend | Gradio |
| Dataset | GroceryStoreDataset (5,502 images, 43 classes) |

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
├── favorites.json      # Saved favorites (auto-generated)
└── dataset/            # GroceryStoreDataset (cloned from GitHub)
    └── dataset/
        ├── train/
        ├── test/
        └── classes.csv
```
