---
title: "Image Search Interface Design Based on the Five-Stage Search Framework"
author: "Student Name, Student ID"
date: "June 2026"
course: "Human-Computer Interaction (HCI)"
---

# 1. Dataset Description

## 1.1 Source

The dataset used in this project is the **GroceryStoreDataset** [1], an open-source dataset published on GitHub by Klasson et al. (https://github.com/marcusklasson/GroceryStoreDataset). It contains natural images of grocery store products captured under varied lighting, angles, and backgrounds, making it well-suited for image retrieval and classification experiments.

## 1.2 Scale

The dataset contains a total of **5,125 images** split into two subsets:

- **Training set**: 2,640 images
- **Test set**: 2,485 images

All images are in JPEG format with varying resolutions. The dataset also provides 81 additional iconic (representative) images and textual product descriptions, though these were not used in the indexing phase.

## 1.3 Content

The images are organized into a two-level hierarchy of **3 coarse-grained categories** and **43 fine-grained classes**:

| Coarse Category | Fine-Grained Classes | Example Classes |
|----------------|---------------------|-----------------|
| Fruit (19 classes) | Multiple varieties per fruit | Apple (Golden-Delicious, Granny-Smith, Pink-Lady, Red-Delicious, Royal-Gala), Banana, Orange, Mango, etc. |
| Vegetables (15 classes) | Single variety per vegetable | Asparagus, Cabbage, Carrots, Cucumber, Garlic, Onion, Potato, Tomato, etc. |
| Packages (9 classes) | Packaged products | Juice, Milk, Yoghurt, Oat-Milk, Soyghurt, Sour-Cream, etc. |

The directory structure follows the pattern: `split/coarse_class/fine_class/variety/filename.jpg` for fruits with multiple varieties, and `split/coarse_class/fine_class/filename.jpg` for vegetables and packaged goods.

Each image in the dataset represents a single grocery product photographed in a realistic store or kitchen environment. This diversity in visual conditions makes the dataset a good benchmark for evaluating the robustness of multimodal embedding-based retrieval systems.

---

# 2. Five-Stage Search Framework Implementation

The Five-Stage Search Framework, proposed by Marchionini and Shneiderman, defines the user's cognitive and interactive journey through a search process. Our interface implements all five stages with explicit UI components in the Gradio web application.

## 2.1 Formulation (Expression)

**Requirement**: Provide input mechanisms for the user to express their information need, and allow them to preview their query before submission.

**Implementation**: The interface provides two parallel input modalities placed side by side (Figure 1, top section):

- **Text input** (`gr.Textbox`): A multi-line text field where users can type natural language descriptions such as "a red apple" or "fresh banana".
- **Image upload** (`gr.Image`): A drag-and-drop area where users can upload a reference image to use as a visual query.

**Query Preview**: To the right of the input area, a real-time preview panel mirrors the user's current query. As the user types text or uploads an image, the corresponding preview field updates instantly via Gradio's reactive `change` event handlers. This allows users to verify their query before initiating a search, reducing errors and increasing confidence.

## 2.2 Initiation (Start Search)

**Requirement**: Provide a clear, explicit action to trigger the search process.

**Implementation**: A prominent **"Search" button** (`gr.Button` with `variant="primary"`) is placed below the input area. The search is not triggered automatically on input change — the user must deliberately click the button. This design choice gives users full control over when to execute a query, preventing premature or accidental searches while the user is still formulating their query.

A secondary **"Clear" button** is also provided to reset all inputs and results.

## 2.3 Review (Examine Results)

**Requirement**: Provide an overview of search results so users can assess the scope and relevance of what was found.

**Implementation**: After a search is executed, a **result summary** (`gr.Markdown`) is displayed above the gallery, showing:

- The total number of results returned
- The search mode used (text, image, or combined)
- The Top-K parameter value

For example: *"Found **12** results (mode: text: \"red apple\", top-12)"*.

The search results are displayed in a **gallery grid** (`gr.Gallery`, 4 columns) where each image is shown with a caption containing the product class name and the similarity score (e.g., "Apple (Fruit) — 0.603"). This allows users to quickly scan results and assess whether the retrieved items match their intent.

## 2.4 Refinement (Adjust Parameters)

**Requirement**: Allow users to modify search parameters based on their review of the results.

**Implementation**: A **Top-K slider** (`gr.Slider`, range 4–50, default 12) is placed between the search button and the results area. Users can adjust the slider to increase or decrease the number of returned images, then click Search again to obtain a refined result set.

This supports the iterative nature of search: a user might first request 12 results, find them too narrow, increase the slider to 30, and search again for broader coverage.

## 2.5 Use (Apply Results)

**Requirement**: Enable users to act on the search results — selecting, saving, or exporting items of interest.

**Implementation**: The interface provides multiple interaction mechanisms:

- **Single-image favorite**: Clicking on any image in the gallery adds it to a favorites list. This leverages Gradio's `gallery.select` event, which provides the index of the clicked image.
- **Batch favorite**: An "Add All to Favorites" button saves all currently displayed results at once.
- **Download**: A "Download Favorites File" button exports the favorites list as a JSON file, which users can save locally.
- **Clear favorites**: A "Clear All Favorites" button resets the favorites list.

The favorites section displays a summary of all saved items with their file paths, providing persistent visibility into the user's collection.

---

# 3. Impact of Input Modalities on User Workflow

## 3.1 Text-Based Search

**Workflow**: The user reads the interface label, types a descriptive query (e.g., "green vegetable", "milk carton"), and clicks Search. The text is sent to the Bailian multimodal embedding API, which returns a 512-dimensional vector. This vector is compared against all indexed image vectors in Upstash Vector using cosine similarity.

**Advantages**:
- Low barrier to entry — users only need a keyboard.
- Supports abstract or conceptual queries (e.g., "something sweet", "healthy snack") that may not correspond to a specific visual reference.
- Allows refinement through natural language — users can iteratively adjust their description.

**Challenges**:
- Requires the user to translate a visual mental image into words, which can be imprecise.
- Effectiveness depends on the user's vocabulary and the model's language understanding.
- Multilingual queries may produce inconsistent results depending on the embedding model's training data.

## 3.2 Image-Based Search

**Workflow**: The user uploads or drags an image into the upload area, previews it to confirm, and clicks Search. The uploaded image is converted to base64, sent to the Bailian API for embedding extraction, and the resulting vector is used for similarity search.

**Advantages**:
- Bypasses the language barrier — no need to describe visual features in words.
- Effective for finding visually similar items when the user has a concrete reference (e.g., "find more like this apple").
- Particularly useful for non-expert users who may not know the exact product name.

**Challenges**:
- Requires the user to have a reference image ready, which adds a step to the workflow.
- Upload time depends on image size and network conditions.
- The user may not have a suitable reference image available.

## 3.3 Combined Text + Image Search

Our interface also supports submitting both a text query and an image simultaneously. In this case, the text and image embeddings are averaged and re-normalized before searching. This combined modality allows users to refine visual similarity with semantic intent (e.g., uploading a fruit photo while typing "red" to bias toward red fruits).

## 3.4 Designing for Both Modalities

To make both input methods equally user-friendly, we adopted the following design principles:

1. **Unified input area**: Both the text box and image uploader are placed side-by-side in the same section, with equal visual prominence. Neither is privileged over the other, reducing bias toward one modality.

2. **Symmetric preview**: Each input has its own preview panel (text preview and image preview), so users can verify their query regardless of modality.

3. **Single search action**: The same "Search" button processes both input types. The system automatically detects which inputs are provided and handles text-only, image-only, or combined queries transparently.

4. **Consistent result format**: Regardless of the input modality, results are displayed in the same gallery format with the same metadata (class name, similarity score), ensuring a uniform experience.

5. **Progressive disclosure**: The Top-K slider and favorites controls are modality-agnostic, so the refinement and use stages remain consistent regardless of how the search was initiated.

---

# 4. Conclusion

This project demonstrates a fully cloud-based image search system that implements the Five-Stage Search Framework through a Gradio web interface. By using the Bailian multimodal embedding API for feature extraction and Upstash Vector for similarity search, the system achieves effective cross-modal retrieval (text-to-image and image-to-image) on the GroceryStoreDataset without requiring any local model deployment or database installation. The interface design prioritizes usability by providing dual input modalities, real-time query preview, adjustable result parameters, and interactive result management through a favorites system.

---

# References

[1] M. Klasson, C. Kjellström, and H. Kjellström, "A Hierarchical Grocery Store Image Dataset with Fine-Grained Product Families," 2019. GitHub: https://github.com/marcusklasson/GroceryStoreDataset
