# 🔹 Mattex Image Search — CLIP Integrated Version

This version integrates **OpenAI CLIP** with **MobileNetV2** for **both visual and text-based** image retrieval.

---

## 🚀 Main Features

- **Dual Search Modes**:
  - 🔍 **Visual Search**: Find similar images using MobileNetV2 features.
  - ✍️ **Text Search**: Use natural language queries powered by CLIP.

- **Incremental Indexing**:
  - Processes only new images in two stages: MobileNet (0–50%) and CLIP (50–100%).
  - Creates and maintains separate index files: `index.npy` & `paths.txt`, `clip_index.npy` & `clip_paths.txt`.

- **Intuitive GUI (PyQt5)**:
  - Folder selection for indexing.
  - Progress bar reflecting both indexing stages.
  - Buttons for image selection and text query.
  - Category/subcategory filters.
  - Rich result items with thumbnail, score, and open-location button.

- **Automatic Backups**:
  - Saves previous indices when changes are detected in `/backup`.

---

## 📂 Folder Structure

```
v2-Clip-Integration/
├── main.py        # Main script with combined CLIP & MobileNet logic
├── index.npy            # (auto) MobileNet feature vectors
├── paths.txt            # (auto) MobileNet image paths
├── clip_index.npy       # (auto) CLIP feature vectors
├── clip_paths.txt       # (auto) CLIP image paths
├── backup/              # (auto) Timestamped backups of all index files
└── requirements.txt     # Python dependencies
```

---

## 🔧 Installation

1. Create a Python environment (tested on 3.12)
2. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📖 Workflow

1. **Indexing**  
   - Click **“Select images folder”** and choose your dataset root.  
   - Click **“Start Indexing”** to build/update both MobileNet and CLIP indices.  

2. **Visual Search**  
   - Click **“Select image to search”**, choose an image, then click **“Search similar images”**.  
   - Results display top matches sorted by cosine similarity (MobileNet).

3. **Text Search**  
   - Type a descriptive query in the text field (e.g., “red sports car on the road”).  
   - Click **“Search by text”** to retrieve images matching the description (CLIP).

4. **Filtering & Results**  
   - Use **Category** and **Subcategory** dropdowns to narrow down results.  
   - Adjust **Results** count with the spinner.  
   - Click on thumbnails or **“Open image location”** for quick access.

---

## ⚙️ Under the Hood

- **MobileNetV2**: Extracts 1280-dimensional image vectors.
- **CLIP ViT-B/32**: Produces joint image-text embeddings.
- **Cosine similarity**: Measures closeness in feature space.
- **Progress bar**: Combines MobileNet (0–50%) and CLIP (50–100%) phases.

---

## 📝 Notes

- Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`.
- First run downloads CLIP model weights (~300 MB).
- GPU acceleration via CUDA for PyTorch speeds up CLIP indexing.

---

## 📄 License

MIT License — © 2025 Mattex  
Feel free to use, modify, and share with credit.
