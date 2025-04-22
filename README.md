# 🧠 Mattex Image Search Tool

A smart desktop tool for **image similarity search**, built with **PyQt5**, **MobileNetV2**, and **CLIP**.

This project offers **two powerful versions** to help artists, designers, and developers manage and search large image collections — whether by visual similarity or natural language.

---

## 🎬 Demo
➡️ [Stop Wasting Time Searching Images – Try This Python Tool!](https://youtu.be/UU976b6hUrY?si=r2ilvNG1ahBkpYMp)

---

## 📦 Available Versions

### 🔹 [`v1-MobileNet`](./v1-MobileNet)

Image similarity search based on **visual features** using **MobileNetV2** (CNN):

- Extracts image embeddings with TensorFlow
- Compares images using **cosine similarity**
- Fast and lightweight
- Ideal for 3D artists, interior designers, or texture managers

➡️ [View README for v1 →](./v1-MobileNet/README.md)

---

### 🔹 [`v2-Clip-Integration`](./v2-Clip-Integration)

Experimental version with **text-based search** via **OpenAI CLIP**:

- All features of v1 included
- Search for images using **text queries**, e.g. `"modern black chair"` or `"sunset over mountain"`
- Combines both **visual** and **textual** search in one interface

➡️ [View README for v2 →](./v2-Clip-Integration/README.md)

---

## ⚖️ Which Version Should You Use?

| Goal                                 | Recommended Version     |
|--------------------------------------|--------------------------|
| Fast, reliable visual similarity search | `v1-MobileNet`           |
| Search by **text description**         | `v2-Clip-Integration`    |
| Both visual and text-based search      | `v2-Clip-Integration`    |

---

## 🖥️ GUI Features (Both Versions)

- 📁 Folder-based image indexing (category + subcategory support)
- 🧠 Feature extraction using pretrained models
- 🔍 Cosine similarity search with ranking
- 🖼️ Preview of results with:
  - Thumbnail
  - Similarity %
  - Category / subcategory
  - File path and quick open
- ♻️ Smart incremental indexing (processes only new images)
- 💾 Backup system for each indexing session

---

## 🛠️ Requirements

Each version includes its own `requirements.txt` and `README.md`.  
Make sure to follow the instructions inside the folder you choose to run.

---

## 🔭 Coming Soon(?)

- Unified version with a switch between CLIP & MobileNet
- Better GPU/multithread indexing performance
- Export search results as PDF or HTML
- Cleaner UI and drag & drop improvements

---

## 📄 License

Released under the **MIT License**.  
© 2025 Mattex — Feel free to use, modify, and share with credit. ✌️
