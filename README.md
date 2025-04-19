# 🖼️ Image Search Tool with PyQt5 + MobileNetV2

This project is a desktop application built with **PyQt5** and **TensorFlow** (MobileNetV2) that allows you to:

- 📁 Index a folder containing images  
- 🧠 Automatically extract features using a pretrained CNN (MobileNetV2)  
- 🔍 Search for similar images by comparing features with cosine similarity  
- 🖼️ Visually browse results with image previews, similarity percentage, and category info  
- 📂 Open file locations directly from the interface  

This tool is especially useful for:
- Interior designers or 3D artists managing large image libraries
- Organizing texture or asset collections
- Finding duplicates or near-duplicate images
---

## 📦 Features

- ✅ Batch indexing of **new images only** (avoids duplicates)  
- ✅ **Backup system** for indexed files  
- ✅ **Category & subcategory extraction** based on folder structure  
- ✅ **Image similarity search** using cosine similarity  
- ✅ Search results include:
  - Thumbnail preview  
  - Similarity percentage  
  - Category and subcategory  
  - Full image path  
- ✅ Fast inference with **MobileNetV2**  
- ✅ Interactive **PyQt5 GUI**  

---

## 🛠️ Requirements

- This project was developed using Python 3.12

- Install dependencies via pip:

```bash
pip install numpy tensorflow scikit-learn PyQt5 tqdm
```

---

## 🚀 How to Run

```bash
python main.py
```

You'll see a GUI where you can:

- Select an image folder to index  
- Drag and drop or choose a query image to find similar ones  
- Browse results, open the matching image or its folder directly  

---

## 📂 Folder Structure

```
your_folder/
├── Category1/
│   ├── SubcategoryA/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   ├──  ... .png
│   ├── SubcategoryB/
│   │   ├── image3.png
│   │   ├── image4.png
│   │   ├──  ... .png
├── Category2/
│   ├── SubcategoryC/
│   │   ├── image5.png
│   │   ├── image6.png
│   │   ├──  ... .png
│   │   ├── imageN.png
│   ├── Subcategory.../
... (hope you get the idea)
```

Categories and subcategories are inferred automatically from the folder path.

---

## 💾 Indexing Behavior

- On each run, the app scans your chosen folder for images  
- It compares hashes of previously saved data to check for changes  
- New images are processed and features are saved in `index.npy`, paths in `paths.txt`  
- Previous versions are backed up in the `backup/` folder with timestamps

## ⚠️ Note: The backup/ folder can grow significantly in size over time, especially if the indexed dataset is large or frequently updated. It is recommended to periodically review and clean up old backups to save disk space.

---

## 📊 How It Works

### 🧠 Feature Extraction

- Uses **MobileNetV2** (no top layer), pre-trained on ImageNet  
- Images are resized to `224x224`, preprocessed, and passed to the model  
- Extracted feature vectors are **normalized** and saved  

### 🔍 Similarity Matching

- Uses **cosine similarity** between the query image and indexed feature vectors  
- Results are shown in descending order of similarity  

---

## 📎 Files Explained

| File        | Purpose                                      |
|-------------|----------------------------------------------|
| `index.npy` | Stores feature vectors of indexed images     |
| `paths.txt` | Stores image paths in order                  |
| `backup/`   | Contains backup versions with timestamps     |
| `main.py`   | Main application logic and GUI               |
| `README.md` | This documentation file                      |

---

## 🧪 Notes

- Supports `.jpg`, `.jpeg`, `.png`, `.bmp`, and `.webp` images  
- If an image cannot be processed, it's skipped with a warning in the console  
- Thumbnails are resized to **15% of screen width** for performance  

---

## 📌 TODO / Ideas for Expansion

- [ ] Add drag & drop support for query images  
- [ ] Export results as a report (HTML or PDF)  
- [ ] Add multi-GPU or threading support for faster indexing  
- [ ] Improve UI

---

## 📄 License

This project is released under the **MIT License**.

Copyright (c) 2025 Mattex

Feel free to use, modify, and distribute — just give credit where it's due! :)
