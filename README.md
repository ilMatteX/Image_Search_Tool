# 🖼️ Image Search Tool

**Image Search Tool** is a desktop application built with Python and PyQt5 that allows you to **automatically find visually similar images** from a folder based on a reference image. It uses **deep learning and feature extraction** with TensorFlow to index and compare images efficiently.

This tool is especially useful for:
- Interior designers or 3D artists managing large image libraries
- Organizing texture or asset collections
- Finding duplicates or near-duplicate images

## ⚙️ Requirements

This app has been tested on **Python 3.12**. Make sure you install the following dependencies:

```bash
pip install pyqt5 tensorflow opencv-python scikit-learn numpy pillow tqdm
```

## 📁 Project Structure

- `feature_indexer.py`: indexes all images in a given folder by extracting features with a neural network and saves them for later use.
- `search_similar.py`: loads the index and compares a given image to find the most similar images in the dataset.
- `image_search_tool.py`: **Graphical User Interface (GUI)** that combines indexing, image selection, and search — all in one easy-to-use window. This is now the preferred way to use the tool.

## 🚀 How to Use

### ✅ Recommended: Using the GUI (no code required)

Run the GUI:
```bash
python image_search_tool.py
```

From the GUI, you can:
1. **Select a folder** of images to index.
2. Click **Start Indexing** — a progress bar shows the indexing status.
3. Select an image to search.
4. Set how many results you want (e.g., top 10).
5. Click **Search** to view large previews of the most similar images.
6. Double-click a result to open the image location directly in your file explorer.

### ⚙️ Advanced: Using the scripts via command line

#### Index images
```bash
python feature_indexer.py --folder "path/to/your/image/folder"
```

#### Search for similar images
```bash
python search_similar.py --image "path/to/query/image.jpg" --top 10
```
- `--top`: defines how many similar images to retrieve, starting from the most similar based on visual similarity score (e.g., `--top 10` shows the 10 best matches).

## 🧠 How It Works

- The tool uses a **pre-trained TensorFlow model** to extract visual features (embeddings) from images.
- These features are stored using NumPy and compared using **cosine similarity** from scikit-learn.
- GUI displays high-resolution image thumbnails and includes direct access to the files.

## 🖥️ Screenshot

![App Screenshot](./screenshot.png) <!-- Optional: Add a screenshot if you like -->

---

Made with ❤️ using Python and open-source libraries.
