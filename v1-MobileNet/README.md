# Image Similarity Search App

This application allows you to index images from folders and then search for similar images by dragging or uploading a new one. It uses MobileNetV2 for feature extraction.

## Features

- Image indexing by category and subcategory.
- Drag & Drop or button-based image input.
- Real-time similarity search with preview and similarity percentage.
- Multilingual support (English ready).

## How to Use

1. Run the app (`main.py`).
2. Set the **Categories Folder** and **Images Folder**.
3. Click **Start** to index all images (or load existing index files).
4. Drag an image or use the **Upload Image** button to search for similar images.
5. The most similar images will appear with previews and similarity scores.

## Files

- `main.py`: Main GUI interface (PyQt5).
- `indexer.py`: Handles image indexing with MobileNetV2.
- `searcher.py`: Computes similarity and retrieves results.
- `utils.py`: Utility functions for image handling and preprocessing.

## Dependencies

- Python 3.13
- PyQt5
- TensorFlow
- NumPy
- scikit-learn
- Pillow

Install them with:

```bash
pip install -r requirements.txt
```

## License

MIT License
