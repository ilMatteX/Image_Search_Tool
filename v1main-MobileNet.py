import sys
import os
import shutil
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import subprocess  # Required to open the file explorer

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QListWidget, QListWidgetItem, QMessageBox, QProgressBar, QSpinBox, QHBoxLayout, QComboBox
)
from PyQt5.QtGui import QPixmap, QIcon, QDesktopServices, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QUrl

import hashlib
import json

# ----------------------------
# CONFIGURATION AND CORE FUNCTIONS
# ----------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_FILE = os.path.join(BASE_DIR, "index.npy")
PATHS_FILE = os.path.join(BASE_DIR, "paths.txt")
BACKUP_FOLDER = os.path.join(BASE_DIR, "backup")

BATCH_SIZE = 128
IMAGE_SIZE = (224, 224)

model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

def compute_file_hash(path):
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

def backup_existing_files():
    os.makedirs(BACKUP_FOLDER, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Compute current file hash
    current_hashes = {
        'index': compute_file_hash(FEATURES_FILE),
        'paths': compute_file_hash(PATHS_FILE)
    }

    # Saved file hash path
    hash_file = os.path.join(BACKUP_FOLDER, "last_backup_hashes.json")

    # Load previous hash
    if os.path.exists(hash_file):
        with open(hash_file, "r") as f:
            last_hashes = json.load(f)
    else:
        last_hashes = {}

    # Compare and Backup only if files have changed
    if current_hashes != last_hashes:
        print("[+] Changes detected, performing backup...")
        if os.path.exists(FEATURES_FILE):
            shutil.copy(FEATURES_FILE, os.path.join(BACKUP_FOLDER, f"index_{timestamp}.npy"))
        if os.path.exists(PATHS_FILE):
            shutil.copy(PATHS_FILE, os.path.join(BACKUP_FOLDER, f"paths_{timestamp}.txt"))
        
        # Save new hash
        with open(hash_file, "w") as f:
            json.dump(current_hashes, f)
    else:
        print("[=] No changes detected. Backup not required.")

def load_existing_data():
    if os.path.exists(FEATURES_FILE) and os.path.exists(PATHS_FILE):
        try:
            features = np.load(FEATURES_FILE)
            with open(PATHS_FILE, 'r', encoding='utf-8') as f:
                paths = [os.path.normpath(line.strip()) for line in f.readlines()]
            return features, set(paths)
        except Exception as e:
            print(f"[!] Error loading existing data: {e}")
    return None, set()

def save_data(features, paths):
    np.save(FEATURES_FILE, features)
    with open(PATHS_FILE, 'w', encoding='utf-8') as f:
        for p in paths:
            f.write(p + '\n')

def list_all_images(folder):
    valid_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    folder = os.path.abspath(folder)
    images = []
    for dp, _, filenames in os.walk(folder):
        for f in filenames:
            ext = os.path.splitext(f.lower())[1]
            if ext in valid_exts:
                full_path = os.path.join(dp, f)
                images.append(os.path.normpath(full_path))
    return images

def get_category_and_subcategory(image_path, base_folder):
    rel_path = os.path.relpath(image_path, base_folder)
    parts = rel_path.split(os.sep)
    category = parts[0] if len(parts) > 0 else "N/A"
    subcategory = parts[1] if len(parts) > 1 else "N/A"
    return category, subcategory

def preprocess_images_batch(image_paths):
    batch = []
    valid_paths = []
    for path in image_paths:
        try:
            img = keras_image.load_img(path, target_size=IMAGE_SIZE)
            arr = keras_image.img_to_array(img)
            arr = preprocess_input(arr)
            batch.append(arr)
            valid_paths.append(path)
        except Exception as e:
            print(f"[!] Error processing image: {path} — {e}")
    return np.array(batch), valid_paths

def extract_features_batch(image_paths, progress_callback=None):
    features = []
    valid_paths = []
    total_batches = (len(image_paths) + BATCH_SIZE - 1) // BATCH_SIZE
    for batch_index, i in enumerate(range(0, len(image_paths), BATCH_SIZE)):
        batch_paths = image_paths[i:i+BATCH_SIZE]
        batch_imgs, batch_valid_paths = preprocess_images_batch(batch_paths)
        if len(batch_imgs) == 0:
            continue
        preds = model.predict(batch_imgs, verbose=0)
        features.append(preds)
        valid_paths.extend(batch_valid_paths)
        if progress_callback is not None:
            progress = int(((batch_index+1) / total_batches) * 100)
            progress_callback(progress)
    if features:
        features = np.vstack(features)
        features = normalize(features)
    else:
        features = np.array([])
    return features, valid_paths

def index_images(folder, progress_callback=None):
    start_time = time.time()
    print("[✓] Scanning images...")
    all_images = list_all_images(folder)
    print(f"[✓] Found {len(all_images)} total images")
    backup_existing_files()
    existing_features, existing_paths = load_existing_data()
    new_images = [p for p in all_images if p not in existing_paths]
    if not new_images:
        print("[✓] No new images to index.")
        return False
    print(f"[+] New images to index: {len(new_images)}")
    new_features, new_paths = extract_features_batch(new_images, progress_callback)
    if existing_features is not None and len(existing_features) > 0:
        combined_features = np.vstack([existing_features, new_features])
        combined_paths = list(existing_paths) + new_paths
    else:
        combined_features = new_features
        combined_paths = new_paths
    save_data(combined_features, combined_paths)
    elapsed = time.time() - start_time
    print(f"[✓] Indexing completed in {elapsed:.2f} seconds.")
    print(f"[✓] Total images indexed: {len(combined_paths)}")
    return True

def extract_feature(image_path, model):
    try:
        img = keras_image.load_img(image_path, target_size=IMAGE_SIZE)
        img = keras_image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        features = model.predict(img, verbose=0)
        return features[0]
    except Exception as e:
        print(f"[!] Error reading {image_path}: {e}")
        return None

def get_similar_images(input_image_path, top_k):
    features = np.load(FEATURES_FILE)
    with open(PATHS_FILE, "r", encoding="utf-8") as f:
        paths = [line.strip() for line in f.readlines()]
    input_feature = extract_feature(input_image_path, model)
    if input_feature is None:
        return []
    similarities = cosine_similarity([input_feature], features)[0]
    sorted_indices = np.argsort(similarities)[::-1]
    results = []
    for i in range(min(100, len(sorted_indices))):
        idx = sorted_indices[i]
        path = paths[idx]
        score = similarities[idx] * 100
        results.append((path, score))
    return results

# ----------------------------
# WORKER THREAD FOR INDEXING
# ----------------------------
class IndexWorker(QThread):
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(bool)

    def __init__(self, folder):
        super().__init__()
        self.folder = folder

    def run(self):
        result = index_images(self.folder, progress_callback=self.progress_signal.emit)
        self.finished_signal.emit(result)

# ----------------------------
# CLASS TO RENDER A CLICKABLE LABEL
# ----------------------------
class ClickableLabel(QLabel):
    clicked = pyqtSignal()
    
    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)

# ----------------------------
# CUSTOM WIDGET FOR SEARCH RESULTS
# ----------------------------
class ResultItemWidget(QWidget):
    def __init__(self, image_path, score, category, subcategory):
        super().__init__()
        self.image_path = image_path
        self.score = score
        self.category = category
        self.subcategory = subcategory
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Use of ClickableLabel for the thumbnail
        self.image_label = ClickableLabel()
        #Loads the image and resizes it to 15% of the screen width, keeping the aspect ratio and using smooth scaling for better quality.
        screen = QApplication.primaryScreen()
        screen_size = screen.size()
        screen_width = screen_size.width()

        # Use 15% of the screen_width for the image
        max_image_width = int(screen_width * 0.15)

        pixmap = QPixmap(self.image_path).scaled(
            max_image_width, max_image_width,
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        #####
        self.image_label.setPixmap(pixmap)
        self.image_label.clicked.connect(self.open_image)
        layout.addWidget(self.image_label)
        
        # Info: filename, similarity score, category & subcategory
        info_layout = QVBoxLayout()
        info_text = f"{os.path.basename(self.image_path)}<br>"
        info_text += f"{self.score:.2f}%<br>"
        info_text += f"{self.category} - {self.subcategory}<br>"
        self.text_label = QLabel(info_text)
        self.text_label.setAlignment(Qt.AlignCenter)
        #self.text_label.setTextFormat(Qt.RichText)
        info_layout.addWidget(self.text_label)

        layout.addLayout(info_layout)
        
        # Button to open the image location
        self.open_path_btn = QPushButton("Open image location")
        self.open_path_btn.clicked.connect(self.open_path)
        layout.addWidget(self.open_path_btn)
        
        self.setLayout(layout)
    
    def open_image(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(self.image_path))
    
    def open_path(self):
        try:
            subprocess.run(["explorer", "/select,", self.image_path])
        except Exception as e:
            print("Error opening path:", e)

# ----------------------------
# GUI WITH PyQT5
# ----------------------------
class ImageSearchApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Search Tool")
        self.image_folder = None
        self.query_image_path = None
        self.index_worker = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Indexing section
        self.select_folder_btn = QPushButton("1. Select images folder to index")
        self.select_folder_btn.clicked.connect(self.select_folder)
        layout.addWidget(self.select_folder_btn)

        self.index_btn = QPushButton("2. Start Indexing")
        self.index_btn.clicked.connect(self.run_indexing)
        layout.addWidget(self.index_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        layout.addWidget(QLabel(""))

        # Search section
        self.select_image_btn = QPushButton("3. Select image to search")
        self.select_image_btn.clicked.connect(self.select_image)
        layout.addWidget(self.select_image_btn)

        self.query_preview = QLabel("Selected image preview")
        self.query_preview.setAlignment(Qt.AlignCenter)
        self.query_preview.setFixedHeight(400)
        layout.addWidget(self.query_preview)

        # Category filter
        filter_layout = QHBoxLayout()
        filter_label = QLabel("Filter by category:")
        self.category_combo = QComboBox()
        self.category_combo.addItem("All")
        filter_layout.addWidget(filter_label)
        filter_layout.addWidget(self.category_combo)
        layout.addLayout(filter_layout)

        # Subcategory filter
        subcategory_layout = QHBoxLayout()
        subcategory_label = QLabel("Filter by subcategory:")
        self.subcategory_combo = QComboBox()
        self.subcategory_combo.addItem("All")
        subcategory_layout.addWidget(subcategory_label)
        subcategory_layout.addWidget(self.subcategory_combo)
        layout.addLayout(subcategory_layout)

        # Number of results section
        top_k_layout = QHBoxLayout()
        top_k_label = QLabel("Number of results:")
        self.top_k_spinbox = QSpinBox()
        self.top_k_spinbox.setRange(1, 1000)
        self.top_k_spinbox.setValue(10)
        top_k_layout.addWidget(top_k_label)
        top_k_layout.addWidget(self.top_k_spinbox)
        layout.addLayout(top_k_layout)

        self.search_btn = QPushButton("4. Search similar images")
        self.search_btn.clicked.connect(self.run_search)
        layout.addWidget(self.search_btn)

        self.results_list = QListWidget()
        layout.addWidget(self.results_list)

        self.setLayout(layout)

        # Update subcategory options when category changes
        self.category_combo.currentIndexChanged.connect(self.update_subcategory_combo)

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select images folder")
        if folder:
            self.image_folder = folder
            QMessageBox.information(self, "Folder Selected", f"You selected:\n{folder}")
            # Refresh category combo
            self.category_combo.clear()
            self.category_combo.addItem("All")
            try:
                subfolders = [f for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f))]
                for sub in subfolders:
                    self.category_combo.addItem(sub)
            except Exception as e:
                print(f"Error listing subfolders: {e}")
            # Also update the subcategory
            self.update_subcategory_combo()

    def update_subcategory_combo(self):
        """Update subcategory options based on selected category."""
        self.subcategory_combo.clear()
        self.subcategory_combo.addItem("All")
        selected_category = self.category_combo.currentText()
        if selected_category != "All" and self.image_folder:
            category_path = os.path.join(self.image_folder, selected_category)
            try:
                subfolders = [f for f in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, f))]
                for sf in subfolders:
                    self.subcategory_combo.addItem(sf)
            except Exception as e:
                print("Error updating subcategories:", e)

    def run_indexing(self):
        if not self.image_folder:
            QMessageBox.warning(self, "Error", "Please select an images folder first")
            return
        self.index_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.index_worker = IndexWorker(self.image_folder)
        self.index_worker.progress_signal.connect(self.progress_bar.setValue)
        self.index_worker.finished_signal.connect(self.indexing_finished)
        self.index_worker.start()

    def indexing_finished(self, executed):
        self.index_btn.setEnabled(True)
        if executed:
            QMessageBox.information(self, "Indexing Completed", "Images have been indexed successfully.")
        else:
            QMessageBox.information(self, "Indexing", "No new images to index.")
        self.progress_bar.setValue(0)

    def select_image(self):
        # Open a file dialog to choose an image to search
        image_path, _ = QFileDialog.getOpenFileName(self, "Select image", "", "Images (*.jpg *.png *.jpeg)")
        if image_path:
            self.query_image_path = image_path
            pixmap = QPixmap(image_path).scaledToHeight(400, Qt.SmoothTransformation)
            self.query_preview.setPixmap(pixmap)

    def run_search(self):
        # Warn if no query image has been selected
        if not self.query_image_path:
            QMessageBox.warning(self, "Error", "Please select an image to search.")
            return
        top_k = self.top_k_spinbox.value()
        try:
            # Retrieve a generous number of results, then filter by category/subcategory
            results = get_similar_images(self.query_image_path, top_k=100)
            selected_category = self.category_combo.currentText()
            selected_subcategory = self.subcategory_combo.currentText()
            if selected_category != "All":
                # If a subcategory is also selected, build its full path
                if selected_subcategory != "All":
                    filter_dir = os.path.join(self.image_folder, selected_category, selected_subcategory)
                else:
                    filter_dir = os.path.join(self.image_folder, selected_category)
                # Keep only images within that (sub)category
                results = [r for r in results if os.path.abspath(r[0]).startswith(os.path.abspath(filter_dir))]
            results = results[:top_k]
            self.results_list.clear()
            if not results:
                QMessageBox.information(self, "No Results", "No images found matching the selected filters.")
                return
            for path, score in results:
                category, subcategory = get_category_and_subcategory(path, self.image_folder)
                item = QListWidgetItem()
                widget = ResultItemWidget(path, score, category, subcategory)
                item.setSizeHint(widget.sizeHint())
                self.results_list.addItem(item)
                self.results_list.setItemWidget(item, widget)
        except Exception as e:
            QMessageBox.critical(self, "Error during search", str(e))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    # If you’d like to set a larger global font size, adjust here:
    FONT_SIZE = 15
    ########
    font = QFont()
    font.setPointSize(FONT_SIZE)
    app.setFont(font)
    ##############################
    window = ImageSearchApp()
    window.showMaximized()
    sys.exit(app.exec_())
