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
import subprocess

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QListWidget, QListWidgetItem, QMessageBox,
    QProgressBar, QSpinBox, QHBoxLayout, QComboBox, QLineEdit
)
from PyQt5.QtGui import QPixmap, QDesktopServices, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QUrl

import hashlib
import json

# --- CLIP imports ---
import torch
import open_clip

# ----------------------------
# CONFIGURATION AND CORE FUNCTIONS
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_FILE = os.path.join(BASE_DIR, "index.npy")
PATHS_FILE = os.path.join(BASE_DIR, "paths.txt")
CLIP_FEATURES_FILE = os.path.join(BASE_DIR, "clip_index.npy")
CLIP_PATHS_FILE = os.path.join(BASE_DIR, "clip_paths.txt")
BACKUP_FOLDER = os.path.join(BASE_DIR, "backup")

BATCH_SIZE = 128
IMAGE_SIZE = (224, 224)

# Load models
tf_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(224,224,3))
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='laion2b_s34b_b79k'
)
clip_model = clip_model.to(device)
clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')

# ----------------------------
# UTILITY FUNCTIONS
# ----------------------------
def compute_file_hash(path):
    if not os.path.exists(path): return None
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def backup_existing_files():
    os.makedirs(BACKUP_FOLDER, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    current = {
        'index': compute_file_hash(FEATURES_FILE),
        'paths': compute_file_hash(PATHS_FILE),
        'clip_index': compute_file_hash(CLIP_FEATURES_FILE),
        'clip_paths': compute_file_hash(CLIP_PATHS_FILE)
    }
    hash_file = os.path.join(BACKUP_FOLDER, "last_backup_hashes.json")
    last = {}
    if os.path.exists(hash_file):
        with open(hash_file, 'r') as f:
            last = json.load(f)
    if current != last:
        for key, file in [('index', FEATURES_FILE), ('paths', PATHS_FILE),
                          ('clip_index', CLIP_FEATURES_FILE), ('clip_paths', CLIP_PATHS_FILE)]:
            if os.path.exists(file):
                ext = os.path.splitext(file)[1]
                dst = os.path.join(BACKUP_FOLDER, f"{key}_{ts}{ext}")
                shutil.copy(file, dst)
        with open(hash_file, 'w') as f:
            json.dump(current, f)


def load_existing_data():
    m_feats, m_paths = None, set()
    if os.path.exists(FEATURES_FILE) and os.path.exists(PATHS_FILE):
        m_feats = np.load(FEATURES_FILE)
        with open(PATHS_FILE, 'r') as f:
            m_paths = set(line.strip() for line in f)
    c_feats, c_paths = None, set()
    if os.path.exists(CLIP_FEATURES_FILE) and os.path.exists(CLIP_PATHS_FILE):
        c_feats = np.load(CLIP_FEATURES_FILE)
        with open(CLIP_PATHS_FILE, 'r') as f:
            c_paths = set(line.strip() for line in f)
    return (m_feats, m_paths), (c_feats, c_paths)


def save_data(features, paths, feat_file, paths_file):
    np.save(feat_file, features)
    with open(paths_file, 'w') as f:
        for p in paths:
            f.write(p + '\n')


def list_all_images(folder):
    exts = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    images = []
    for dp, _, files in os.walk(folder):
        for f in files:
            if os.path.splitext(f.lower())[1] in exts:
                images.append(os.path.normpath(os.path.join(dp, f)))
    return images

# ----------------------------
# FEATURE EXTRACTION WITH PROGRESS CALLBACK
# ----------------------------
def extract_batch(model_obj, image_paths, is_clip=False, callback=None, offset=0, span=100):
    feats, paths = [], []
    total = len(image_paths)
    for idx, start in enumerate(range(0, total, BATCH_SIZE)):
        batch = image_paths[start:start+BATCH_SIZE]
        imgs, valid = [], []
        for p in batch:
            try:
                if is_clip:
                    img = keras_image.load_img(p).convert('RGB')
                    inp = clip_preprocess(img).unsqueeze(0).to(device)
                    imgs.append(inp)
                else:
                    img = keras_image.load_img(p, target_size=IMAGE_SIZE)
                    arr = keras_image.img_to_array(img)
                    arr = preprocess_input(arr)
                    imgs.append(arr)
                valid.append(p)
            except:
                pass
        if not imgs:
            continue
        if is_clip:
            batch_input = torch.cat(imgs)
            with torch.no_grad():
                out = clip_model.encode_image(batch_input)
                out = out / out.norm(dim=-1, keepdim=True)
            feat_batch = out.cpu().numpy()
        else:
            batch_input = np.stack(imgs)
            feat_batch = model_obj.predict(batch_input, verbose=0)
        feats.append(feat_batch)
        paths += valid
        if callback:
            progress = offset + int((start + len(batch)) / total * span)
            callback(progress)
    feats = np.vstack(feats) if feats else np.array([])
    if not is_clip:
        feats = normalize(feats)
    return feats, paths


def index_images(folder, progress_callback=None):
    imgs = list_all_images(folder)
    if not imgs:
        return False
    backup_existing_files()
    (m_feats, m_paths), (c_feats, c_paths) = load_existing_data()
    to_index = [p for p in imgs if p not in m_paths]
    if not to_index:
        return False
    # MobileNet (0-50%)
    m_new_feats, m_new_paths = extract_batch(
        tf_model, to_index,
        is_clip=False,
        callback=progress_callback,
        offset=0, span=50
    )
    all_m_feats = np.vstack([m_feats, m_new_feats]) if m_feats is not None else m_new_feats
    all_m_paths = list(m_paths) + m_new_paths
    save_data(all_m_feats, all_m_paths, FEATURES_FILE, PATHS_FILE)
    # CLIP (50-100%)
    c_new_feats, c_new_paths = extract_batch(
        None, to_index,
        is_clip=True,
        callback=progress_callback,
        offset=50, span=50
    )
    all_c_feats = np.vstack([c_feats, c_new_feats]) if c_feats is not None else c_new_feats
    all_c_paths = list(c_paths) + c_new_paths
    save_data(all_c_feats, all_c_paths, CLIP_FEATURES_FILE, CLIP_PATHS_FILE)
    if progress_callback:
        progress_callback(100)
    return True

# ----------------------------
# SEARCH FUNCTIONS
# ----------------------------
def extract_feature(image_path):
    img = keras_image.load_img(image_path, target_size=IMAGE_SIZE)
    arr = keras_image.img_to_array(img)
    arr = np.expand_dims(arr, 0)
    arr = preprocess_input(arr)
    return tf_model.predict(arr, verbose=0)[0]


def get_similar_images(image_path, top_k, category=None, subcategory=None, image_folder=None):
    feats = np.load(FEATURES_FILE)
    with open(PATHS_FILE, 'r') as f:
        paths = [l.strip() for l in f]
    # Filter by category/subcategory if provided
    if category and category != "All" and image_folder:
        filtered = []
        filtered_feats = []
        for i, p in enumerate(paths):
            rel = os.path.relpath(p, image_folder)
            parts = rel.split(os.sep)
            if len(parts) >= 1 and parts[0] == category:
                if subcategory and subcategory != "All":
                    if len(parts) >= 2 and parts[1] == subcategory:
                        filtered.append(p)
                        filtered_feats.append(feats[i])
                else:
                    filtered.append(p)
                    filtered_feats.append(feats[i])
        paths = filtered
        feats = np.array(filtered_feats)
    inp = extract_feature(image_path)
    if len(feats) == 0:
        return []
    sims = cosine_similarity([inp], feats)[0]
    idxs = np.argsort(sims)[::-1][:top_k]
    return [(paths[i], sims[i] * 100) for i in idxs]


def search_with_text(query, top_k, category=None, subcategory=None, image_folder=None):
    feats = np.load(CLIP_FEATURES_FILE)
    with open(CLIP_PATHS_FILE, 'r') as f:
        paths = [l.strip() for l in f]
    # Filter by category/subcategory if provided
    if category and category != "All" and image_folder:
        filtered = []
        filtered_feats = []
        for i, p in enumerate(paths):
            rel = os.path.relpath(p, image_folder)
            parts = rel.split(os.sep)
            if len(parts) >= 1 and parts[0] == category:
                if subcategory and subcategory != "All":
                    if len(parts) >= 2 and parts[1] == subcategory:
                        filtered.append(p)
                        filtered_feats.append(feats[i])
                else:
                    filtered.append(p)
                    filtered_feats.append(feats[i])
        paths = filtered
        feats = np.array(filtered_feats)
    tokens = clip_tokenizer([query]).to(device)
    with torch.no_grad():
        t_feat = clip_model.encode_text(tokens)
        t_feat = t_feat / t_feat.norm(dim=-1, keepdim=True)
    if len(feats) == 0:
        return []
    sims = cosine_similarity(t_feat.cpu().numpy(), feats)[0]
    idxs = np.argsort(sims)[::-1][:top_k]
    return [(paths[i], sims[i] * 100) for i in idxs]

# ----------------------------
# GUI COMPONENTS
# ----------------------------
class IndexWorker(QThread):
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(bool)

    def __init__(self, folder):
        super().__init__()
        self.folder = folder

    def run(self):
        ok = index_images(self.folder, progress_callback=self.progress_signal.emit)
        self.finished_signal.emit(ok)

class ClickableLabel(QLabel):
    clicked = pyqtSignal()
    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)

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
        # Thumbnail
        thumb = ClickableLabel()
        pix = QPixmap(self.image_path).scaledToWidth(150, Qt.KeepAspectRatio | Qt.SmoothTransformation)
        thumb.setPixmap(pix)
        thumb.clicked.connect(self.open_image)
        layout.addWidget(thumb)
        # Info
        info = QLabel(f"{os.path.basename(self.image_path)}\n{self.score:.2f}%\n{self.category} - {self.subcategory}")
        info.setAlignment(Qt.AlignCenter)
        layout.addWidget(info)
        # Open location button
        btn = QPushButton("Open image location")
        btn.clicked.connect(self.open_path)
        layout.addWidget(btn)
        self.setLayout(layout)

    def open_image(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(self.image_path))
    def open_path(self):
        subprocess.run(["explorer", "/select,", self.image_path])

class ImageSearchApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Search Tool with CLIP")
        self.image_folder = None
        self.query_image = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        # Folder & indexing controls
        self.btn_folder = QPushButton("1. Select images folder")
        self.btn_folder.clicked.connect(self.select_folder)
        layout.addWidget(self.btn_folder)
        self.btn_index = QPushButton("2. Start Indexing")
        self.btn_index.clicked.connect(self.run_index)
        layout.addWidget(self.btn_index)
        self.pb = QProgressBar()
        layout.addWidget(self.pb)
        # Image query
        self.btn_qimg = QPushButton("3. Select image to search")
        self.btn_qimg.clicked.connect(self.select_image)
        layout.addWidget(self.btn_qimg)
        self.lbl_preview = QLabel("Preview")
        self.lbl_preview.setAlignment(Qt.AlignCenter)
        self.lbl_preview.setFixedHeight(200)
        layout.addWidget(self.lbl_preview)
        # Text query
        htxt = QHBoxLayout()
        self.txt = QLineEdit()
        self.txt.setPlaceholderText("Type text to search...")
        self.btn_txt = QPushButton("Search by text")
        self.btn_txt.clicked.connect(self.run_text_search)
        htxt.addWidget(self.txt)
        htxt.addWidget(self.btn_txt)
        layout.addLayout(htxt)
        # Filters & top_k
        h1 = QHBoxLayout()
        self.cmb_cat = QComboBox(); self.cmb_cat.addItem("All")
        self.cmb_sub = QComboBox(); self.cmb_sub.addItem("All")
        h1.addWidget(QLabel("Category:")); h1.addWidget(self.cmb_cat)
        h1.addWidget(QLabel("Subcategory:")); h1.addWidget(self.cmb_sub)
        layout.addLayout(h1)
        h2 = QHBoxLayout()
        self.spin = QSpinBox(); self.spin.setRange(1, 1000); self.spin.setValue(10)
        h2.addWidget(QLabel("Results:")); h2.addWidget(self.spin)
        layout.addLayout(h2)
        # Image search button
        self.btn_search = QPushButton("4. Search similar images")
        self.btn_search.clicked.connect(self.run_image_search)
        layout.addWidget(self.btn_search)
        # Results list
        self.list_results = QListWidget()
        layout.addWidget(self.list_results)
        self.setLayout(layout)
        self.cmb_cat.currentIndexChanged.connect(self.update_subcategories)

    def select_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Select folder")
        if d:
            self.image_folder = d
            self.cmb_cat.clear(); self.cmb_cat.addItem("All")
            for f in os.listdir(d):
                if os.path.isdir(os.path.join(d, f)):
                    self.cmb_cat.addItem(f)
            self.update_subcategories()

    def update_subcategories(self):
        self.cmb_sub.clear(); self.cmb_sub.addItem("All")
        cat = self.cmb_cat.currentText()
        if cat != "All":
            base = os.path.join(self.image_folder, cat)
            if os.path.isdir(base):
                for f in os.listdir(base):
                    if os.path.isdir(os.path.join(base, f)):
                        self.cmb_sub.addItem(f)

    def run_index(self):
        if not self.image_folder:
            return QMessageBox.warning(self, "Error", "Select an images folder first.")
        self.btn_index.setEnabled(False)
        self.worker = IndexWorker(self.image_folder)
        self.worker.progress_signal.connect(self.pb.setValue)
        self.worker.finished_signal.connect(self.index_done)
        self.worker.start()

    def index_done(self, ok):
        self.btn_index.setEnabled(True)
        msg = "Indexing completed." if ok else "No new images to index."
        QMessageBox.information(self, "Indexing", msg)
        self.pb.setValue(0)

    def select_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select image", "", "Images (*.jpg *.png *.jpeg)")
        if path:
            self.query_image = path
            pix = QPixmap(path).scaledToHeight(200, Qt.SmoothTransformation)
            self.lbl_preview.setPixmap(pix)

    def run_image_search(self):
        if not self.query_image:
            return QMessageBox.warning(self, "Error", "Select an image first.")
        cat = self.cmb_cat.currentText()
        sub = self.cmb_sub.currentText()
        res = get_similar_images(
            self.query_image,
            top_k=self.spin.value(),
            category=cat,
            subcategory=sub,
            image_folder=self.image_folder
        )
        self.display_results(res)

    def run_text_search(self):
        txt = self.txt.text().strip()
        if not txt:
            return QMessageBox.warning(self, "Error", "Enter a text query.")
        cat = self.cmb_cat.currentText()
        sub = self.cmb_sub.currentText()
        res = search_with_text(
            txt,
            top_k=self.spin.value(),
            category=cat,
            subcategory=sub,
            image_folder=self.image_folder
        )
        self.display_results(res)

    def display_results(self, results):
        self.list_results.clear()
        if not results:
            return QMessageBox.information(self, "No Results", "No matches found.")
        for path, score in results:
            cat, sub = os.path.relpath(path, self.image_folder).split(os.sep)[:2] if self.image_folder else ("N/A","N/A")
            item = QListWidgetItem()
            widget = ResultItemWidget(path, score, cat, sub)
            item.setSizeHint(widget.sizeHint())
            self.list_results.addItem(item)
            self.list_results.setItemWidget(item, widget)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    font = QFont(); font.setPointSize(14); app.setFont(font)
    window = ImageSearchApp()
    window.showMaximized()
    sys.exit(app.exec_())
