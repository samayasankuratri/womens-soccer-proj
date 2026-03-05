from typing import Generator, Iterable, List, TypeVar

import cv2
import numpy as np
import supervision as sv
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

V = TypeVar("V")

# ORIGINAL (SiglipVisionModel) imports — restore these to swap back:
# import torch
# from tqdm import tqdm
# from transformers import SiglipImageProcessor, SiglipVisionModel
# SIGLIP_MODEL_PATH = 'google/siglip-base-patch16-224'


def create_batches(
    sequence: Iterable[V], batch_size: int
) -> Generator[List[V], None, None]:
    """
    Generate batches from a sequence with a specified batch size.

    Args:
        sequence (Iterable[V]): The input sequence to be batched.
        batch_size (int): The size of each batch.

    Yields:
        Generator[List[V], None, None]: A generator yielding batches of the input
            sequence.
    """
    batch_size = max(batch_size, 1)
    current_batch = []
    for element in sequence:
        if len(current_batch) == batch_size:
            yield current_batch
            current_batch = []
        current_batch.append(element)
    if current_batch:
        yield current_batch


class TeamClassifier:
    """
    A classifier that uses HSV color histograms for feature extraction,
    PCA for dimensionality reduction, and KMeans for clustering.

    ORIGINAL version used SiglipVisionModel. To swap back:
      - Restore commented imports above
      - Replace __init__ and extract_features with the originals below:

    def __init__(self, device: str = 'cpu', batch_size: int = 32):
        self.device = device
        self.batch_size = batch_size
        self.features_model = SiglipVisionModel.from_pretrained(
            SIGLIP_MODEL_PATH).to(device)
        self.processor = SiglipImageProcessor.from_pretrained(SIGLIP_MODEL_PATH)
        self.reducer = PCA(n_components=3)
        self.cluster_model = KMeans(n_clusters=2)

    def extract_features(self, crops: List[np.ndarray]) -> np.ndarray:
        crops = [sv.cv2_to_pillow(crop) for crop in crops]
        batches = create_batches(crops, self.batch_size)
        data = []
        with torch.no_grad():
            for batch in tqdm(batches, desc='Embedding extraction'):
                inputs = self.processor(
                    images=batch, return_tensors="pt").to(self.device)
                outputs = self.features_model(**inputs)
                embeddings = torch.mean(
                    outputs.last_hidden_state, dim=1).cpu().numpy()
                data.append(embeddings)
        return np.concatenate(data)
    """
    def __init__(self, device: str = 'cpu', batch_size: int = 32):
        self.reducer = PCA(n_components=3)
        self.cluster_model = KMeans(n_clusters=2)
        self.outlier_threshold = float('inf')

    def extract_features(self, crops: List[np.ndarray]) -> np.ndarray:
        data = []
        for crop in crops:
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            # Only use saturated pixels (jersey colors); ignore skin/grass/background
            sat_mask = hsv[:, :, 1] > 40
            if sat_mask.sum() < 10:
                sat_mask = np.ones(hsv.shape[:2], dtype=bool)
            h_vals = hsv[:, :, 0][sat_mask]
            s_vals = hsv[:, :, 1][sat_mask]
            h_hist, _ = np.histogram(h_vals, bins=16, range=(0, 180))
            s_hist, _ = np.histogram(s_vals, bins=8, range=(0, 256))
            hist = np.concatenate([h_hist.astype(float), s_hist.astype(float)])
            hist = hist / (hist.sum() + 1e-6)
            data.append(hist)
        return np.array(data)

    def fit(self, crops: List[np.ndarray]) -> None:
        data = self.extract_features(crops)
        projections = self.reducer.fit_transform(data)
        self.cluster_model.fit(projections)
        # Store outlier threshold: mean intra-cluster distance + 2.5 std devs
        centers = self.cluster_model.cluster_centers_
        labels = self.cluster_model.labels_
        dists = np.array([
            np.linalg.norm(p - centers[l])
            for p, l in zip(projections, labels)
        ])
        self.outlier_threshold = dists.mean() + 2.5 * dists.std()

    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        if len(crops) == 0:
            return np.array([])

        data = self.extract_features(crops)
        projections = self.reducer.transform(data)
        return self.cluster_model.predict(projections)

    def get_outlier_mask(self, crops: List[np.ndarray]) -> np.ndarray:
        """Return True for crops whose features are far from both cluster centroids (likely refs)."""
        if len(crops) == 0:
            return np.array([], dtype=bool)
        data = self.extract_features(crops)
        projections = self.reducer.transform(data)
        centers = self.cluster_model.cluster_centers_
        labels = self.cluster_model.predict(projections)
        dists = np.array([
            np.linalg.norm(p - centers[l])
            for p, l in zip(projections, labels)
        ])
        return dists > self.outlier_threshold
