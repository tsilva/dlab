from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def plot_curves(df: pd.DataFrame, x: str = "step", y: str = "value", hue: str = "metric"):
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, group in df.groupby(hue):
        ax.plot(group[x], group[y], label=str(label))
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None = None,
):
    cm = confusion_matrix(y_true, y_pred)
    display = ConfusionMatrixDisplay(cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(6, 6))
    display.plot(ax=ax, xticks_rotation=45)
    fig.tight_layout()
    return fig, ax


def plot_embedding_2d(
    embeddings: np.ndarray,
    labels: np.ndarray | None = None,
    title: str = "Embedding",
):
    if embeddings.shape[1] > 2:
        embeddings = TSNE(n_components=2, init="pca", learning_rate="auto").fit_transform(
            embeddings
        )
    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, s=8, cmap="tab10")
    ax.set_title(title)
    if labels is not None:
        fig.colorbar(scatter, ax=ax)
    fig.tight_layout()
    return fig, ax


def plot_latent_space(z: np.ndarray, labels: np.ndarray | None = None):
    return plot_embedding_2d(z, labels=labels, title="Latent space")
