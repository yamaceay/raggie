from typing import List, Optional, Union

import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import seaborn as sns

from .types import RaggiePlotterClass

class RaggiePlotter(RaggiePlotterClass):
    """
    Raggie plotter for visualizing keys using t-SNE and optional k-means clustering.

    This class provides methods to reduce embeddings to 2D space and visualize
    them with clustering and annotations.
    """

    def __init__(self, model):
        """
        Initialize the RaggiePlotter instance.

        Args:
            model: The model used to encode keys into embeddings.
        """
        self.model = model

    def plot(self, 
             keys: List[str], 
             perplexity: Optional[float] = None, 
             learning_rate: Union[float, str] = 'auto', 
             n_iter_without_progress: int = 1000, 
             random_state: int = 42, 
             n_clusters: int = None,
             show: bool = True,
             save_path: Optional[str] = None
    ) -> None:
        """
        Perform t-SNE dimensionality reduction and visualize keys with optional k-means clustering.

        Args:
            keys (List[str]): List of keys to visualize.
            perplexity (Optional[float]): Perplexity parameter for t-SNE.
            learning_rate (Union[float, str]): Learning rate for t-SNE.
            n_iter_without_progress (int): Number of iterations without progress before stopping.
            random_state (int): Random seed for reproducibility.
            n_clusters (int): Number of clusters for k-means (optional).
            show (bool): Whether to display the plot.
            save_path (Optional[str]): Path to save the plot (optional).
        """
        sns.set_theme(style="whitegrid")

        embeddings = self.model.model.encode(keys, convert_to_numpy=True)
        perplexity = perplexity or min(30, len(keys) - 1)
        assert len(keys) >= 2, "At least two keys are required for t-SNE visualization."

        reduced_embeddings = self._compute_tsne_embeddings(
            embeddings, perplexity, learning_rate, n_iter_without_progress, random_state
        )

        fig, ax = plt.subplots(figsize=(12, 8))

        if n_clusters is not None:
            cluster_labels = self._perform_kmeans_clustering(reduced_embeddings, n_clusters)
            palette = sns.color_palette("tab10", n_clusters)
            cluster_colors = {cluster: palette[cluster] for cluster in range(n_clusters)}

            centroids = []
            group_names = []
            for cluster in range(n_clusters):
                cluster_points = reduced_embeddings[cluster_labels == cluster]
                centroids.append(cluster_points.mean(axis=0))
                group_names.append(", ".join([keys[i] for i in range(len(keys)) if cluster_labels[i] == cluster]))
            centroids = np.array(centroids)

            scatter = self._plot_data_points(reduced_embeddings, cluster_labels, palette)
            self._plot_centroids(centroids, group_names, cluster_colors)
        else:
            scatter = self._plot_data_points(reduced_embeddings, keys, "viridis")

        ax.set_title("t-SNE Visualization of Keys", fontsize=16)
        ax.set_xlabel("Dimension 1", fontsize=12)
        ax.set_ylabel("Dimension 2", fontsize=12)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Plot saved to {save_path}")

        if show:
            plt.show()

        return fig

    def _compute_tsne_embeddings(self, 
                                 embeddings: np.ndarray, 
                                 perplexity: Optional[float], 
                                 learning_rate: Union[float, str], 
                                 n_iter_without_progress: int, 
                                 random_state: int
    ) -> np.ndarray:
        """Compute t-SNE embeddings."""
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate=learning_rate,
            n_iter_without_progress=n_iter_without_progress,
            random_state=random_state,
        )
        return tsne.fit_transform(embeddings)

    def _perform_kmeans_clustering(self, 
                                   embeddings: np.ndarray, 
                                   n_clusters: int
    ) -> np.ndarray:
        """Perform k-means clustering on the embeddings."""
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        return cluster_labels

    def _plot_data_points(self, 
                          reduced_embeddings: np.ndarray,
                          hue: Union[List[str], np.ndarray], 
                          palette: List[str]
    ) -> sns.scatterplot:
        """Plot data points with optional clustering."""
        scatter = sns.scatterplot(
            x=reduced_embeddings[:, 0],
            y=reduced_embeddings[:, 1],
            hue=hue,
            palette=palette,
            s=100,
            alpha=0.8
        )
        return scatter

    def _plot_centroids(self, 
                        centroids: np.ndarray, 
                        group_names: list, 
                        cluster_colors: dict
    ) -> None:
        """Plot centroids with matching colors and annotate them."""
        for cluster, centroid in enumerate(centroids):
            plt.scatter(
                centroid[0],
                centroid[1],
                c=[cluster_colors[cluster]],
                s=300,
                edgecolors='black'
            )

            wrapped_group_name = "\n".join(group_names[cluster].split(", "))
            plt.text(
                centroid[0],
                centroid[1],
                wrapped_group_name,
                fontsize=10,
                ha='center',
                va='center',
                color='black',
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='black')
            )