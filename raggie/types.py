from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Union, Iterator
import numpy

class RaggieDataClass(ABC):
    """
    Abstract base class for Raggie data handling.

    This class defines the interface for iterating over paired data.
    Users can implement their own data handling logic by extending this class.
    """

    @property
    @abstractmethod
    def data(self) -> List[Tuple[str, str]]:
        """
        Get the paired data.

        Returns:
            List[Tuple[str, str]]: A list of tuples containing paired data.
        """
        pass

class RaggieDataLoaderClass(ABC):
    """
    Abstract base class for Raggie data handling.

    This class defines the interface for loading and managing training, testing,
    and validation data. Users can implement their own data handling logic by
    extending this class.
    """

    @property
    @abstractmethod
    def train(self) -> RaggieDataClass:
        """
        Load and return the training data.

        Returns:
            RaggieDataClass: An instance of RaggieDataClass containing the training data.
        """
        pass

    @property
    @abstractmethod
    def test(self) -> RaggieDataClass:
        """
        Load and return the testing data.

        Returns:
            RaggieDataClass: An instance of RaggieDataClass containing the testing data.
        """
        pass

    @property
    @abstractmethod
    def val(self) -> RaggieDataClass:
        """
        Load and return the validation data.

        Returns:
            RaggieDataClass: An instance of RaggieDataClass containing the validation data.
        """
        pass


class RaggieModelClass(ABC):
    """
    Abstract base class for Raggie models.

    This class defines the interface for training, saving, and predicting embeddings
    using a machine learning model.
    """

    @abstractmethod
    def train(self, data: RaggieDataClass, *args, **kwargs) -> None:
        """
        Train the model using the provided data.

        Args:
            data (RaggieDataClass): The data handler providing training and validation data.
            *args: Additional arguments for training.
            **kwargs: Additional keyword arguments for training.
        """
        pass

    @abstractmethod
    def save(self) -> None:
        """
        Save the trained model to the output directory.
        """
        pass

    @abstractmethod
    def predict(self, docs: List[str]) -> numpy.ndarray:
        """
        Generate embeddings for the given documents.

        Args:
            docs (List[str]): List of documents to encode.

        Returns:
            numpy.ndarray: Array of embeddings.
        """
        pass

class RaggiePlotterClass(ABC):
    """
    Abstract base class for Raggie plotters.

    This class defines the interface for visualizing embeddings using t-SNE
    and optional clustering.
    """

    @abstractmethod
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
        Plot the t-SNE visualization of keys with optional clustering.

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
        pass