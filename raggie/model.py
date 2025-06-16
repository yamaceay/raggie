import os
from typing import List
import random

import numpy as np

from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator

from .types import RaggieModelClass, RaggieDataClass

class RaggieModel(RaggieModelClass):
    """
    Default implementation of the RaggieModelClass.

    This class provides methods for training, saving, and predicting embeddings
    using a Sentence Transformer model.
    """

    def __init__(self, 
                 model_path: str = None, 
                 base_model_name: str = None, 
                 output_dir: str = None
    ):
        """
        Initialize the RaggieModel instance.

        Args:
            model_path (str): Path to a pre-trained model.
            base_model_name (str): Name of the base model to use if no pre-trained model is provided.
            output_dir (str): Directory to save the trained model.
        """
        self.output_dir = output_dir
        self.model_path = model_path or self.output_dir
        self.base_model_name = base_model_name or "sentence-transformers/all-MiniLM-L6-v2"

        loaded_model_path = self.model_path if os.path.exists(self.model_path) else self.base_model_name
        self._load(loaded_model_path)
    
        self.train_examples: List[InputExample] = []
        self.val_examples: List[InputExample] = []

    def train(self, data: RaggieDataClass, epochs: int = 5) -> None:
        """
        Train the model using the provided data.

        Args:
            data (RaggieDataClass): The data handler providing training and validation data.
            epochs (int): Number of training epochs.
        """
        self._load_data(data)

        train_dataloader = DataLoader(self.train_examples, shuffle=True, batch_size=16)
        train_loss = losses.CosineSimilarityLoss(self.model)

        evaluator = None
        if self.val_examples:
            evaluator = self._get_evaluator(self.val_examples, name="val")

        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=epochs,
            output_path=self.output_dir
        )

    def save(self, model_path: str) -> None:
        """
        Save the trained model to the specified path.
        """
        self.model.save(model_path)

    def predict(self, docs: List[str]) -> np.ndarray:
        """
        Generate embeddings for the given documents.

        Args:
            docs (List[str]): List of documents to encode.

        Returns:
            np.ndarray: Array of embeddings.
        """
        return self.model.encode(docs, convert_to_numpy=True)

    def _load(self, model_path: str) -> None:
        """
        Load a pre-trained model from the given path.

        Args:
            model_path (str): Path to the pre-trained model.
        """
        self.model = SentenceTransformer(model_path)

    def _get_evaluator(self, pairs: List[InputExample], name="eval") -> InformationRetrievalEvaluator:
        """
        Create an evaluator for assessing the model's performance.

        Args:
            pairs (List[InputExample]): List of input examples for evaluation.
            name (str): Name of the evaluator instance.

        Returns:
            InformationRetrievalEvaluator: Configured evaluator instance.
        """
        queries = {f"q{i}": ex.texts[0] for i, ex in enumerate(pairs)}
        docs = {f"d{i}": ex.texts[1] for i, ex in enumerate(pairs)}
        rel = {f"q{i}": [f"d{i}"] for i in range(len(pairs))}
        return InformationRetrievalEvaluator(queries, docs, rel, name=name)

    def _load_data(self, data: RaggieDataClass, *args, **kwargs) -> None:
        """
        Load training and validation data from the data handler.

        Args:
            data (RaggieDataClass): The data handler providing training and validation data.
        """
        train_data = data.train_data
        val_data = data.val_data

        self.train_examples = [InputExample(texts=[x[0], x[1]], label=1.0) for x in train_data]
        self.val_examples = [InputExample(texts=[x[0], x[1]], label=1.0) for x in val_data]

        train_examples_negative = self._sample_negatives(self.train_examples, *args, **kwargs)
        self.train_examples.extend(train_examples_negative)

        val_examples_negative = self._sample_negatives(self.val_examples, *args, **kwargs)
        self.val_examples.extend(val_examples_negative)

    def _sample_negatives(self, examples: List[InputExample], num_negatives=3) -> List[InputExample]:
        """
        Sample negative examples for training using cosine similarity.

        Args:
            examples (List[InputExample]): List of input examples.
            num_negatives (int): Number of negative examples to sample for each positive example.

        Returns:
            List[InputExample]: List of sampled negative examples.
        """
        texts_a = [ex.texts[0] for ex in examples]
        texts_b = [ex.texts[1] for ex in examples]

        embeddings_a = self.model.encode(texts_a, convert_to_numpy=True)
        embeddings_b = self.model.encode(texts_b, convert_to_numpy=True)

        positive_sims = np.sum(embeddings_a * embeddings_b, axis=1) / (
            np.linalg.norm(embeddings_a, axis=1) * np.linalg.norm(embeddings_b, axis=1)
        )

        min_sim = np.mean(positive_sims) - 2 * np.std(positive_sims)
        max_sim = np.mean(positive_sims) - np.std(positive_sims)

        new_pairs = []
        embeddings = embeddings_b

        for ex in examples:
            query_emb = self.model.encode([ex.texts[0]], convert_to_numpy=True)
            sims = embeddings @ query_emb.T / (
                np.linalg.norm(embeddings, axis=1, keepdims=True) * np.linalg.norm(query_emb)
            )
            sims = sims.squeeze()
            candidates = [(i, sim) for i, sim in enumerate(sims) if texts_a[i] != ex.texts[0] and min_sim < sim < max_sim]
            sampled = random.sample(candidates, min(len(candidates), num_negatives))
            for idx, _ in sampled:
                new_pairs.append(InputExample(texts=[ex.texts[0], texts_b[idx]], label=0.0))

        return new_pairs
