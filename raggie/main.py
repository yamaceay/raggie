from typing import List, Tuple, Optional, Union
import faiss

from .types import RaggieDataClass, RaggieModelClass

class Raggie:
    """Raggie is a retriever that uses a model to find similar keys and values based on embeddings."""
    def __init__(self, model: RaggieModelClass, data: RaggieDataClass):
        """Initialize with model and key-value pairs."""
        self.model = model
        self.data = data

        self.keys = [pair[0] for pair in self.data.data]
        self.values = [pair[1] for pair in self.data.data]

        embedding_dim = self.model.model.get_sentence_embedding_dimension()
        self.key_index = faiss.IndexFlatL2(embedding_dim)
        self.value_index = faiss.IndexFlatL2(embedding_dim)
        self._build_indices()

    def evaluate_rank(self, value: str, ground_truth: str, top_k: int = 10) -> int:
        """Evaluate the rank of a key based on its ground truth value."""
        value_embedding = self.model.predict([value])
        distances, indices = self.value_index.search(value_embedding, top_k)
        for rank, idx in enumerate(indices[0], start=1):
            if self.keys[idx] == ground_truth:
                return rank
        return -1

    def most_similar(self, 
                     queries: Optional[List[str]] = None, 
                     keys: Optional[List[str]] = None, 
                     return_all_scores: bool = False, 
                     *args, **kwargs
    ) -> Union[List[List[Tuple[str, float]]], List[List[str]]]:
        """Retrieve most similar keys for given queries."""
        if queries is None and keys is None:
            raise ValueError("Either 'queries' or 'keys' must be provided.")

        if keys is None:
            results = self._query_similar_docs(queries, *args, **kwargs)
        elif queries is None:
            results = self._query_similar_keys(keys, *args, **kwargs)
        else:
            raise ValueError("Only one of 'queries' or 'keys' should be provided.")

        if not return_all_scores:
            results = [[res[0] for res in result] for result in results]
        return results

    def retrieve(self, 
                 queries: List[str], 
                 top_k: int = 5, 
                 verbose: bool = True,
                 return_all_scores: bool = False
    ) -> List[List[Tuple[str, float]]]:
        """Retrieve keys based on queries."""
        value_embeddings = self.model.predict(queries)
        distances, indices = self.value_index.search(value_embeddings, top_k)
        results = []
        for i, query in enumerate(queries):
            result = [(self.keys[idx], distances[i][j]) for j, idx in enumerate(indices[i])]
            if verbose:
                print(f"Results for query '{query}': {result}")
            results.append(result)
        if not return_all_scores:
            results = [[res[0] for res in result] for result in results]
        return results

    def _query_similar_keys(self, 
                            keys: List[str], 
                            top_k: int = 5, 
                            verbose: bool = True
    ) -> List[List[Tuple[str, float]]]:
        """Find similar keys based on embeddings."""
        key_embeddings = self.model.predict(keys)
        distances, indices = self.key_index.search(key_embeddings, top_k)
        results = []
        for i, key in enumerate(keys):
            result = [(self.keys[idx], distances[i][j]) for j, idx in enumerate(indices[i])]
            if verbose:
                print(f"Similar keys for '{key}': {result}")
            results.append(result)
        return results

    def _query_similar_docs(self, 
                            queries: List[str], 
                            top_k: int = 5, 
                            verbose: bool = True
    ) -> List[List[Tuple[str, float]]]:
        """Find similar documents based on embeddings."""
        query_embeddings = self.model.predict(queries)
        distances, indices = self.value_index.search(query_embeddings, top_k)
        results = []
        for i, query in enumerate(queries):
            result = [(self.values[idx], distances[i][j]) for j, idx in enumerate(indices[i])]
            if verbose:
                print(f"Similar documents for '{query}': {result}")
            results.append(result)
        return results

    def _build_indices(self) -> None:
        """Build separate indices for keys and values."""
        key_embeddings = self.model.predict(self.keys)
        value_embeddings = self.model.predict(self.values)
        self.key_index.add(key_embeddings)
        self.value_index.add(value_embeddings)
