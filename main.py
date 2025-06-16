import random
import torch
import numpy as np
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
import faiss
from typing import List, Tuple, Optional

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(42)

class LatentSpaceTrainer:
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2', output_dir: str = "output"):
        self.model_name = model_name
        self.output_dir = output_dir
        self.model = SentenceTransformer(model_name, device=device)

        self.train_pairs: List[InputExample] = []
        self.eval_pairs: Optional[List[InputExample]] = None

    def load_data(self, train_data: List[Tuple[str, str]], eval_data: Optional[List[Tuple[str, str]]] = None):
        self.train_pairs = [InputExample(texts=[a, b], label=1.0) for a, b in train_data]
        if eval_data:
            self.eval_pairs = [InputExample(texts=[a, b], label=1.0) for a, b in eval_data]

    def sample_negatives(self, num_negatives=3, min_sim=-0.1, max_sim=0.1) -> List[InputExample]:
        texts_a, texts_b = zip(*[(ex.texts[0], ex.texts[1]) for ex in self.train_pairs])
        embeddings = self.model.encode(texts_b, convert_to_numpy=True)
        new_pairs = []

        for ex in self.train_pairs:
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

    def get_evaluator(self, pairs: List[InputExample], name: str = "eval"):
        queries = {f"q{i}": ex.texts[0] for i, ex in enumerate(pairs)}
        docs = {f"d{i}": ex.texts[1] for i, ex in enumerate(pairs)}
        rel = {f"q{i}": [f"d{i}"] for i in range(len(pairs))}
        return evaluation.InformationRetrievalEvaluator(queries, docs, rel, name=name)

    def train(self, epochs=10, batch_size=16, neg_sampling=False):
        if neg_sampling:
            negatives = self.sample_negatives()
            self.train_pairs.extend(negatives)

        train_dataloader = DataLoader(self.train_pairs, shuffle=True, batch_size=batch_size)
        train_loss = losses.MultipleNegativesRankingLoss(self.model)

        evaluator = self.get_evaluator(self.eval_pairs) if self.eval_pairs else None

        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=epochs,
            warmup_steps=100,
            evaluation_steps=500,
            output_path=self.output_dir,
            save_best_model=True
        )

    def save(self):
        self.model.save(self.output_dir)

    def load(self, path: str):
        self.model = SentenceTransformer(path, device=device)

class LatentRetriever:
    def __init__(self, model: SentenceTransformer, documents: List[str], ids: Optional[List[str]] = None):
        self.model = model
        self.documents = documents
        self.ids = ids or [f"doc_{i}" for i in range(len(documents))]

        self.embeddings = model.encode(documents, convert_to_numpy=True)
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.embeddings)

    def retrieve(self, queries: List[str], top_k: int = 5) -> List[List[Tuple[str, float]]]:
        query_emb = self.model.encode(queries, convert_to_numpy=True)
        dists, idxs = self.index.search(query_emb, top_k)
        results = []
        for row_dists, row_idxs in zip(dists, idxs):
            results.append([(self.ids[i], float(d)) for i, d in zip(row_idxs, row_dists)])
        return results

    def evaluate_rank(self, query: str, ground_truth: str, top_k: int = 10) -> int:
        results = self.retrieve([query], top_k=top_k)[0]
        for rank, (doc_id, _) in enumerate(results, 1):
            doc = self.documents[int(doc_id.split('_')[1])] if isinstance(doc_id, str) else doc_id
            if doc == ground_truth:
                return rank
        return -1

if __name__ == "__main__":
    import json

    train_data = []
    with open("data/user_train.jsonl", "r") as f:
        for line in f:
            data = json.loads(line)
            data_pair = (data["topic"], data["text"])
            train_data.append(data_pair)

    eval_data = []
    with open("data/user_test.jsonl", "r") as f:
        for line in f:
            data = json.loads(line)
            data_pair = (data["topic"], data["text"])
            eval_data.append(data_pair)

    trainer = LatentSpaceTrainer()
    trainer.load_data(train_data, eval_data)
    trainer.train(epochs=5, neg_sampling=True)
    trainer.save()

    model = SentenceTransformer("output")
    retriever = LatentRetriever(model, [x[1] for x in train_data])
    results = retriever.retrieve(["basketball"])
    print("Retrieved documents:")
    for doc, score in results[0]:
        print(f"Document: {doc}, Score: {score}")

    rank = retriever.evaluate_rank("sports", "sports are fun")
    print("Rank of 'sports are fun':", rank)

