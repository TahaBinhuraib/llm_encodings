import torch
from sentence_transformers import SentenceTransformer, utils

device = "cuda" if torch.cuda.is_available() else "cpu"


class Embedding:
    """Embedding class for sentence-transformers"""

    def __init__(self):
        self.model_name = "flax-sentence-embeddings/all_datasets_v3_mpnet-base"
        self.model = SentenceTransformer(self.model_name)  # should load from cache

    def encode(self, text):
        # this can take a batch or a single text
        return self.model.encode(text)

    # simple cosine similarity between query and docs
    def similarity(self, docs, query) -> dict:
        # encode query and docs
        query_emb = self.model.encode(query)
        doc_emb = self.model.encode(docs)

        scores = utils.dot_score(query_emb, doc_emb)[0].cpu().tolist()
        doc_score_pairs = list(zip(docs, scores))
        doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)

        return dict(doc_score_pairs)


try:
    encoder
except Exception:
    print("loading model...")
    encoder = Embedding()
