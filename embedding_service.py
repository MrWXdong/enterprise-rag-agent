import torch
from sentence_transformers import SentenceTransformer
from config import settings

_model: SentenceTransformer | None = None


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_embedding_model() -> SentenceTransformer:
    global _model

    if _model is None:
        _model = SentenceTransformer(
            settings.embedding_model,
            device=get_device(),
        )

    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    model = get_embedding_model()
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=False,
    )

    return vectors.tolist()


def embed_text(text: str) -> list[float]:
    vectors = embed_texts([text])
    return vectors[0]
