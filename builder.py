import fire

from models.rallm import RALLM, REPLUG
from retrievers.base import BaseRetriever
from utils import from_yaml
from retrievers import DenseRetriever, BM25Retriever, TFIDFRetriever


def get_retriever(cfg) -> BaseRetriever:
    method = cfg['retriever']
    if method == 'dense':
        retriever = DenseRetriever(cfg)
    elif method == 'bm25':
        retriever = BM25Retriever(cfg)
    elif method == 'tfidf':
        retriever = TFIDFRetriever(cfg)
    else:
        raise NotImplementedError
    return retriever


def get_model(cfg) -> RALLM:
    method = cfg['model']
    if method == 'rallm':
        model = RALLM(cfg)
    elif method == 'replug':
        model = REPLUG(cfg)
    else:
        raise NotImplementedError
    return model


def build_retriever(
        config: str
):
    cfg = from_yaml(config)
    retriever = get_retriever(cfg)
    retriever.build()


def load_model(
        config: str
) -> RALLM:
    cfg = from_yaml(config)
    model = get_model(cfg)
    return model


def load_retriever(
        config: str
) -> BaseRetriever:
    cfg = from_yaml(config)
    retriever = get_retriever(cfg)
    retriever.load()
    return retriever


if __name__ == '__main__':
    fire.Fire(build_retriever)
