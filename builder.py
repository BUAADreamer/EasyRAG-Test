from typing import Union

import fire
from utils import from_yaml
from models.base import RALLM
from retrievers.base import BaseRetriever


def get_retriever(cfg: dict) -> BaseRetriever:
    from retrievers import DenseRetriever, BM25Retriever, TFIDFRetriever
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


def get_model(cfg: dict) -> RALLM:
    from models.rallm import ICRALM, REPLUG
    method = cfg['model']
    if method == 'icralm':
        model = ICRALM(cfg)
    elif method == 'replug':
        model = REPLUG(cfg)
    else:
        raise NotImplementedError
    return model


def build_retriever(
        config: Union[str, dict]
):
    if isinstance(config, str):
        config = from_yaml(config)
    retriever = get_retriever(config)
    retriever.build()


def load_model(
        config: Union[str, dict]
) -> RALLM:
    if isinstance(config, str):
        config = from_yaml(config)
    model = get_model(config)
    return model


def load_retriever(
        config: Union[str, dict]
) -> BaseRetriever:
    if isinstance(config, str):
        config = from_yaml(config)
    retriever = get_retriever(config)
    retriever.load()
    return retriever


if __name__ == '__main__':
    fire.Fire(build_retriever)
