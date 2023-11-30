import fire
from utils import from_yaml
from retriever import DenseRetriever, BM25Retriever, ESBM25Retriever, TFIDFRetriever


def get_retriever(cfg):
    method = cfg['method']
    if method == 'dense':
        retriever = DenseRetriever(cfg)
    elif method == 'bm25':
        retriever = BM25Retriever(cfg)
    elif method == 'esbm25':
        retriever = ESBM25Retriever(cfg)
    elif method == 'tfidf':
        retriever = TFIDFRetriever(cfg)
    return retriever


def build_retriever(
        config: str
):
    cfg = from_yaml(config)
    retriever = get_retriever(cfg)
    retriever.build()


def load_retriever(
        config: str
):
    cfg = from_yaml(config)
    retriever = get_retriever(cfg)
    retriever.load()
    return retriever


if __name__ == '__main__':
    fire.Fire(build_retriever)
