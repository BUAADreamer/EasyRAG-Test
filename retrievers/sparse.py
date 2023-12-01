import os.path
import numpy as np
from langchain.retrievers import TFIDFRetriever as langchain_TFIDFRetriever
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from utils import to_pkl, from_pkl
from utils.preprocess import get_docs, tokenize
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerBase,
)
from retrievers.base import BaseRetriever


class BM25Retriever(BaseRetriever):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.output_path = self.cfg['output_path']
        self.model_path = os.path.join(self.output_path, 'model.pkl')
        self.doc_path = os.path.join(self.output_path, 'docs.pkl')
        os.makedirs(self.output_path, exist_ok=True)
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            self.cfg['llm_name'],
            trust_remote_code=True,
            use_fast=True,
            add_bos_token=False,
            add_eos_token=False,
            padding_side="left",
        )

    def build(self):
        print("obtaining docs...")
        self.docs = get_docs(self.cfg['chunk_size'], self.cfg['chunk_overlap'], self.doc_path, self.cfg['meta_datas'])
        print("calculate sparse features and save index...")
        texts = [doc.page_content for doc in self.docs]
        tokenized_corpus = tokenize(texts, self.tokenizer)
        self.retriever: BM25Okapi = BM25Okapi(tokenized_corpus)
        to_pkl(self.retriever, self.model_path)

    def load(self):
        print("obtaining retriever and docs...")
        self.retriever: BM25Okapi = from_pkl(self.model_path)
        self.documents: list[Document] = from_pkl(self.doc_path)
        self.texts = [doc_obj.page_content for doc_obj in self.documents]

    def retrieve(self,
                 query: str,
                 topk: int
                 ) -> tuple[list[str], list[float]]:
        query = tokenize([query], self.tokenizer)[0]
        scores = self.retriever.get_scores(query)
        sorted_idxs = np.argsort(1 - scores)[:topk]
        scores = sorted(scores, reverse=True)
        docs = [self.texts[sorted_idx] for sorted_idx in sorted_idxs]
        return docs, scores

    def augment(self,
                query: str,
                prompt: str
                ) -> str:
        topk = 2
        docs, scores = self.retrieve(query, topk)
        context = ''
        for doc in docs:
            context += doc + '\n'
        prompt = prompt.format(context)
        return prompt


class TFIDFRetriever(BaseRetriever):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.output_path = self.cfg['output_path']
        self.model_path = os.path.join(self.output_path, 'model.pkl')
        self.doc_path = os.path.join(self.output_path, 'docs.pkl')
        os.makedirs(self.output_path, exist_ok=True)

    def build(self):
        self.docs = get_docs(self.cfg['chunk_size'], self.cfg['chunk_overlap'], self.doc_path, self.cfg['meta_datas'])
        print("calculate sparse features and save index...")
        self.retriever = langchain_TFIDFRetriever.from_documents(self.docs)
        to_pkl(self.retriever, self.model_path)

    def load(self):
        print("obtaining retriever and docs...")
        self.retriever: langchain_TFIDFRetriever = from_pkl(self.model_path)

    def retrieve(self,
                 query: str,
                 topk: int
                 ) -> list[str]:
        results = self.retriever.get_relevant_documents(query)
        docs = []
        for result in results[:topk]:
            docs.append(result.page_content)
        return docs

    def augment(self,
                query: str,
                prompt: str
                ) -> str:
        topk = 2
        docs = self.retrieve(query, topk)
        context = ''
        for doc in docs:
            context += doc + '\n'
        prompt = prompt.format(context)
        return prompt
