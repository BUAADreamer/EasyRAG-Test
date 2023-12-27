import os.path
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

from retrievers.base import BaseRetriever
from utils.preprocess import get_docs


class DenseRetriever(BaseRetriever):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.output_path = self.cfg['output_path']
        self.index_path = os.path.join(self.output_path, 'index')
        self.doc_path = os.path.join(self.output_path, 'docs.jsonl')
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(self.index_path, exist_ok=True)

    def build(self):
        self.hf_embed_model = HuggingFaceEmbeddings(
            model_name=self.cfg['embed_name'],
            model_kwargs=self.cfg['model_kwargs'],
            encode_kwargs=self.cfg['encode_kwargs'],
        )
        self.docs = get_docs(self.cfg['chunk_size'], self.cfg['chunk_overlap'], self.doc_path, self.cfg['meta_datas'])
        print("calculate dense features and save index...")
        self.db = FAISS.from_documents(self.docs, self.hf_embed_model)
        self.db.save_local(self.index_path)

    def load(self):
        print("obtaining retriever and docs...")
        self.hf_embed_model = HuggingFaceEmbeddings(
            model_name=self.cfg['embed_name'],
            model_kwargs=self.cfg['model_kwargs'],
            encode_kwargs=self.cfg['encode_kwargs'],
        )
        self.db = FAISS.load_local(self.index_path, self.hf_embed_model)

    def retrieve(self,
                 query: str,
                 topk: int
                 ) -> tuple[list[str], list[float]]:
        docs_and_scores = self.db.similarity_search_with_score(query, k=topk)
        docs = []
        scores = []
        for doc_and_score in docs_and_scores[:topk]:
            docs.append(doc_and_score[0].page_content)
            scores.append(1 - doc_and_score[1])
        return docs, scores

    def augment(self,
                query: str,
                prompt: str,
                prefix: str = ''
                ) -> str:
        topk = 1
        docs, _ = self.retrieve(query, topk)
        context = ''
        for doc in docs:
            context += doc + '\n'
        prompt = prefix + context + prompt
        return prompt
