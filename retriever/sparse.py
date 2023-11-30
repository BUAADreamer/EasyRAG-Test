import os.path
import datasets
from langchain.retrievers import ElasticSearchBM25Retriever, \
    BM25Retriever as langchain_BM25Retriever, \
    TFIDFRetriever as langchain_TFIDFRetriever, \
    KNNRetriever as langchain_KNNRetriever

from utils import to_pkl, from_pkl
from utils.preprocess import get_docs


class BM25Retriever:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.output_path = self.cfg['output_path']
        self.model_path = os.path.join(self.output_path, 'model.pkl')
        self.doc_path = os.path.join(self.output_path, 'docs.jsonl')
        os.makedirs(self.output_path, exist_ok=True)

    def build(self):
        self.docs = get_docs(self.cfg['chunk_size'], self.cfg['chunk_overlap'], self.doc_path, self.cfg['meta_datas'])
        print("calculate sparse features and save index...")
        self.retriever = langchain_BM25Retriever.from_documents(self.docs)
        to_pkl(self.retriever, self.model_path)

    def load(self):
        self.retriever: langchain_BM25Retriever = from_pkl(self.model_path)

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


class ESBM25Retriever:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.output_path = self.cfg['output_path']
        self.model_path = os.path.join(self.output_path, 'model.pkl')
        self.doc_path = os.path.join(self.output_path, 'docs.jsonl')
        os.makedirs(self.output_path, exist_ok=True)
        self.elasticsearch_url = self.cfg['elasticsearch_url']

    def build(self):
        self.docs = get_docs(self.cfg['chunk_size'], self.cfg['chunk_overlap'], self.doc_path, self.cfg['meta_datas'])
        print("calculate sparse features and save index...")
        self.retriever = ElasticSearchBM25Retriever.create(self.elasticsearch_url, "langchain-index")
        self.retriever.add_texts(self.texts)

    def load(self):
        self.docs = datasets.load_dataset('json', data_files=self.doc_path)['train'].to_list()
        self.texts = [doc['page_content'] for doc in self.docs]
        self.retriever = ElasticSearchBM25Retriever.create(self.elasticsearch_url, "langchain-index")

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


class TFIDFRetriever:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.output_path = self.cfg['output_path']
        self.model_path = os.path.join(self.output_path, 'model.pkl')
        self.doc_path = os.path.join(self.output_path, 'docs.jsonl')
        os.makedirs(self.output_path, exist_ok=True)

    def build(self):
        self.docs = get_docs(self.cfg['chunk_size'], self.cfg['chunk_overlap'], self.doc_path, self.cfg['meta_datas'])
        print("calculate sparse features and save index...")
        self.retriever = langchain_TFIDFRetriever.from_documents(self.docs)
        to_pkl(self.retriever, self.model_path)

    def load(self):
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
