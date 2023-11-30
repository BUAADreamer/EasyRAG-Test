import os.path

import datasets
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from utils import to_pkl, from_pkl


def get_docs(
        chunk_size: int,
        chunk_overlap: int,
        doc_path: str,
        meta_datas: list[dict]
) -> list[Document]:
    if os.path.exists(doc_path):
        docs = from_pkl(doc_path)
    else:
        documents: list[Document] = []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                       chunk_overlap=chunk_overlap)
        for meta_data in meta_datas:
            path = meta_data['path']
            desc = meta_data['desc']
            data_ls = datasets.load_dataset('json', data_files=path)['train'].to_list()
            for data in tqdm(data_ls, desc=f'obtaining documents of {desc}...'):
                documents.append(
                    Document(
                        page_content=data['content'],
                        metadata=dict(
                            source=data['source'],
                        ),
                    )
                )
        print("split documents...")
        docs = text_splitter.split_documents(documents)
        for i, doc in enumerate(docs):
            doc.metadata['id'] = i
        print("write to doc file...")
        to_pkl(docs, doc_path)
    return docs


def tokenize(texts: list[str], tokenizer: PreTrainedTokenizerBase) -> list[list[str]]:
    bs = 1000000
    N = len(texts)
    tokens_ls = []
    print("tokenize...")
    if N > 1:
        for i in tqdm(range(0, N, bs)):
            max_id = min(N, i + bs)
            token_ids_ls = tokenizer(texts[i:max_id])['input_ids']
            tokens_ls_ = [tokenizer.convert_ids_to_tokens(token_ids) for token_ids in token_ids_ls]
            tokens_ls.extend(tokens_ls_)
    else:
        token_ids_ls = tokenizer(texts)['input_ids']
        tokens_ls = [tokenizer.convert_ids_to_tokens(token_ids) for token_ids in token_ids_ls]
    return tokens_ls
