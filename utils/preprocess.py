import datasets
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from tqdm import tqdm

from utils import to_json


def get_docs(
        chunk_size: int,
        chunk_overlap: int,
        doc_path: str,
        meta_datas: list[dict]
) -> list[Document]:
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
    print("write to json file...")
    to_json(docs, doc_path)
    return docs
