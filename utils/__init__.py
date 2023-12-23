import json
import pickle
from typing import List
import yaml
from langchain.schema import Document
from tqdm import tqdm


def to_json(
        documents: List[Document],
        output_path: str
) -> list[dict]:
    parsed_docs = [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in documents]
    with open(output_path, 'w', encoding='utf-8') as f:
        for parsed_doc in tqdm(parsed_docs):
            f.write(json.dumps(parsed_doc, ensure_ascii=False) + '\n')
    return parsed_docs


def from_json(
        path: str
):
    with open(path, 'r') as f:
        return json.loads(f.read())


def from_yaml(
        path: str
):
    with open(path, 'r') as f:
        return yaml.load(f, yaml.FullLoader)


def to_pkl(
        model,
        output_path
) -> None:
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)


def from_pkl(
        output_path
) -> object:
    with open(output_path, 'rb') as f:
        model = pickle.load(f)
    return model
