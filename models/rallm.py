import numpy as np
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from builder import load_retriever
from models.base import RALLM
from retrievers import BaseRetriever


class ICRALM(RALLM):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.cfg = cfg
        self.retriever: BaseRetriever = load_retriever(self.cfg)
        self.model_name_or_path = self.cfg['llm_name']
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=(
                torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float32
            ),
        ).eval()
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
            use_fast=True,
            add_bos_token=False,
            add_eos_token=False,
            padding_side="left",
        )

    def retrieve(self,
                 prompt: str,
                 data: dict  # contains 'query' key
                 ) -> None:
        """
        retrieval stage: get retrieved docs
        :param prompt: complete prompt sentence
        :param data: at least contains 'query' key
        """
        query = data['query']
        prompt = '\n请根据这些检索到的文本回答下列题目\n' + prompt
        self.prompt = self.retriever.augment(query, prompt)

    def read(self) -> None:
        """
        read stage: get token distribution
        """
        self.simple_read()

    def retrieve_and_read(self,
                          prompt: str,
                          data: dict  # contains 'query' key
                          ) -> None:
        self.retrieve(prompt, data)
        self.read()

    def simple_read(self) -> None:
        """
        only for without retrieval method
        """
        self.logits = self.get_logits(self.prompt)

    def generate_choice(self) -> str:
        candidate_logits = [self.logits[self.tokenizer(label).input_ids[-1]] for label in ["A", "B", "C", "D"]]
        candidate_logits = torch.tensor(candidate_logits).to(torch.float32)
        probs = (
            F.softmax(
                candidate_logits,
                dim=0,
            )
            .detach()
            .cpu()
            .numpy()
        )
        answer = {i: k for i, k in enumerate(["A", "B", "C", "D"])}[np.argmax(probs)]
        return answer

    def generate_class(self, class_num) -> int:
        label_ls = [str(label) for label in range(class_num)]
        candidate_logits = [self.logits[self.tokenizer(label).input_ids[-1]] for label in label_ls]
        candidate_logits = torch.tensor(candidate_logits).to(torch.float32)
        probs = (
            F.softmax(
                candidate_logits,
                dim=0,
            )
            .detach()
            .cpu()
            .numpy()
        )
        answer = {i: k for i, k in enumerate(label_ls)}[np.argmax(probs)]
        return int(answer)

    def generate(self, prompt) -> str:
        model_inputs = self.tokenizer([prompt], return_tensors="pt").to('cuda')
        generated_ids = self.model.generate(**model_inputs)
        output = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return output

    def get_logits(self, prompt: str):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").cuda()
        logits = self.model(
            input_ids=input_ids,
        ).logits[:, -1].flatten()
        return logits


class REPLUG(ICRALM):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.topk = 5

    def retrieve(self,
                 prompt: str,
                 data: dict
                 ) -> None:
        query = data['query']
        self.prompt = '\n请根据这些检索到的文本回答下列题目\n' + prompt
        self.docs, self.scores = self.retriever.retrieve(query, topk=self.topk)

    def read(self) -> None:
        self.scores = torch.tensor(self.scores)
        weights = F.softmax(self.scores, dim=-1)
        self.logits = None
        for i, doc in enumerate(self.docs):
            prompt = doc + self.prompt
            logits = self.get_logits(prompt)
            if self.logits is None:
                self.logits = logits * weights[i]
            else:
                self.logits += logits * weights[i]


class SelfRAG(ICRALM):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.topk = 5

    def retrieve(self,
                 prompt: str,
                 data: dict
                 ) -> None:
        self.query = data['query']
        self.original_prompt = prompt
        self.prompt = '\n请根据这些检索到的文本回答下列题目\n' + self.original_prompt
        self.docs, self.scores = self.retriever.retrieve(self.query, topk=self.topk)

    def get_is_rel(self, doc):
        prompt = "判断以下的问题和参考文章的相关性，相关请输出1，不相关请输出0。\n问题:" + self.query + "\n参考文章:" + doc
        self.logits = self.get_logits(prompt)
        return self.generate_class(2)

    def get_is_sup(self, doc, output):
        prompt = "判断对于以下的问题，模型回复是否和参考文章相符，很相符请输出2，部分相符请输出1，不相符请输出0。\n问题:" + self.query + "\n参考文章:" + doc + "\n模型回复:" + output
        self.logits = self.get_logits(prompt)
        return self.generate_class(3)

    def get_is_use(self, output):
        prompt = "判断对于以下的问题，模型回复是否有用，根据有用程度打分，很有用输出4，比较有用输出3，一般输出2，比较没用输出1，完全无用输出0。\n问题:" + self.query + "\n模型回复:" + output
        self.logits = self.get_logits(prompt)
        return self.generate_class(5)

    def reflect(self) -> str:
        prompt = self.original_prompt
        docs = []
        scores = []
        need_retrieve = False
        for i, score in enumerate(self.scores):
            if score > 0.5:
                docs.append(self.docs[i])
                scores.append(score)
                need_retrieve = True
        # use retrieved docs or directly use original prompt
        reflect_scores = []
        if need_retrieve:
            for doc in docs:
                reflect_score = 0
                reflect_score += self.get_is_rel(doc)
                prompt = doc + self.prompt
                output = self.generate(prompt)
                reflect_score += self.get_is_sup(doc, output)
                reflect_score += self.get_is_use(output)
                reflect_scores.append(reflect_score)
            zipped = zip(docs, reflect_scores)
            sort_zipped = sorted(zipped, key=lambda x: x[1], reverse=True)
            result = zip(*sort_zipped)
            docs, reflect_scores = [list(x) for x in result]
            prompt = docs[0] + self.prompt
        return prompt

    def read(self) -> None:
        # filter and rank retrieved docs
        prompt = self.reflect()
        self.logits = self.get_logits(prompt)
