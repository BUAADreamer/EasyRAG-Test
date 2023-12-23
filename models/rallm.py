import numpy as np
import torch
from transformers import PreTrainedModel, AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from builder import load_retriever
from retrievers import BaseRetriever
import torch.nn.functional as F
from models.base import RALLM


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
        input_ids = self.tokenizer.encode(self.prompt, return_tensors="pt").cuda()
        self.logits = self.model(
            input_ids=input_ids,
        ).logits[:, -1].flatten()

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


class REPLUG(RALLM):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.topk = 5

    def retrieve(self,
                 prompt: str,
                 data: dict
                 ) -> None:
        query = data['query']
        self.prompt = '{}\n请根据这些检索到的文本回答下列题目\n' + prompt
        self.docs, self.scores = self.retriever.retrieve(query, topk=self.topk)

    def read(self) -> None:
        self.scores = torch.tensor(self.scores)
        weights = F.softmax(self.scores, dim=-1)
        self.logits = None
        for i, doc in enumerate(self.docs):
            prompt = self.prompt.format(doc)
            input_ids = self.tokenizer.encode(self.prompt, return_tensors="pt").cuda()
            logits = self.model(
                input_ids=input_ids,
            ).logits[:, -1].flatten()
            if self.logits is None:
                self.logits = logits * weights[i]
            else:
                self.logits += logits * weights[i]
