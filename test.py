# coding=utf-8
import argparse
import json
import os
from tqdm import tqdm
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from builder import load_retriever
from utils import from_yaml


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='configs/bge_small.yaml')
    parser.add_argument(
        "--shot", type=int, default=5, help="number of shot for few-shot learning"
    )
    parser.add_argument(
        "--split", type=str, default="test", help="split of dataset to evaluate"
    )
    parser.add_argument(
        "--output_dir", type=str, default="aeroeval_output", help="output directory"
    )
    parser.add_argument(
        "--wo_rag", action='store_true', default=False, help="with out using retrieval-augmented generation"
    )
    parser.add_argument(
        "--data_path", default="evaluation/aeroeval", help="data path of evaluation"
    )
    return parser.parse_args()


class AeroEval:
    DATA_PATH = "evaluation/aeroeval"
    TASK2DESC = {
        # "aero_basic": "航空航天科普",
        "aero_hard": "航空航天专业",
    }

    def __init__(
            self,
            output_dir: str,
            wo_rag: bool,
            config_path: str,
    ):
        self.retriever = load_retriever(config_path)
        self.cfgs = from_yaml(config_path)
        model_name_or_path = self.cfgs['llm_name']
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=(
                torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float32
            ),
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            use_fast=True,
            add_bos_token=False,
            add_eos_token=False,
            padding_side="left",
        )
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        self.output_dir = output_dir
        self.wo_rag = wo_rag

    def run(self, shot: int, split: str):
        results, accs = {}, {}

        # run all task
        for task_name in self.TASK2DESC:
            print("=" * 100)
            print(f"run task: {task_name}")
            result, acc = self.run_single_task(task_name, shot, split)
            results[task_name] = result
            accs[task_name] = acc
            result_path = os.path.join(self.output_dir, f"{task_name}.json")
            with open(result_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"save result to {result_path}")

        # results
        acc_path = os.path.join(self.output_dir, "acc.json")
        with open(acc_path, "w", encoding='utf-8') as f:
            out_str = json.dumps(accs, indent=2, ensure_ascii=False)
            f.write(out_str)
        average_acc = sum(accs.values()) / len(accs)
        print(f"average acc: {average_acc}")

    @torch.inference_mode()
    def run_single_task(self, task_name: str, shot: int, split: str):
        dataset = load_dataset(self.DATA_PATH, task_name)
        results = []
        acc = 0
        for data in tqdm(dataset[split]):
            if self.wo_rag:
                prompt = f"以下是中国关于{self.TASK2DESC[task_name]}考试的单项选择题，请选出其中的正确答案。\n"
            else:
                query = data['question']
                prompt = '{}' + f'\n以下是中国关于{self.TASK2DESC[task_name]}考试的单项选择题，请参考这些文本选出其中的正确答案。\n'
                prompt = self.retriever.augment(query, prompt)
            if shot != 0:
                shuffled = dataset["train"].shuffle()
                for i in range(min(shot, len(shuffled))):
                    prompt += "\n" + self.build_example(shuffled[i], with_answer=True)
            prompt += "\n" + self.build_example(data, with_answer=False)
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").cuda()

            logits = self.model(
                input_ids=input_ids,
            ).logits[:, -1].flatten()

            candidate_logits = [logits[self.tokenizer(label).input_ids[-1]] for label in ["A", "B", "C", "D"]]
            candidate_logits = torch.tensor(candidate_logits).to(torch.float32)
            probs = (
                torch.nn.functional.softmax(
                    candidate_logits,
                    dim=0,
                )
                .detach()
                .cpu()
                .numpy()
            )
            answer = {i: k for i, k in enumerate(["A", "B", "C", "D"])}[np.argmax(probs)]

            results.append(
                {
                    "prompt": prompt,
                    "correct": answer == data["answer"].strip().upper(),
                    "answer": answer,
                }
            )
            acc += answer == data["answer"].strip().upper()
        acc /= len(dataset[split])
        return results, acc

    def build_example(self, data, with_answer: bool = True):
        question = data["question"]
        choice = "\n".join(
            [
                "A. " + data["A"],
                "B. " + data["B"],
                "C. " + data["C"],
                "D. " + data["D"],
            ]
        )
        answer = data["answer"].strip().upper() if with_answer else ""
        return f"{question}\n{choice}\n答案：{answer}"


def main():
    args = parse_argument()
    aeroeval = AeroEval(args.output_dir, args.wo_rag, args.config)
    aeroeval.run(args.shot, args.split)


if __name__ == "__main__":
    main()
