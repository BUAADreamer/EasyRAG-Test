# coding=utf-8
import argparse
import json
import os
from tqdm import tqdm
import torch
from datasets import load_dataset

from builder import load_model


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
        self.wo_rag = wo_rag
        self.rallm = load_model(config_path)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        self.output_dir = output_dir

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
            prompt = f"以下是中国关于{self.TASK2DESC[task_name]}考试的单项选择题，请选出其中的正确答案。\n"
            if shot != 0:
                shuffled = dataset["train"].shuffle()
                for i in range(min(shot, len(shuffled))):
                    prompt += "\n" + self.build_example(shuffled[i], with_answer=True)
            prompt += "\n" + self.build_example(data, with_answer=False)
            data['query'] = data['question']
            if self.wo_rag:
                self.rallm.prompt = prompt
                self.rallm.simple_read()
            else:
                self.rallm.retrieve_and_read(prompt, data)
            answer = self.rallm.generate_choice()
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
