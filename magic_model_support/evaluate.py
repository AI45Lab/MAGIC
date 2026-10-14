"""Generate answers and evaluate caller-supplied JSONL examples."""

import argparse
import asyncio
import json
import math
from pathlib import Path

from magic_model_support.http_client import request
from magic_model_support.reasoning import mistral_reasoning_api_messages


async def evaluate(config):
    records = [json.loads(line) for line in Path(config["input_file"]).read_text().splitlines() if line.strip()]
    ids = [str(row["id"]) for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Evaluation IDs must be unique")
    system = Path(config["system_prompt_file"]).read_text() if config.get("system_prompt_file") else None
    semaphore = asyncio.Semaphore(config.get("concurrency", 1))
    judge = config.get("judge")
    judge_template = Path(judge["prompt_file"]).read_text() if judge else None

    async def generate(row):
        async with semaphore:
            messages = row.get("messages") or [{"role": "user", "content": row["prompt"]}]
            if system:
                messages = [{"role": "system", "content": system}, *messages]
            if config.get("tokenizer_mode") == "mistral":
                messages = mistral_reasoning_api_messages(messages)
            result = await request(config["endpoint"], "/chat/completions", {"messages": messages, **config["generation"]})
            choice = result["choices"][0]
            message = choice["message"]
            answer = message.get("content") or ""
            valid = bool(answer.strip()) and choice["finish_reason"] == "stop"
            output = {"id": row["id"], "response": answer,
                      "reasoning": message.get("reasoning_content") or message.get("reasoning") or "",
                      "finish_reason": choice["finish_reason"], "valid": valid}
            if "reference" in row:
                output["exact_match"] = answer.strip() == str(row["reference"]).strip() if valid else None
            if judge and valid:
                prompt = judge_template.format(prompt=json.dumps(messages, ensure_ascii=False), response=answer,
                                               reference=row.get("reference", ""))
                judged = await request(judge["endpoint"], "/chat/completions", {
                    "messages": [{"role": "user", "content": prompt}], **judge["generation"],
                })
                judgement = judged["choices"][0]
                if judgement["finish_reason"] != "stop":
                    raise ValueError(f"Judge did not finish for example {row['id']}")
                score = json.loads(judgement["message"]["content"])
                if isinstance(score["score"], bool) or not isinstance(score["score"], (int, float)) or not math.isfinite(score["score"]):
                    raise ValueError("Judge must return a finite numeric score in a JSON object")
                output["judge"] = score
            return output

    output_path = Path(config["output_file"])
    summary_path = output_path.with_suffix(".metrics.json")
    if summary_path.exists():
        raise FileExistsError(summary_path)
    with output_path.open("x") as handle:
        outputs = await asyncio.gather(*(generate(row) for row in records))
        for output in outputs:
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")
    valid = sum(row["valid"] for row in outputs)
    scores = [row["judge"]["score"] for row in outputs if "judge" in row]
    exact = [row["exact_match"] for row in outputs if row.get("exact_match") is not None]
    summary = {"total": len(outputs), "valid": valid, "invalid": len(outputs) - valid,
               "coverage": valid / len(outputs) if outputs else 0,
               "exact_match_count": len(exact), "exact_match": sum(exact) / len(exact) if exact else None,
               "judge_count": len(scores), "mean_judge_score": sum(scores) / len(scores) if scores else None}
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(evaluate(json.loads(args.config.read_text()))), indent=2))


if __name__ == "__main__":
    main()
