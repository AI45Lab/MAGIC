"""Convert one MAGIC SFT JSON file to Ministral native reasoning format."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


CANONICAL_MAGIC_OUTPUT = re.compile(
    r"^\s*<think>(?P<thinking>.*?)</think>\s*<answer>(?P<answer>.*?)</answer>\s*$",
    flags=re.IGNORECASE | re.DOTALL,
)


def _convert_record(
    record: dict[str, Any], official_system_prompt: str
) -> tuple[dict[str, Any] | None, str | None]:
    output = record.get("output")
    if not isinstance(output, str):
        return None, "output_not_string"
    match = CANONICAL_MAGIC_OUTPUT.fullmatch(output)
    if match is None:
        return None, "noncanonical_magic_output"
    thinking = match.group("thinking").strip()
    answer = match.group("answer").strip()
    if not thinking:
        return None, "empty_thinking"
    if not answer:
        return None, "empty_answer"

    task_system_prompt = str(record.get("system") or "").strip()
    combined_system_prompt = official_system_prompt.strip()
    if task_system_prompt:
        combined_system_prompt += (
            "\n\n# TASK-SPECIFIC SYSTEM INSTRUCTIONS\n\n" + task_system_prompt
        )

    converted = dict(record)
    converted["system"] = combined_system_prompt
    converted["output"] = f"[THINK]{thinking}[/THINK]{answer}"
    return converted, None



def convert_file(source_path: Path, output_path: Path, official_system_prompt: str) -> int:
    if source_path.resolve() == output_path.resolve():
        raise ValueError("Use a separate output file to preserve the source.")
    records = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON array in {source_path}.")
    converted_records = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Record {index} must be an object.")
        converted, issue = _convert_record(record, official_system_prompt)
        if issue is not None:
            raise ValueError(f"Cannot convert record {index}: {issue}")
        converted_records.append(converted)
    output_path.write_text(
        json.dumps(converted_records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return len(converted_records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--system-prompt-file", required=True, type=Path)
    args = parser.parse_args()
    system_prompt = args.system_prompt_file.read_text(encoding="utf-8").strip()
    if "[THINK]" not in system_prompt or "[/THINK]" not in system_prompt:
        raise ValueError("The system prompt must contain the native reasoning markers.")
    count = convert_file(args.input, args.output, system_prompt)
    print(f"Converted {count} records to {args.output}")


if __name__ == "__main__":
    main()
