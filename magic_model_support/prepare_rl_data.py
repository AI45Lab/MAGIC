"""Convert caller-provided MAGIC task templates to VERL parquet."""

import argparse
import json
from pathlib import Path


def main():
    import pyarrow as pa
    import pyarrow.parquet as pq
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    rows = []
    for i, row in enumerate(records):
        if not isinstance(row["attacker_prompt"], str) or not isinstance(row["seed_prompt"], str):
            raise ValueError(f"Record {i} requires string attacker_prompt and seed_prompt")
        rows.append({"data_source": "game", "prompt": [{"role": "user", "content": row["attacker_prompt"]}],
                     "ability": "safety", "reward_model": {"style": "rule", "ground_truth": ""},
                     "extra_info": {"index": i, "raw_prompt": row["seed_prompt"]}})
    if args.output.exists():
        raise FileExistsError(args.output)
    pq.write_table(pa.Table.from_pylist(rows), args.output)


if __name__ == "__main__":
    main()
