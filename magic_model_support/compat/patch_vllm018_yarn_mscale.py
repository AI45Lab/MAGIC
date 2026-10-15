#!/usr/bin/env python3
"""Bridge Transformers-5 YaRN mscale fields into vLLM 0.18 attn_factor."""

from __future__ import annotations

import argparse
from pathlib import Path


MARKER = "MAGIC Transformers-5 YaRN mscale compatibility"
OLD = '''        }
        if "mrope_section" in rope_parameters:
'''
NEW = '''        }
        # MAGIC Transformers-5 YaRN mscale compatibility: Transformers infers
        # a final attention factor as get_mscale(factor, mscale) divided by
        # get_mscale(factor, mscale_all_dim). vLLM 0.18 ignores both fields and
        # its YaRN constructor additionally multiplies the supplied
        # attn_factor by get_mscale(factor, 1.0), so compensate for that here.
        mscale = rope_parameters.get("mscale")
        mscale_all_dim = rope_parameters.get("mscale_all_dim")
        if "attn_factor" not in extra_kwargs and mscale and mscale_all_dim:
            import math

            def get_mscale(value: float) -> float:
                if scaling_factor <= 1:
                    return 1.0
                return 0.1 * value * math.log(scaling_factor) + 1.0

            hf_attention_factor = get_mscale(mscale) / get_mscale(mscale_all_dim)
            vllm_default_attention_factor = get_mscale(1.0)
            extra_kwargs["attn_factor"] = (
                hf_attention_factor / vllm_default_attention_factor
            )
        if "mrope_section" in rope_parameters:
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, type=Path)
    args = parser.parse_args()
    text = args.target.read_text(encoding="utf-8")
    if MARKER in text:
        print(f"MAGIC: YaRN mscale compatibility already installed in {args.target}")
        return

    yarn_start = text.index('    elif scaling_type == "yarn":')
    yarn_end = text.index('    elif scaling_type in ["deepseek_yarn"', yarn_start)
    before = text[:yarn_start]
    yarn_block = text[yarn_start:yarn_end]
    after = text[yarn_end:]
    if yarn_block.count(OLD) != 1:
        raise RuntimeError(
            "expected exactly one ordinary-YaRN insertion point; "
            f"found {yarn_block.count(OLD)}"
        )
    yarn_block = yarn_block.replace(OLD, NEW, 1)
    args.target.write_text(before + yarn_block + after, encoding="utf-8")
    installed = args.target.read_text(encoding="utf-8")
    if installed.count(MARKER) != 1:
        raise RuntimeError("YaRN mscale compatibility marker validation failed")
    print(f"MAGIC: installed Transformers-5 YaRN mscale compatibility in {args.target}")


if __name__ == "__main__":
    main()
