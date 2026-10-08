#!/usr/bin/env python3
"""Restore missing base weights into a local fine-tuned safetensors checkpoint."""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

if __package__:
    from .checkpoint import (
        DEFAULT_BASE, DEFAULT_FINETUNE, Checkpoint, check_shapes,
        default_output, group_by_shard, verify_checkpoint,
    )
else:
    from checkpoint import (
        DEFAULT_BASE, DEFAULT_FINETUNE, Checkpoint, check_shapes,
        default_output, group_by_shard, verify_checkpoint,
    )


AUX_FILES = [
    "chat_template.jinja", "config.json", "generation_config.json",
    "processor_config.json", "preprocessor_config.json",
    "tokenizer.json", "tokenizer_config.json", "tokenizer.model",
    "special_tokens_map.json", "added_tokens.json",
]


def merge_checkpoint(finetune, base, output):
    """Write to a new directory only after every output tensor passes verification."""
    output = Path(output).expanduser().resolve()
    if output in (finetune.directory, base.directory):
        raise ValueError("Output must differ from the fine-tuned and base directories.")
    if output.exists():
        raise FileExistsError(f"Output already exists; choose a new directory: {output}")
    if not (finetune.directory / "config.json").is_file():
        raise FileNotFoundError(f"Missing config.json in {finetune.directory}")
    check_shapes(finetune, base)
    missing = sorted(base.weight_map.keys() - finetune.weight_map.keys())
    print(f"Fine-tuned tensors: {len(finetune.weight_map)}", flush=True)
    print(f"Base tensors: {len(base.weight_map)}", flush=True)
    print(f"Missing tensors to restore: {len(missing)}", flush=True)
    for key in missing:
        print(f"  {key}")
    if not missing:
        print("Already complete; copying and verifying the existing weights.")

    tensors = {}
    for source, keys in ((finetune, finetune.weight_map), (base, missing)):
        for shard, shard_keys in group_by_shard(source.weight_map, keys).items():
            print(f"Loading {source.source}: {shard}", flush=True)
            with safe_open(source.shard_path(shard), framework="pt", device="cpu") as f:
                for key in shard_keys:
                    tensors[key] = f.get_tensor(key)

    output.parent.mkdir(parents=True, exist_ok=True)
    # Staging keeps incomplete weights and stale shard indexes out of the output.
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as tmp:
        staging = Path(tmp)
        model_path = staging / "model.safetensors"
        print(f"Writing a single model.safetensors to {output} ...", flush=True)
        save_file(tensors, model_path, metadata={"format": "pt"})
        tensors.clear()
        model_path.chmod(0o644)
        for filename in AUX_FILES:
            src = finetune.directory / filename
            if src.is_file():
                shutil.copy2(src, staging / filename)
        if (finetune.directory / "chat_templates").is_dir():
            shutil.copytree(finetune.directory / "chat_templates", staging / "chat_templates")

        print("Verifying all saved weights against their source files ...", flush=True)
        report = verify_checkpoint(finetune, base, model_path)
        report["merged_model"] = str(output / "model.safetensors")
        if not report["success"]:
            raise RuntimeError(f"Verification failed: {json.dumps(report, indent=2)}")
        (staging / "merge_report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8",
        )
        staging.chmod(0o755)
        staging.rename(output)

    print(
        f"MERGE COMPLETE: restored {report['restored_verified']} tensors; "
        f"preserved {report['finetune_verified']} fine-tuned tensors.\n"
        f"Output: {output}\nReport: {output / 'merge_report.json'}",
        flush=True,
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finetune", type=Path, default=DEFAULT_FINETUNE,
                        help="Local fine-tuned model directory (never downloaded from HF)")
    parser.add_argument("--base", default=DEFAULT_BASE,
                        help="Base HF repo ID or local model directory (must match the model size)")
    parser.add_argument("--base-revision", help="Base HF branch, tag, or commit")
    parser.add_argument("--out", type=Path,
                        help="New output directory (default: <finetune>-merged)")
    args = parser.parse_args()
    finetune = Checkpoint.local(args.finetune)
    output = args.out if args.out is not None else default_output(args.finetune)
    # Fail before any base downloads if an output would be overwritten.
    if output.expanduser().exists():
        parser.error(f"Output already exists; choose a new --out directory: {output}")
    base = Checkpoint.base(args.base, revision=args.base_revision)
    merge_checkpoint(finetune, base, output)


if __name__ == "__main__":
    main()
