#!/usr/bin/env python3
"""Verify restored weights and unchanged fine-tuned weights against their sources."""

import argparse
import json
from pathlib import Path

if __package__:
    from .checkpoint import (
        DEFAULT_BASE, DEFAULT_FINETUNE, Checkpoint, default_output, verify_checkpoint,
    )
else:
    from checkpoint import (
        DEFAULT_BASE, DEFAULT_FINETUNE, Checkpoint, default_output, verify_checkpoint,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finetune", type=Path, help="Original local fine-tuned directory")
    parser.add_argument("--base", help="Base HF repo ID or local directory")
    parser.add_argument("--base-revision", help="Base HF branch, tag, or commit")
    parser.add_argument("--merged", type=Path,
                        help="Merged directory or model.safetensors path")
    parser.add_argument("--report", type=Path, help="Optionally write a fresh JSON verification report")
    args = parser.parse_args()

    merged = (args.merged or default_output(args.finetune or DEFAULT_FINETUNE)).expanduser()
    if merged.is_dir():
        merged = merged / "model.safetensors"
    if not merged.is_file():
        parser.error(f"Merged model not found: {merged}")

    # Reuse recorded sources and pinned HF revision; CLI overrides are explicit.
    manifest = merged.parent / "merge_report.json"
    saved = json.loads(manifest.read_text(encoding="utf-8")) if manifest.is_file() else {}
    finetune_path = args.finetune or saved.get("finetune") or DEFAULT_FINETUNE
    base_source = args.base or saved.get("base") or DEFAULT_BASE
    revision = args.base_revision
    if revision is None and args.base is None:
        revision = saved.get("base_revision")

    finetune = Checkpoint.local(finetune_path)
    base = Checkpoint.base(base_source, revision=revision)
    report = verify_checkpoint(finetune, base, merged)
    if args.report:
        report_path = args.report.expanduser().resolve()
        # Reports must not overwrite the model or its config/tokenizer files.
        protected_dirs = {merged.parent.resolve(), finetune.directory, base.directory}
        if report_path.suffix != ".json" or (
            report_path.parent in protected_dirs and report_path.name not in (
                "merge_report.json", "verification_report.json",
            )
        ):
            parser.error("Use a separate JSON report or verification_report.json in the model directory")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if not report["success"]:
        print("VERIFY FAILED")
        print(json.dumps(report, indent=2))
        raise SystemExit(1)
    print(
        f"VERIFY SUCCESS: {report['restored_verified']} restored tensors match the base; "
        f"{report['finetune_verified']} fine-tuned tensors are unchanged.\n"
        "All tensor names, shapes, dtypes, and bytes match their sources."
    )


if __name__ == "__main__":
    main()
