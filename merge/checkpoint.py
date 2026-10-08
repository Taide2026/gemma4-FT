"""Shared checkpoint discovery and exact verification for merge.py and verify.py."""

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from huggingface_hub import HfApi, get_safetensors_metadata, hf_hub_download
from safetensors import safe_open


DEFAULT_FINETUNE = Path(__file__).resolve().parents[1] / "output/gemma4_e2b_kinetics384K_FFT"
DEFAULT_BASE = "google/gemma-4-E2B-it"


def default_output(finetune):
    path = Path(finetune).expanduser().resolve()
    return path.with_name(path.name + "-merged")


def group_by_shard(weight_map, keys):
    groups = defaultdict(list)
    for key in sorted(keys):
        groups[weight_map[key]].append(key)
    return dict(sorted(groups.items()))


class Checkpoint:
    def __init__(self, source, directory, weight_map, shapes, revision=None):
        self.source = source
        self.directory = directory
        self.weight_map = weight_map
        self.shapes = shapes
        self.revision = revision
        self._paths = {}

    @classmethod
    def local(cls, directory):
        directory = Path(directory).expanduser().resolve()
        if not directory.is_dir():
            raise FileNotFoundError(f"Local model directory not found: {directory}")
        if (directory / "adapter_config.json").exists():
            raise ValueError("LoRA adapters are not full checkpoints; merge the adapter first.")
        single = directory / "model.safetensors"
        index = directory / "model.safetensors.index.json"
        if single.exists() and index.exists():
            raise ValueError(f"Ambiguous checkpoint: both single-file weights and shard index in {directory}")
        expected_map = None
        if index.is_file():
            expected_map = json.loads(index.read_text(encoding="utf-8")).get("weight_map")
            if not isinstance(expected_map, dict) or not expected_map:
                raise ValueError(f"Empty or invalid weight_map in {index}")
            if not all(isinstance(k, str) and isinstance(v, str)
                       for k, v in expected_map.items()):
                raise ValueError(f"Invalid tensor or shard name in {index}")
            shards = sorted(set(expected_map.values()))
        elif single.is_file():
            shards = [single.name]
        else:
            raise FileNotFoundError(
                f"Expected model.safetensors or model.safetensors.index.json in {directory}. "
                "Point to the exported model directory, not a DeepSpeed optimizer checkpoint."
            )

        weight_map, shapes = {}, {}
        for shard in shards:
            # HF cache files may be symlinks to blobs outside the snapshot directory.
            shard_name = Path(shard)
            if shard_name.is_absolute() or ".." in shard_name.parts or shard_name.suffix != ".safetensors":
                raise ValueError(f"Invalid shard filename: {shard}")
            path = directory / shard
            if not path.is_file():
                raise FileNotFoundError(f"Missing checkpoint shard: {path}")
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key in weight_map:
                        raise ValueError(f"Duplicate tensor across shards: {key}")
                    weight_map[key] = shard
                    shapes[key] = tuple(f.get_slice(key).get_shape())
        if expected_map is not None and weight_map != expected_map:
            raise ValueError(f"Shard contents do not match the weight_map in {index}")
        if not weight_map:
            raise ValueError(f"Checkpoint contains no tensors: {directory}")
        return cls(str(directory), directory, weight_map, shapes)

    @classmethod
    def base(cls, source, revision=None):
        path = Path(source).expanduser()
        if path.exists() or path.is_absolute() or str(source).startswith((".", "~")):
            if revision is not None:
                raise ValueError("--base-revision applies only to a Hugging Face repo.")
            return cls.local(path)
        # Pin metadata and downloads to the same commit, including later verification.
        revision = HfApi().model_info(str(source), revision=revision).sha
        metadata = get_safetensors_metadata(str(source), revision=revision)
        shapes = {
            key: tuple(metadata.files_metadata[shard].tensors[key].shape)
            for key, shard in metadata.weight_map.items()
        }
        if not shapes:
            raise ValueError(f"Base checkpoint contains no tensors: {source}")
        return cls(str(source), None, dict(metadata.weight_map), shapes, revision)

    def shard_path(self, shard):
        if self.directory is not None:
            return self.directory / shard
        if shard not in self._paths:
            self._paths[shard] = Path(hf_hub_download(
                repo_id=self.source, filename=shard, revision=self.revision,
            ))
        return self._paths[shard]


def check_shapes(finetune, base):
    shared = finetune.weight_map.keys() & base.weight_map.keys()
    if not shared:
        raise ValueError("Fine-tuned and base checkpoints have no tensor names in common.")
    mismatches = [key for key in sorted(shared) if finetune.shapes[key] != base.shapes[key]]
    if mismatches:
        raise ValueError(
            "Fine-tuned/base tensor shapes differ; choose the matching base model "
            f"(e.g. E2B vs E4B). First mismatches: {mismatches[:10]}"
        )


def verify_checkpoint(finetune, base, merged_path):
    """Compare every tensor to the original FT or base file, including raw bytes."""
    check_shapes(finetune, base)
    merged_path = Path(merged_path).resolve()
    restored = sorted(base.weight_map.keys() - finetune.weight_map.keys())
    expected = finetune.weight_map.keys() | base.weight_map.keys()
    report = {
        "success": False,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "finetune": finetune.source,
        "base": base.source,
        "base_revision": base.revision,
        "merged_model": str(merged_path),
        "finetune_tensor_count": len(finetune.weight_map),
        "base_tensor_count": len(base.weight_map),
        "restored_tensor_count": len(restored),
        "restored_tensors": restored,
        "finetune_verified": 0,
        "restored_verified": 0,
        "mismatched_tensors": [],
    }
    with safe_open(merged_path, framework="pt", device="cpu") as merged:
        actual = set(merged.keys())
        report["merged_tensor_count"] = len(actual)
        report["missing_tensors"] = sorted(expected - actual)
        report["unexpected_tensors"] = sorted(actual - expected)
        for source, keys, counter in (
            (finetune, finetune.weight_map, "finetune_verified"),
            (base, restored, "restored_verified"),
        ):
            for shard, shard_keys in group_by_shard(source.weight_map, keys).items():
                print(f"Verifying {len(shard_keys)} tensors from {source.source}/{shard}", flush=True)
                with safe_open(source.shard_path(shard), framework="pt", device="cpu") as original:
                    for key in shard_keys:
                        if key not in actual:
                            continue
                        left = merged.get_tensor(key)
                        right = original.get_tensor(key)
                        if left.shape != right.shape:
                            reason = "shape"
                        elif left.dtype != right.dtype:
                            reason = "dtype"
                        elif not torch.equal(left.reshape(-1).view(torch.uint8),
                                             right.reshape(-1).view(torch.uint8)):
                            # Byte comparison handles copied NaNs and signed zero exactly.
                            reason = "bytes"
                        else:
                            reason = None
                        if reason:
                            report["mismatched_tensors"].append({"name": key, "reason": reason})
                        else:
                            report[counter] += 1
                        del left, right
    report["success"] = not any(report[key] for key in (
        "missing_tensors", "unexpected_tensors", "mismatched_tensors",
    ))
    return report
