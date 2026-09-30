#!/usr/bin/env python3
"""Compare Gemma 4 model architectures without downloading weights.

Reads config.json and the safetensors header (via HTTP Range request) from the
Hugging Face Hub, then prints a side-by-side config diff and a per-module
parameter breakdown.

Usage:
    python scripts/inspect_gemma4_arch.py                          # E4B vs 12B (it)
    python scripts/inspect_gemma4_arch.py google/gemma-4-E2B-it google/gemma-4-31B-it
    python scripts/inspect_gemma4_arch.py --full                   # show all config keys, not just diffs
    python scripts/inspect_gemma4_arch.py --tensors "layers.0."    # list tensors matching a substring
    python scripts/inspect_gemma4_arch.py --json out.json          # dump everything as JSON

Set HF_TOKEN if the repo is gated. Only stdlib is required.
"""
import argparse
import json
import os
import re
import struct
import urllib.request
from collections import defaultdict

DEFAULT_MODELS = ["google/gemma-4-E4B-it", "google/gemma-4-12B-it"]
DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I8": 1, "U8": 1, "I32": 4, "I64": 8, "F8_E4M3": 1, "F8_E5M2": 1}


def _open(url, headers=None):
    headers = dict(headers or {})
    if token := os.environ.get("HF_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers))


def fetch_json(repo, filename):
    with _open(f"https://huggingface.co/{repo}/resolve/main/{filename}") as r:
        return json.load(r)


def safetensor_files(repo):
    with _open(f"https://huggingface.co/api/models/{repo}") as r:
        files = [s["rfilename"] for s in json.load(r)["siblings"]]
    return sorted(f for f in files if f.endswith(".safetensors"))


def fetch_safetensors_header(repo, filename):
    url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    with _open(url, {"Range": "bytes=0-7"}) as r:
        (n,) = struct.unpack("<Q", r.read(8))
    with _open(url, {"Range": f"bytes=8-{8 + n - 1}"}) as r:
        header = json.loads(r.read())
    header.pop("__metadata__", None)
    return header


def load_tensors(repo):
    tensors = {}
    for f in safetensor_files(repo):
        tensors.update(fetch_safetensors_header(repo, f))
    return tensors


def numel(shape):
    n = 1
    for d in shape:
        n *= d
    return n


def group_key(name):
    """Collapse a tensor name into a coarse module bucket."""
    name = re.sub(r"^model\.", "", name)
    name = re.sub(r"\.layers\.\d+\.", ".layers.*.", name)
    parts = name.split(".")
    top = parts[0]
    if top == "language_model":
        if "layers.*" in name:
            sub = name.split("layers.*.")[1].split(".")[0]
            return f"language_model.layers.*.{sub}"
        return ".".join(parts[:2])
    return ".".join(parts[:2]) if len(parts) > 2 else top


def breakdown(tensors):
    groups = defaultdict(int)
    for name, meta in tensors.items():
        groups[group_key(name)] += numel(meta["shape"])
    return dict(groups)


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and k not in ("rope_parameters", "id2label", "label2id"):
            out.update(flatten(v, key + "."))
        elif k == "layer_types" and isinstance(v, list):
            full = [i for i, t in enumerate(v) if t == "full_attention"]
            out[key] = f"{len(v)} layers, {len(full)} full @ {full}"
        else:
            out[key] = json.dumps(v) if isinstance(v, (dict, list)) else v
    return out


SKIP_KEYS = re.compile(r"(id2label|label2id|output_attentions|output_hidden_states|return_dict|problem_type|"
                       r"_name_or_path|is_encoder_decoder|chunk_size_feed_forward|initializer_range)")


def fmt(n):
    return f"{n / 1e9:.3f}B" if n >= 1e9 else f"{n / 1e6:.1f}M"


def print_table(rows, headers):
    widths = [max(len(str(r[i])) for r in rows + [headers]) for i in range(len(headers))]
    widths[1:] = [min(w, 45) for w in widths[1:]]
    line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    print(line)
    print("-" * len(line))
    for r in rows:
        print("  ".join(str(c)[:w].ljust(w) for c, w in zip(r, widths)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--full", action="store_true", help="show all config keys, not only differing ones")
    ap.add_argument("--tensors", metavar="SUBSTR", help="list tensors whose name contains SUBSTR")
    ap.add_argument("--no-weights", action="store_true", help="skip safetensors header fetch")
    ap.add_argument("--json", metavar="PATH", help="dump configs + breakdowns to a JSON file")
    args = ap.parse_args()

    short = [m.split("/")[-1] for m in args.models]
    configs = {m: fetch_json(m, "config.json") for m in args.models}

    # ---- config diff ----
    flat = {m: flatten(c) for m, c in configs.items()}
    keys = sorted(set().union(*flat.values()))
    rows = []
    for k in keys:
        if SKIP_KEYS.search(k):
            continue
        vals = [flat[m].get(k, "—") for m in args.models]
        if args.full or len({str(v) for v in vals}) > 1:
            rows.append([k, *vals])
    print(f"\n=== config.json {'(all keys)' if args.full else '(differences only)'} ===\n")
    print_table(rows, ["key", *short])

    # ---- parameter breakdown ----
    result = {"configs": configs, "breakdown": {}}
    if not args.no_weights:
        tensors = {m: load_tensors(m) for m in args.models}
        bds = {m: breakdown(t) for m, t in tensors.items()}
        result["breakdown"] = bds
        groups = sorted(set().union(*bds.values()), key=lambda g: -max(bds[m].get(g, 0) for m in args.models))
        rows = [[g, *[fmt(bds[m][g]) if g in bds[m] else "—" for m in args.models]] for g in groups]
        rows.append(["TOTAL", *[fmt(sum(bds[m].values())) for m in args.models]])
        print("\n=== parameters per module (from safetensors header) ===\n")
        print_table(rows, ["module", *short])

        if args.tensors:
            for m in args.models:
                print(f"\n=== tensors in {m} matching '{args.tensors}' ===")
                for name, meta in sorted(tensors[m].items()):
                    if args.tensors in name:
                        print(f"  {name:90s} {meta['dtype']:5s} {meta['shape']}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
