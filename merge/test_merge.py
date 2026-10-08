"""Offline regression tests with real safetensors and deliberately damaged outputs."""

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import load_file, save_file

from merge.checkpoint import Checkpoint, verify_checkpoint
from merge.merge import merge_checkpoint


class MergeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ft_tensors = {
            "shared.weight": torch.tensor([9., 8.], dtype=torch.float16),
            "ft_only.weight": torch.tensor([float("nan"), -0.], dtype=torch.float32),
        }
        self.base_tensors = {
            "shared.weight": torch.tensor([1., 2.], dtype=torch.bfloat16),
            "missing.weight": torch.tensor([[3., 4.]], dtype=torch.bfloat16),
            "missing.scalar": torch.tensor(5., dtype=torch.float16),
        }
        # Local-only operations must never touch Hugging Face.
        self.hub_info = self.enterContext(patch("merge.checkpoint.HfApi"))
        self.hub_metadata = self.enterContext(patch("merge.checkpoint.get_safetensors_metadata"))
        self.hub_download = self.enterContext(patch("merge.checkpoint.hf_hub_download"))
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))

    def checkpoint(self, name, tensors, sharded=False):
        directory = self.root / name
        directory.mkdir()
        (directory / "config.json").write_text('{"model_type":"test"}\n')
        (directory / "tokenizer.json").write_text('{"test":true}\n')
        if sharded:
            weight_map = {}
            for i, (key, tensor) in enumerate(tensors.items()):
                shard = f"model-{i:05d}-of-{len(tensors):05d}.safetensors"
                save_file({key: tensor}, directory / shard)
                weight_map[key] = shard
            (directory / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": weight_map}))
        else:
            save_file(tensors, directory / "model.safetensors")
        return Checkpoint.local(directory)

    def inputs(self, sharded=False):
        return (self.checkpoint("ft", self.ft_tensors, sharded),
                self.checkpoint("base", self.base_tensors, sharded))

    def test_single_local_merge_preserves_bytes_and_auxiliary_files(self):
        ft, base = self.inputs()
        out = self.root / "merged"
        report = merge_checkpoint(ft, base, out)
        self.assertTrue(report["success"])
        self.assertEqual(report["restored_verified"], 2)
        self.assertEqual(report["finetune_verified"], 2)
        self.assertEqual(report["merged_tensor_count"], 4)
        self.assertEqual(json.loads((out / "merge_report.json").read_text()), report)
        self.assertEqual((out / "config.json").read_bytes(), (ft.directory / "config.json").read_bytes())
        self.assertEqual((out / "tokenizer.json").read_bytes(), (ft.directory / "tokenizer.json").read_bytes())
        self.hub_info.assert_not_called()
        self.hub_metadata.assert_not_called()
        self.hub_download.assert_not_called()

    def test_sharded_local_merge_produces_one_weight_file(self):
        ft, base = self.inputs(sharded=True)
        out = self.root / "merged"
        self.assertTrue(merge_checkpoint(ft, base, out)["success"])
        self.assertEqual([p.name for p in out.glob("*.safetensors")], ["model.safetensors"])
        self.assertFalse((out / "model.safetensors.index.json").exists())

    def test_already_complete_still_writes_and_verifies(self):
        ft = self.checkpoint("ft", self.base_tensors)
        base = self.checkpoint("base", self.base_tensors)
        report = merge_checkpoint(ft, base, self.root / "merged")
        self.assertTrue(report["success"])
        self.assertEqual(report["restored_verified"], 0)
        self.assertEqual(report["finetune_verified"], 3)

    def test_verifier_detects_missing_extra_changed_shape_dtype_and_bytes(self):
        ft, base = self.inputs()
        expected = {**self.base_tensors, **self.ft_tensors}
        for name in ("missing", "extra", "base_bytes", "ft_bytes", "dtype", "shape"):
            with self.subTest(name=name):
                damaged = {k: v.clone() for k, v in expected.items()}
                if name == "missing":
                    del damaged["missing.weight"]
                elif name == "extra":
                    damaged["unexpected"] = torch.ones(1)
                elif name == "base_bytes":
                    damaged["missing.weight"] += 1
                elif name == "ft_bytes":
                    damaged["shared.weight"] += 1
                elif name == "dtype":
                    damaged["missing.weight"] = damaged["missing.weight"].float()
                else:
                    damaged["missing.weight"] = damaged["missing.weight"].reshape(-1)
                path = self.root / f"{name}.safetensors"
                save_file(damaged, path)
                report = verify_checkpoint(ft, base, path)
                self.assertFalse(report["success"])
                if name == "missing":
                    self.assertEqual(report["missing_tensors"], ["missing.weight"])
                elif name == "extra":
                    self.assertEqual(report["unexpected_tensors"], ["unexpected"])
                else:
                    self.assertEqual(report["mismatched_tensors"][0]["reason"],
                                     name if name in ("dtype", "shape") else "bytes")

    def test_shape_mismatch_rejected_before_output_is_created(self):
        ft, base = self.inputs()
        base.shapes["shared.weight"] = (10,)
        out = self.root / "merged"
        with self.assertRaisesRegex(ValueError, "shapes differ"):
            merge_checkpoint(ft, base, out)
        self.assertFalse(out.exists())

    def test_rejects_overwriting_inputs_or_existing_output(self):
        ft, base = self.inputs()
        with self.assertRaises(ValueError):
            merge_checkpoint(ft, base, ft.directory)
        out = self.root / "merged"
        out.mkdir()
        (out / "sentinel").write_text("keep")
        with self.assertRaises(FileExistsError):
            merge_checkpoint(ft, base, out)
        self.assertEqual((out / "sentinel").read_text(), "keep")

    def test_failed_verification_does_not_publish_output(self):
        ft, base = self.inputs()
        out = self.root / "merged"
        with patch("merge.merge.verify_checkpoint", return_value={"success": False}):
            with self.assertRaisesRegex(RuntimeError, "Verification failed"):
                merge_checkpoint(ft, base, out)
        self.assertFalse(out.exists())
        self.assertEqual(list(self.root.glob(".merged-*")), [])

    def test_missing_or_inconsistent_shards_are_rejected(self):
        ft, _ = self.inputs(sharded=True)
        index = ft.directory / "model.safetensors.index.json"
        mapping = dict(ft.weight_map)
        mapping["not_in_file"] = next(iter(mapping.values()))
        index.write_text(json.dumps({"weight_map": mapping}))
        with self.assertRaisesRegex(ValueError, "do not match"):
            Checkpoint.local(ft.directory)
        mapping["not_in_file"] = "absent.safetensors"
        index.write_text(json.dumps({"weight_map": mapping}))
        with self.assertRaisesRegex(FileNotFoundError, "Missing checkpoint shard"):
            Checkpoint.local(ft.directory)

    def test_adapter_and_nonexistent_local_directory_are_rejected(self):
        ft, _ = self.inputs()
        (ft.directory / "adapter_config.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "LoRA"):
            Checkpoint.local(ft.directory)
        with self.assertRaises(FileNotFoundError):
            Checkpoint.local(self.root / "missing")
        self.hub_info.assert_not_called()

    def test_hub_base_pins_revision_and_downloads_only_required_shards(self):
        ft, disk_base = self.inputs(sharded=True)
        sha = "a" * 40
        self.hub_info.return_value.model_info.return_value.sha = sha
        files_metadata = {
            shard: SimpleNamespace(tensors={key: SimpleNamespace(shape=disk_base.shapes[key])})
            for key, shard in disk_base.weight_map.items()
        }
        self.hub_metadata.return_value = SimpleNamespace(
            weight_map=disk_base.weight_map, files_metadata=files_metadata)
        self.hub_download.side_effect = lambda **kw: str(disk_base.directory / kw["filename"])
        base = Checkpoint.base("test/base", revision="requested-tag")
        report = merge_checkpoint(ft, base, self.root / "merged")
        self.assertTrue(report["success"])
        self.assertEqual(report["base_revision"], sha)
        self.hub_metadata.assert_called_once_with("test/base", revision=sha)
        calls = self.hub_download.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertEqual({c.kwargs["filename"] for c in calls},
                         {disk_base.weight_map[k] for k in ("missing.weight", "missing.scalar")})
        self.assertTrue(all(c.kwargs["revision"] == sha for c in calls))

    def test_cli_rechecks_weights_and_returns_nonzero_on_corruption(self):
        ft, base = self.inputs()
        out = self.root / "merged"
        script_dir = Path(__file__).resolve().parent
        merge = subprocess.run([
            sys.executable, str(script_dir / "merge.py"), "--finetune", str(ft.directory),
            "--base", str(base.directory), "--out", str(out),
        ], capture_output=True, text=True)
        self.assertEqual(merge.returncode, 0, merge.stdout + merge.stderr)
        cmd = [sys.executable, str(script_dir / "verify.py"), "--merged", str(out)]
        valid = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(valid.returncode, 0, valid.stdout + valid.stderr)
        tensors = load_file(out / "model.safetensors")
        tensors["missing.weight"] = tensors["missing.weight"] + 1
        save_file(tensors, out / "model.safetensors")
        invalid = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(invalid.returncode, 1, invalid.stdout + invalid.stderr)
        self.assertIn("VERIFY FAILED", invalid.stdout)
        self.assertIn("missing.weight", invalid.stdout)


if __name__ == "__main__":
    unittest.main()
