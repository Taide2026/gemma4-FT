import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_ucf11 import LABELS, build_splits


class PrepareUcf11Test(unittest.TestCase):
    def test_group_disjoint_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for label in LABELS:
                for group in range(1, 26):
                    folder = root / label / f"v_action_{group:02d}"
                    folder.mkdir(parents=True)
                    (folder / f"v_action_{group:02d}_01.mpg").touch()

            splits, metadata = build_splits(root, seed=42)
            self.assertEqual({name: len(rows) for name, rows in splits.items()},
                             {"train": 165, "val": 55, "test": 55})
            self.assertTrue(set(metadata["groups"]["train"]).isdisjoint(metadata["groups"]["val"]))
            self.assertTrue(set(metadata["groups"]["train"]).isdisjoint(metadata["groups"]["test"]))
            self.assertTrue(set(metadata["groups"]["val"]).isdisjoint(metadata["groups"]["test"]))


if __name__ == "__main__":
    unittest.main()
