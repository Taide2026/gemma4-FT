### This branch is used to do ablation study on gemma4 PLEs.

## PLE ablation pilot

This branch carries the full fine-tuning pipeline from local `main` and uses
`google/gemma-4-E4B-it` at revision
`ee0ef6023621cff504d758262d4e04895a5af4a2` as the common starting
checkpoint. All three arms use the
same [UCF11](https://www.crcv.ucf.edu/data/UCF_YouTube_Action.php) videos,
eight uniformly sampled frames as separate images, full fine-tuning for three
epochs, and epoch-level validation. Only the PLE branch differs.

UCF11 has 11 action classes and 25 related-clip groups per class. Use the
official **updated** archive (`UCF11_updated_mpg.rar`, about 1.05 GB). After
extracting it, create a deterministic group-disjoint 60/20/20 train/val/test
split. The JSON paths point to the original MPEG clips; no transcoding is
needed.

```bash
mkdir -p dataset/ucf11
curl -L --fail -o dataset/ucf11/UCF11_updated_mpg.rar \
  https://www.crcv.ucf.edu/data/UCF11_updated_mpg.rar
bsdtar -xf dataset/ucf11/UCF11_updated_mpg.rar -C dataset/ucf11
python3 scripts/prepare_ucf11.py \
  --root dataset/ucf11 \
  --output-dir dataset/ucf11/annotations
```

Run each arm with the same seed. Use separate output directories; the training
entrypoint automatically resumes if a directory already contains checkpoints.

| Arm | PLE in forward pass | PLE weights updated |
|---|---|---|
| `on` | yes | yes |
| `frozen` | yes | no |
| `off` | no (zero input) | no |

```bash
bash scripts/run_ple_ablation.sh train frozen 42   # one arm
bash scripts/run_ple_ablation.sh train all 42      # on, frozen, off in sequence
```

On Nano5, submit the two jobs from the repository root after placing the
dataset there:

```bash
sbatch --export=ALL,PLE_MODE=on,SEED=42 scripts/full_ft_nano5.sbatch
sbatch --export=ALL,PLE_MODE=frozen,SEED=42 scripts/full_ft_nano5.sbatch
sbatch --export=ALL,PLE_MODE=off,SEED=42 scripts/full_ft_nano5.sbatch
```

The PLE-frozen run keeps PLE active but freezes PLE-only weights. The PLE-off run feeds zeros to each decoder layer's PLE input and freezes PLE-only
weights. It saves `ple_enabled=false` in the checkpoint config; `test_set.py`
restores the same behavior automatically. The checkpoint itself still has the
original Gemma 4 tensor shapes.

Evaluate all three selected checkpoints (`bash scripts/run_ple_ablation.sh eval all 42`
runs the commands below for every arm) on the untouched test split with identical
decoding. Compare `macro_f1` first and `top1_accuracy` second in `metrics.json`.

```bash
python3 src/test_set.py \
  --model_id output/ucf11_ple_on_seed42 \
  --data_path dataset/ucf11/annotations/test.json \
  --image_folder dataset/ucf11 \
  --candidate_labels dataset/ucf11/annotations/labels.txt \
  --max_seq_length 3072 --skip_bertscore --max_new_tokens 32
python3 src/test_set.py \
  --model_id output/ucf11_ple_off_seed42 \
  --data_path dataset/ucf11/annotations/test.json \
  --image_folder dataset/ucf11 \
  --candidate_labels dataset/ucf11/annotations/labels.txt \
  --max_seq_length 3072 --skip_bertscore --max_new_tokens 32
```

Repeat with seeds 43 and 44 (e.g. `bash scripts/run_ple_ablation.sh train all 42 43 44`) for a more stable comparison. Record mean and
standard deviation of macro-F1 across seeds. This is a small pilot for the
effect of keeping PLE active during FFT from an existing Gemma 4 checkpoint.
