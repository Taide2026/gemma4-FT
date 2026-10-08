# 補齊本機訓練模型的權重

`merge.py` 直接讀取本機 trained model，不需要先上傳到 Hugging Face。
保留 trained model 的所有 tensor，只從 base model 補入缺少的名稱，
輸出單一 `model.safetensors`，並複製本機的 config、tokenizer、processor 等檔案。
這是補齊 checkpoint，不是 LoRA adapter 合併，也不會混合或平均權重。

請在已安裝 `torch`、`safetensors`、`huggingface_hub` 的訓練環境執行。
以下指令從專案根目錄執行。

## 合併並自動驗證

```bash
python merge/merge.py \
  --finetune ./output/gemma4_e2b_kinetics384K_FFT \
  --base google/gemma-4-E2B-it \
  --out ./output/gemma4_e2b_kinetics384K_FFT-merged
```

上述 trained model 與 base 是目前預設值，因此也可以直接執行
`python merge/merge.py`。未指定 `--out` 時，輸出為 trained model
資料夾旁的 `<資料夾名稱>-merged`。輸出目錄必須尚未存在，避免覆蓋原始模型
或殘留舊的分片索引。即使沒有缺失，仍會輸出並驗證完整模型。

`--finetune` 支援 `model.safetensors` 單檔，或
`model.safetensors.index.json` 加上對應分片的本機資料夾。
可以指向訓練最終輸出，也可以指向含上述檔案的 `checkpoint-*`。
不支援只有 LoRA adapter 或 DeepSpeed optimizer state 的資料夾。

`--base` 支援 HF repo 或本機模型資料夾。選擇與訓練模型相符的 base；
例如 E2B 使用 `google/gemma-4-E2B-it`，E4B 使用 `google/gemma-4-E4B-it`。
程式會先檢查共用 tensor 的形狀，但形狀相同不代表是相同模型版本。
HF base 會固定到查得的 commit，且只下載包含缺失 tensor 的權重分片。
需要 HF 模型存取權限時，沿用環境既有的登入設定。

若 base 已在本機，可完全離線執行：

```bash
python merge/merge.py \
  --finetune /path/to/trained_model \
  --base /path/to/base_model \
  --out /path/to/merged_model
```

HF cache 的 snapshot 資料夾也可作為本機 base；必須包含完整權重檔，
若有 index，所有列出的分片都必須存在。
使用 HF repo 時，請沿用訓練環境的 `HF_HOME`，以共用下載快取；
本專案 Nano5 腳本預設為 `/work/$USER/.cache/huggingface`。

## 獨立重跑驗證

```bash
python merge/verify.py \
  --merged ./output/gemma4_e2b_kinetics384K_FFT-merged
```

驗證會讀取 `merge_report.json` 記錄的本機 trained model 路徑、base 來源及
HF commit，重新比對權重檔，不會直接採信報告中的成功結果。
請保留原始 trained model；若搬移模型或驗證舊版合併結果，可明確指定：

```bash
python merge/verify.py \
  --finetune /path/to/trained_model \
  --base /path/to/base_model \
  --merged /path/to/merged_model/model.safetensors \
  --report /path/to/merged_model/verification_report.json
```

合併時及獨立驗證時都會檢查：

1. 輸出 tensor 名稱恰好等於 trained model 與 base model 的聯集。
2. 所有補入的 tensor 與 base 的形狀、dtype、每個 byte 完全相同。
3. 所有 trained model tensor 與原始本機檔案完全相同。

原樣保留 NaN、無限值與正負零；位元一致不代表數值健康，也不代表模型推論品質。
合併與驗證使用 CPU，不需要 GPU；大型權重的寫入與完整比對需要時間和足夠的 RAM、磁碟空間。
只有驗證通過才會建立正式輸出目錄，並輸出 `MERGE COMPLETE`；
獨立驗證成功會輸出 `VERIFY SUCCESS`。失敗時結束碼非零。

合併報告 `merge_report.json` 包含缺失／補入清單、驗證數量、來源及 base commit。
`verify.py --report ...` 可另存本次驗證結果，包含失敗項目。

## 離線回歸測試

```bash
python -m unittest merge.test_merge -v
```

使用小型實際 safetensors 測試單檔、分片、無缺失、錯誤 base、
來源保護、逐項驗證及 CLI 失敗結束碼，不會下載真實模型。
