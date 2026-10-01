"""
Dataset wrapper for Gemma 4 E4B action-recognition SFT.

Data format (messages JSON):
[
  {"video_metadata": {"fps": 25.0, "duration_sec": 8.3},
   "messages": [
    {"role": "user", "content": [
      {"type": "video", "video": "path/to/clip.mp4"},
      {"type": "text",  "text": "What action is performed?"}
    ]},
    {"role": "assistant", "content": [{"type": "text", "text": "riding a bicycle"}]}
  ]}
]
Legacy top-level fps/duration still accepted. Images also supported.
"""
import copy
import os
import pathlib
import sys
import ast
from typing import Dict, List, Optional, Sequence, Union
import torch
import ujson as json
import transformers
from torch.utils.data import Dataset
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from utils import _pad_sequence

IGNORE_INDEX = -100

PathLike = Union[str, os.PathLike]
ImageFolderInput = Optional[
    Union[PathLike, Sequence[PathLike]]
]

class SupervisedDataset(Dataset):
    def __init__(
            self,
            data_path: str,
            processor: transformers.ProcessorMixin,
            image_folder: ImageFolderInput = None,
            max_seq_length: int = 2304,
            max_decode_frames: int = 8,
            ) -> None:
        self.processor = processor
        self.image_folder = image_folder
        self.image_folders = self._normalize_image_folders(image_folder)
        print(f"[DEBUG] raw image_folder: {image_folder!r}")
        print(f"[DEBUG] normalized image_folders: {self.image_folders!r}")
        self.max_seq_length = max_seq_length
        # Must be >= processor.video_processor.num_frames (default 32).
        # The processor re-samples from this pre-decoded array; passing fewer
        # frames than it expects raises ValueError.
        self.max_decode_frames = max_decode_frames
        with open(data_path, "r") as f:
            self.samples: List[dict] = json.load(f)

    @staticmethod
    def _normalize_image_folders(
            image_folder: ImageFolderInput,
            ) -> List[str]:
        if image_folder is None:
            return []

        # 命令列可能把 list 轉成：
        # "['folder_a', 'folder_b']"
        if isinstance(image_folder, str):
            stripped = image_folder.strip()

            if stripped.startswith("[") and stripped.endswith("]"):
                try:
                    parsed = ast.literal_eval(stripped)
                except (ValueError, SyntaxError) as exc:
                    raise ValueError(
                        f"Invalid image_folder list: {image_folder}"
                    ) from exc

                if not isinstance(parsed, (list, tuple)):
                    raise TypeError(
                        "Parsed image_folder must be a list or tuple"
                    )

                raw_folders = list(parsed)
            else:
                raw_folders = [image_folder]

        elif isinstance(image_folder, os.PathLike):
            raw_folders = [image_folder]

        elif isinstance(image_folder, Sequence):
            raw_folders = list(image_folder)

        else:
            raise TypeError(
                "image_folder must be a path string, os.PathLike, "
                "a sequence of paths, or None; "
                f"got {type(image_folder).__name__}"
            )

        normalized: List[str] = []

        for folder in raw_folders:
            if not isinstance(folder, (str, os.PathLike)):
                raise TypeError(
                    "Every image_folder entry must be a path string or "
                    f"os.PathLike; got {type(folder).__name__}"
                )

            folder_str = os.path.expanduser(os.fspath(folder))

            if not folder_str:
                raise ValueError("image_folder entries must not be empty")

            normalized.append(folder_str)

        return normalized

    def _resolve_path(self, path: str) -> str:
        if path.startswith(("http://", "https://")):
            return path

        # JSON 本身已是有效路徑
        if os.path.exists(path):
            return path

        # JSON 路徑缺少 .mp4
        if os.path.exists(path + ".mp4"):
            return path + ".mp4"

        # 依序搜尋所有 image_folder
        for image_folder in self.image_folders:
            candidate = os.path.join(image_folder, path)

            if os.path.exists(candidate):
                return candidate

            candidate_mp4 = candidate + ".mp4"
            if os.path.exists(candidate_mp4):
                return candidate_mp4

        raise FileNotFoundError(
            f"找不到媒體檔案：{path}\n"
            f"搜尋資料夾：{self.image_folders}"
        )

    def _load_image(self, src) -> Image.Image:
        if isinstance(src, Image.Image):
            return src.convert("RGB")
        if isinstance(src, str):
            path = self._resolve_path(src)
            if path.startswith(("http://", "https://")):
                import requests
                from io import BytesIO
                resp = requests.get(path, timeout=15)
                resp.raise_for_status()
                return Image.open(BytesIO(resp.content)).convert("RGB")
            return Image.open(path).convert("RGB")
        raise TypeError(f"Unsupported image type: {type(src)}")

    def _load_video_as_array(self, src, num_frames: int = 32):
        """
        Decode video with PyAV.

        Returns:
            frames:
                numpy.ndarray with shape [T, H, W, C], dtype uint8
            fps:
                Detected video FPS
            total_num_frames:
                Number of sampled frames returned
        """
        import av
        import numpy as np

        path = self._resolve_path(src) if isinstance(src, str) else src

        if isinstance(path, str) and not os.path.exists(path):
            if os.path.exists(path + ".mp4"):
                path = path + ".mp4"

        container = None

        try:
            try:
                container = av.open(path)
            except Exception as exc:
                raise RuntimeError(
                    f"PyAV failed to open video: {path}"
                ) from exc

            video_streams = container.streams.video

            if len(video_streams) == 0:
                raise ValueError(
                    f"Video contains no video stream: {path}"
                )

            stream = video_streams[0]

            fps = (
                float(stream.average_rate)
                if stream.average_rate
                else 25.0
            )

            original_total_num_frames = stream.frames or 0

            if original_total_num_frames <= 0:
                duration_sec = None

                if (
                    stream.duration is not None
                    and stream.time_base is not None
                ):
                    duration_sec = float(
                        stream.duration * stream.time_base
                    )

                elif container.duration is not None:
                    duration_sec = (
                        float(container.duration) / av.time_base
                    )

                if duration_sec and duration_sec > 0:
                    original_total_num_frames = max(
                        1,
                        int(duration_sec * fps),
                    )

            if original_total_num_frames > 0:
                indices = torch.linspace(
                    0,
                    original_total_num_frames - 1,
                    steps=min(
                        num_frames,
                        original_total_num_frames,
                    ),
                ).long().tolist()

                wanted = set(indices)
                last_wanted = max(wanted)

                kept: Dict[int, "np.ndarray"] = {}

                try:
                    for frame_index, frame in enumerate(
                        container.decode(video=0)
                    ):
                        if frame_index in wanted:
                            kept[frame_index] = frame.to_ndarray(
                                format="rgb24"
                            )

                        if frame_index >= last_wanted:
                            break

                except Exception as exc:
                    raise RuntimeError(
                        f"Failed while decoding video: {path}"
                    ) from exc

                if kept:
                    valid_indices = [
                        index
                        for index in indices
                        if index in kept
                    ]

                    sampled = np.stack(
                        [kept[index] for index in valid_indices],
                        axis=0,
                    )

                    sampled_total_num_frames = sampled.shape[0]

                    return (
                        sampled,
                        fps,
                        sampled_total_num_frames,
                    )

                # Metadata may be incorrect. Reopen and use fallback decoding.
                container.close()
                container = av.open(path)

                if len(container.streams.video) == 0:
                    raise ValueError(
                        f"Video contains no video stream after reopen: {path}"
                    )

            all_frames = []

            try:
                for frame in container.decode(video=0):
                    all_frames.append(
                        frame.to_ndarray(format="rgb24")
                    )

            except Exception as exc:
                raise RuntimeError(
                    f"Failed during full video decode: {path}"
                ) from exc

            if not all_frames:
                raise ValueError(
                    f"Video contains no decodable frames: {path}"
                )

            original_total_num_frames = len(all_frames)

            indices = torch.linspace(
                0,
                original_total_num_frames - 1,
                steps=min(
                    num_frames,
                    original_total_num_frames,
                ),
            ).long().tolist()

            sampled = np.stack(
                [all_frames[index] for index in indices],
                axis=0,
            )

            sampled_total_num_frames = sampled.shape[0]

            return (
                sampled,
                fps,
                sampled_total_num_frames,
            )

        finally:
            if container is not None:
                container.close()

    def _normalize_messages(self, messages: List[dict]):
        """Returns (normalized_messages, video_meta_list).
        video_meta_list has one entry per video content item encountered.
        """
        messages = copy.deepcopy(messages)
        fps_list: List[float] = []
        for msg in messages:
            content = msg.get("content", [])
            if not isinstance(content, list):
                content = [content]
            new_content = []

            for item in content:
                # 如果 content 裡面是純文字字串，轉成 Gemma processor 要的格式
                if isinstance(item, str):
                    new_content.append({
                        "type": "text",
                        "text": item,
                    })
                    continue

                if isinstance(item, dict):
                    if item.get("type") == "image":
                        for key in ("image", "path", "url"):
                            if key in item:
                                item = {**item, key: self._load_image(item[key])}
                                break
                    elif item.get("type") == "video":
                        converted = False
                        for key in ("video", "path", "url"):
                            if key in item:
                                frames, fps, total_num_frames = self._load_video_as_array(
                                    item[key], num_frames=self.max_decode_frames
                                )

                                # 把 video 拆成多張 image，讓 template 產生 image tokens
                                for frame in frames:
                                    new_content.append({
                                        "type": "image",
                                        "image": Image.fromarray(frame).convert("RGB"),
                                    })
                                
                                fps_list.append({
                                    "fps": fps,
                                    "total_num_frames": total_num_frames,
                                })
                                converted = True
                                break
                        if converted:
                            continue

                new_content.append(item)
            msg["content"] = new_content
        return messages, fps_list

    def _build_sample(
            self, messages: List[dict], fps_override: Optional[float] = None
            ) -> Dict[str, torch.Tensor]:
        processor = self.processor
        normalized, detected_fps = self._normalize_messages(messages)

        # Build video_metadata for Gemma 4 frame-timestamp computation.
        # fps_override (stored in JSON during preprocessing) takes priority over
        # the value detected live from the video stream.
        if detected_fps:
            video_metadata = []
            for meta in detected_fps:
                fps = fps_override if fps_override is not None else meta["fps"]
                video_metadata.append({
                    "fps": fps,
                    "total_num_frames": meta["total_num_frames"],
                })
        else:
            video_metadata = None

        processor_kwargs = (
            {"videos_kwargs": {"video_metadata": video_metadata}}
            if video_metadata
            else None
        )

        encoded = processor.apply_chat_template(
                normalized,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=False,
                enable_thinking=False,
                processor_kwargs=processor_kwargs,
        )
        input_ids = encoded["input_ids"].squeeze(0).long()
        attention_mask = encoded["attention_mask"].squeeze(0).long()
        labels = torch.full_like(input_ids, IGNORE_INDEX)

        assistant_roles = {"assistant", "model"}
        for idx, msg in enumerate(normalized):
            if msg["role"] not in assistant_roles:
                continue
            if idx == 0:
                start_len = 0
            else:
                prefix = processor.apply_chat_template(
                        normalized[:idx],
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                        add_generation_prompt=True,
                        enable_thinking=False,
                        processor_kwargs=processor_kwargs,
                        )
                start_len = prefix["input_ids"].size(1)
                # Guard: verify prefix tokens align with full input_ids.
                # Misalignment means the tokenizer is not prefix-stable across
                # add_generation_prompt, which would silently corrupt labels.
                prefix_ids = prefix["input_ids"].squeeze(0)
                if start_len > input_ids.size(0) or not torch.equal(
                        input_ids[:start_len], prefix_ids
                        ):
                    from utils import _log
                    _log(
                            f"WARNING: label span misalignment at turn {idx} "
                            f"(prefix_len={start_len}, seq_len={input_ids.size(0)}) "
                            "— labels skipped for this turn"
                            )
                    continue

            prefix_with_answer = processor.apply_chat_template(
                    normalized[:idx + 1],
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    add_generation_prompt=False,
                    enable_thinking=False,
                    processor_kwargs=processor_kwargs,
                    )
            end_len = prefix_with_answer["input_ids"].size(1)
            labels[start_len:end_len] = input_ids[start_len:end_len]

        if labels.numel() > 0 and labels[0].item() != IGNORE_INDEX:
            labels[0] = IGNORE_INDEX

        # Enforce max_seq_length — truncate text tensors only.
        # Vision tensors (pixel_values*) are kept intact; their placeholder
        # tokens are inserted near the start of input_ids (user turn), so
        # truncating from the tail is safe.
        L = self.max_seq_length
        if input_ids.size(0) > L:
            input_ids = input_ids[:L]
            attention_mask = attention_mask[:L]
            labels = labels[:L]

        data = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                )
        if "pixel_values" in encoded:
            data["pixel_values"] = encoded["pixel_values"]

        if "pixel_values_videos" in encoded:
            pv = encoded["pixel_values_videos"]

            # Gemma4 processor output is usually:
            # [batch, frames, num_patches, hidden_dim]
            # Example from your debug: [1, 32, 630, 768]
            if pv.dim() > 3:
                num_patches, hidden_dim = pv.shape[-2:]
                pv = pv.reshape(-1, num_patches, hidden_dim)

            data["pixel_values"] = pv

        if "image_position_ids" in encoded:
            data["image_position_ids"] = encoded["image_position_ids"]

        if "video_position_ids" in encoded:
            vpos = encoded["video_position_ids"]
            
            # [batch, frames, num_patches, 2] -> [batch * frames, num_patches, 2]
            if vpos.dim() > 3:
                num_patches, two = vpos.shape[-2:]
                vpos = vpos.reshape(-1, num_patches, two)

            data["image_position_ids"] = vpos

        if "mm_token_type_ids" in encoded:
            mm = encoded["mm_token_type_ids"].squeeze(0).long()
            data["mm_token_type_ids"] = mm[:L]

        if not hasattr(self, "_printed_debug_shapes"):
            self._printed_debug_shapes = True
            print("[DEBUG] encoded keys:", list(encoded.keys()))
            for k, v in encoded.items():
                if hasattr(v, "shape"):
                    print(f"[DEBUG] encoded {k}: {tuple(v.shape)}")
            for k, v in data.items():
                if hasattr(v, "shape"):
                    print(f"[DEBUG] data {k}: {tuple(v.shape)}")

        return data

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[i]

        # fps from new-format JSON (video_metadata.fps)
        # with fallback to legacy top-level fps
        meta = sample.get("video_metadata") or {}
        fps = meta.get("fps") or sample.get("fps")

        video_paths = []

        for message in sample.get("messages", []):
            content = message.get("content", [])

            if not isinstance(content, list):
                content = [content]

            for item in content:
                if not isinstance(item, dict):
                    continue

                if item.get("type") != "video":
                    continue

                for key in ("video", "path", "url"):
                    if key in item:
                        video_paths.append(str(item[key]))
                        break

        try:
            return self._build_sample(
                sample["messages"],
                fps_override=fps,
            )

        except Exception as exc:
            raise RuntimeError(
                "\nFailed to build dataset sample.\n"
                f"dataset_index: {i}\n"
                f"video_paths: {video_paths}\n"
                f"video_metadata: {meta}\n"
                f"original_error: {type(exc).__name__}: {exc}"
            ) from exc


class DataCollatorForSupervisedDataset:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids_list = [ex["input_ids"] for ex in examples]
        labels_list = [ex["labels"] for ex in examples]
        has_image = any("pixel_values" in ex for ex in examples)
        has_image_pos = any("image_position_ids" in ex for ex in examples)
        has_video = any("pixel_values_videos" in ex for ex in examples)
        has_video_pos = any("video_position_ids" in ex for ex in examples)

        input_ids = _pad_sequence(input_ids_list, padding_value=self.pad_token_id)
        labels = _pad_sequence(labels_list, padding_value=IGNORE_INDEX)
        attention_mask = (input_ids != self.pad_token_id).long()

        batch = dict(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
                )

        if has_image:
            batch["pixel_values"] = torch.cat(
                    [ex["pixel_values"] for ex in examples if "pixel_values" in ex], dim=0
                    )

        """if has_video:
            batch["pixel_values_videos"] = torch.cat(
                    [ex["pixel_values_videos"] for ex in examples if "pixel_values_videos" in ex], dim=0
                    )
        """

        if has_image_pos:
            batch["image_position_ids"] = torch.cat(
                    [ex["image_position_ids"] for ex in examples if "image_position_ids" in ex],
                    dim=0
                    )

        """if has_video_pos:
            # shape per sample: (num_videos, num_frames, max_patches, 2) → cat on dim=0
            batch["video_position_ids"] = torch.cat(
                    [ex["video_position_ids"] for ex in examples if "video_position_ids" in ex], dim=0
                    )
        """

        mm_token_type_ids_list = [
                ex.get("mm_token_type_ids", torch.zeros_like(ex["input_ids"])) for ex in examples
                ]
        batch["mm_token_type_ids"] = _pad_sequence(mm_token_type_ids_list, padding_value=0)

        """if "pixel_values_videos" in batch and "pixel_values" not in batch:
            batch["pixel_values"] = batch["pixel_values_videos"]
        """

        """if "video_position_ids" in batch and "image_position_ids" not in batch:
            batch["image_position_ids"] = batch["video_position_ids"]
        """

        return batch


def make_data_module(
        processor: transformers.ProcessorMixin,
        data_path: str,
        image_folder: ImageFolderInput = None,
        max_seq_length: int = 2304,
        max_decode_frames: int = 8,
        ) -> dict:
    dataset = SupervisedDataset(
            data_path=data_path,
            processor=processor,
            image_folder=image_folder,
            max_seq_length=max_seq_length,
            max_decode_frames=max_decode_frames,
            )
    collator = DataCollatorForSupervisedDataset(
            pad_token_id=processor.tokenizer.pad_token_id
            )
    return dict(
            train_dataset=dataset,
            eval_dataset=None,
            data_collator=collator,
            )
