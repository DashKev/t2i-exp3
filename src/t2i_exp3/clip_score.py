from __future__ import annotations

from pathlib import Path


class ClipImageTextScorer:
    """Pinned local CLIP cosine scorer shared by online and post-hoc plugins."""

    def __init__(self, model_path: Path, device: str):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        if not (model_path / "config.json").is_file():
            raise ValueError(f"CLIP 快照不完整：{model_path}")
        self.model_path = model_path
        self.device = device
        self._torch = torch
        dtype = torch.float32 if str(device) == "cpu" else torch.float16
        self._model = CLIPModel.from_pretrained(
            str(model_path), local_files_only=True, torch_dtype=dtype
        ).to(device)
        self._model.eval()
        self._processor = CLIPProcessor.from_pretrained(
            str(model_path), local_files_only=True, use_fast=False
        )

    def score(self, image, text: str) -> float:
        inputs = self._processor(
            text=[text], images=[image], return_tensors="pt", padding=True, truncation=True
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self._torch.inference_mode():
            outputs = self._model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pixel_values=inputs["pixel_values"],
            )
            image_features = outputs.image_embeds
            text_features = outputs.text_embeds
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            return float((image_features @ text_features.T).item())
