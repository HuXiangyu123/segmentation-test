import re
from typing import Any, Optional

import torch


def normalize_case_id(value: Any) -> str:
    """Match the patient-id normalization used by generate_embeddings.py."""
    if value is None:
        return ""
    match = re.search(r"\d+", str(value))
    return match.group() if match else str(value).strip()


class CaseTextEmbeddingStore:
    """Lookup table for per-case text embeddings saved by generate_embeddings.py."""

    def __init__(self, embedding_path: str):
        data = torch.load(embedding_path, map_location="cpu")
        if not isinstance(data, dict) or "embeddings" not in data or "id_map" not in data:
            raise ValueError(
                f"Case text embedding file must contain 'embeddings' and 'id_map': {embedding_path}"
            )

        self.lookup = {}
        embeddings = data["embeddings"]
        id_map = data["id_map"]

        if embeddings.ndim == 3:
            if len(embeddings) != len(id_map):
                raise ValueError(
                    f"Embedding count ({len(embeddings)}) does not match id_map ({len(id_map)})."
                )
            modality_order = [
                str(modality).upper()
                for modality in data.get("modality_order", ["MR", "CT"])
            ]
            if embeddings.shape[1] != len(modality_order):
                raise ValueError(
                    "3D case text embeddings must align with modality_order: "
                    f"shape={tuple(embeddings.shape)}, modality_order={modality_order}"
                )

            for emb_pair, meta in zip(embeddings, id_map):
                patient_id = self._extract_patient_id(meta)
                if not patient_id:
                    continue
                for modality, emb in zip(modality_order, emb_pair):
                    self.lookup[(patient_id, modality)] = emb.float().cpu()
            return

        if len(embeddings) != len(id_map):
            raise ValueError(
                f"Embedding count ({len(embeddings)}) does not match id_map ({len(id_map)})."
            )

        for emb, meta in zip(embeddings, id_map):
            patient_id = self._extract_patient_id(meta)
            modality = ""
            if isinstance(meta, dict):
                modality = str(meta.get("modality", "")).upper()
            if not patient_id or not modality:
                continue
            self.lookup[(patient_id, modality)] = emb.float().cpu()

    @staticmethod
    def _extract_patient_id(meta: Any) -> str:
        if isinstance(meta, dict):
            for key in ("patient_id", "reg_id", "case_id", "case_key"):
                value = meta.get(key)
                patient_id = normalize_case_id(value)
                if patient_id:
                    return patient_id
            return ""
        return normalize_case_id(meta)

    def get(self, patient_id: Any, modality: str) -> Optional[torch.Tensor]:
        key = (normalize_case_id(patient_id), str(modality).upper())
        return self.lookup.get(key)


def _to_str_list(values: Any) -> list[str]:
    if isinstance(values, torch.Tensor):
        return [str(v.item()) for v in values]
    if isinstance(values, (list, tuple)):
        return [str(v) for v in values]
    return [str(values)]


def get_case_text_embedding_from_batch(
    batch: dict,
    store: Optional[CaseTextEmbeddingStore],
    device: torch.device,
    modality: Optional[str] = None,
) -> Optional[torch.Tensor]:
    """
    Build a [B, 512] tensor for a batch.

    Falls back to None when any sample in the batch is missing a text embedding,
    so the model will use static class embeddings for that batch.
    """
    if store is None:
        return None

    patient_ids = batch.get("patient_id", batch.get("name"))
    if patient_ids is None:
        return None

    batch_modality = modality
    if batch_modality is None:
        batch_modality = batch.get("modality")
        if isinstance(batch_modality, (list, tuple)):
            batch_modality = batch_modality[0]

    if batch_modality is None:
        return None

    embeddings = []
    for patient_id in _to_str_list(patient_ids):
        emb = store.get(patient_id, str(batch_modality))
        if emb is None:
            return None
        embeddings.append(emb)

    if not embeddings:
        return None
    return torch.stack(embeddings, dim=0).to(device)
