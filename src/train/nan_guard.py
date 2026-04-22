from __future__ import annotations

import datetime
import json
import os
import os.path as osp
from typing import Any, Dict, Iterable, Optional, Tuple

from accelerate import Accelerator
import torch


def _detach_to_cpu(obj: Any):
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {key: _detach_to_cpu(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_detach_to_cpu(value) for value in obj]
    if isinstance(obj, tuple):
        return tuple(_detach_to_cpu(value) for value in obj)
    return obj


def collect_nonfinite_named_tensors(
    named_tensors: Iterable[Tuple[str, Optional[torch.Tensor]]],
    max_items: int = 64,
) -> Dict[str, Any]:
    items = []
    total_tensors = 0
    total_values = 0
    for name, tensor in named_tensors:
        if tensor is None:
            continue
        detached = tensor.detach()
        nonfinite_mask = ~torch.isfinite(detached)
        nonfinite_count = int(nonfinite_mask.sum().item())
        if nonfinite_count <= 0:
            continue
        total_tensors += 1
        total_values += nonfinite_count
        if len(items) < max_items:
            items.append(
                {
                    "name": name,
                    "shape": list(detached.shape),
                    "dtype": str(detached.dtype),
                    "nonfinite_count": nonfinite_count,
                }
            )
    return {
        "num_nonfinite_tensors": total_tensors,
        "num_nonfinite_values": total_values,
        "items": items,
    }


def save_nonfinite_step_artifacts(
    accelerator: Accelerator,
    output_dir: str,
    step: int,
    trigger_type: str,
    loss: torch.Tensor,
    batch: Dict[str, Any],
    trans_2d_mat: torch.Tensor,
    output_state: Dict[str, Any],
    local_triggered: bool,
    local_nonfinite_summary: Optional[Dict[str, Any]] = None,
):
    device = accelerator.device
    local_nonfinite = int(bool(local_triggered))
    local_rank_code = torch.tensor(
        [accelerator.process_index if local_nonfinite else -1],
        device=device,
        dtype=torch.int32,
    )
    gathered_rank_codes = accelerator.gather(local_rank_code)
    offending_ranks = sorted(
        int(rank) for rank in gathered_rank_codes.detach().cpu().tolist() if int(rank) >= 0
    )

    save_root = osp.join(output_dir, "nonfinite_stop", f"step-{step:06d}")
    model_state_dir = osp.join(save_root, "model_state")

    if accelerator.is_main_process:
        os.makedirs(save_root, exist_ok=True)
        summary = {
            "step": int(step),
            "trigger_type": trigger_type,
            "timestamp": datetime.datetime.now().isoformat(),
            "offending_ranks": offending_ranks,
            "world_size": int(accelerator.num_processes),
            "loss_is_finite": bool(torch.isfinite(loss.detach()).all().item()),
            "loss_repr": str(loss.detach().float().cpu()),
            "model_state_dir": model_state_dir,
        }
        with open(osp.join(save_root, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    accelerator.wait_for_everyone()

    local_payload = {
        "step": int(step),
        "process_index": int(accelerator.process_index),
        "world_size": int(accelerator.num_processes),
        "trigger_type": trigger_type,
        "local_nonfinite_loss": bool(local_nonfinite),
        "loss_repr": str(loss.detach().float().cpu()),
        "local_nonfinite_summary": local_nonfinite_summary or {},
        "batch": _detach_to_cpu(batch),
        "trans_2d_mat": _detach_to_cpu(trans_2d_mat),
        "state": _detach_to_cpu(output_state),
    }
    torch.save(local_payload, osp.join(save_root, f"rank{accelerator.process_index:02d}_batch.pt"))

    accelerator.wait_for_everyone()
    accelerator.save_state(model_state_dir)
    accelerator.wait_for_everyone()
