from __future__ import annotations

"""Training loop, dataloader construction, validation, and runtime orchestration."""

import datetime
import os
import os.path as osp
import sys
import threading
from typing import Dict, Iterable, Optional, Tuple

from accelerate import Accelerator, DataLoaderConfiguration, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import broadcast_object_list, set_seed
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
import torch
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

from ..data.preprocess import PixelLevelAugmentation, preprocess_batch
from ..data.config import build_train_data_plan, collect_supervision_dataset_groups
from ..data.sampler import (
    build_clip_sample_filter_fn,
)
from ..data.wds import (
    build_balanced_clip_segments,
    equalize_rank_clip_segments,
    estimate_wds_shard_clip_counts,
    get_dataloader,
    get_group_reweight_precut_clip_dataloader,
    get_segmented_wds_dataloader,
)
from ..model.net import PoseNet
from ..utils.metric import StreamingMetricMeter
from ..utils.metric import build_dataset_group_mask
from ..utils.misc import expand_glob_patterns
from ..utils.train_utils import get_progressive_dropout
from ..utils.vis import vis
from .checkpoint import (
    build_run_dir,
    load_best_metric_info,
    manage_checkpoints,
    save_best_model_variant,
    save_config_snapshot,
)
from .nan_guard import collect_nonfinite_named_tensors, save_nonfinite_step_artifacts
from .tracker import Tracker


logger = get_logger(__name__)


class StepTimeoutTerminator:
    """Hard-stop the current process if a long-running train/validation stage exceeds a deadline."""

    def __init__(self, timeout_seconds: int, label: str):
        self.timeout_seconds = int(timeout_seconds)
        self.label = str(label)
        self._cancel_event = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self):
        if self.timeout_seconds <= 0:
            return self

        def _watchdog():
            if self._cancel_event.wait(timeout=float(self.timeout_seconds)):
                return
            sys.stderr.write(
                f"\n[timeout] {self.label} exceeded {self.timeout_seconds}s; terminating process.\n"
            )
            sys.stderr.flush()
            os._exit(124)

        self._thread = threading.Thread(
            target=_watchdog,
            name=f"timeout-watchdog-{self.label}",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._cancel_event.set()
        return False


def create_accelerator(cfg: DictConfig) -> Accelerator:
    """Construct the project-standard Accelerate runtime."""
    return Accelerator(
        mixed_precision=cfg.TRAIN.mixed_precision,
        gradient_accumulation_steps=cfg.TRAIN.grad_accum_step,
        # Our WebDataset batches contain string metadata such as `__key__` and `data_source`.
        # Let each process fetch its own iterable batch instead of having rank0 concatenate them.
        dataloader_config=DataLoaderConfiguration(dispatch_batches=False),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)],
    )


def build_train_dataloader(cfg: DictConfig):
    """Build the config-driven two-level ego/aux training dataloader from clip-native shards."""
    train_filter = build_clip_sample_filter_fn(cfg.DATA.train.get("filter", {}))
    train_plan = build_train_data_plan(cfg.DATA)
    return get_group_reweight_precut_clip_dataloader(
        group_dataset_sources=train_plan.group_dataset_sources,
        group_dataset_weights=train_plan.group_dataset_weights,
        group_weights=train_plan.group_weights,
        batch_size=cfg.TRAIN.sample_per_device,
        num_workers=cfg.GENERAL.num_worker,
        prefetch_factor=cfg.GENERAL.prefetch_factor,
        infinite=True,
        seed=cfg.GENERAL.seed,
        shardshuffle=cfg.DATA.train.get("shardshuffle", 64),
        post_clip_shuffle=cfg.DATA.train.get("post_clip_shuffle", 64),
        default_source_split=cfg.DATA.train.get("split", "train"),
        data_source_alias_map=train_plan.dataset_alias_map,
        sample_filter=train_filter,
    )


def build_eval_dataloader(
    source_patterns,
    cfg_split,
    num_frames: int,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    seed: Optional[int],
    accelerator: Optional[Accelerator] = None,
    infinite: bool = True,
):
    """
    Build the evaluation dataloader for validation or test.

    Unlike training, evaluation reads an explicit source list and can optionally split work into
    balanced shard segments for full evaluation across multiple ranks.
    """
    sources = expand_glob_patterns([str(x) for x in source_patterns])
    if len(sources) == 0:
        return None

    eval_filter = build_clip_sample_filter_fn(cfg_split.get("filter", {}))
    sampling_cfg = cfg_split.get("sampling", {})
    if accelerator is not None and accelerator.num_processes > 1 and not infinite:
        # Finite multi-process evaluation must keep every rank on the same number of iterations,
        # otherwise later `gather_for_metrics` collectives can deadlock after one rank exhausts
        # its iterable earlier than the others.
        shard_clip_counts_obj = [None]
        if accelerator.is_main_process:
            shard_clip_counts_obj[0] = estimate_wds_shard_clip_counts(
                urls=sources,
                num_frames=num_frames,
                stride=cfg_split.stride,
            )
        broadcast_object_list(shard_clip_counts_obj, from_process=0)
        rank_segments = build_balanced_clip_segments(
            urls=sources,
            clip_counts=shard_clip_counts_obj[0],
            num_parts=accelerator.num_processes,
        )
        rank_segments = equalize_rank_clip_segments(rank_segments)
        return get_segmented_wds_dataloader(
            segments=rank_segments[accelerator.process_index],
            num_frames=num_frames,
            stride=cfg_split.stride,
            batch_size=batch_size,
            num_workers=0,
            prefetch_factor=prefetch_factor,
        )

    return get_dataloader(
        url=sources,
        num_frames=num_frames,
        stride=cfg_split.stride,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        infinite=infinite,
        seed=seed,
        clip_sampling_mode=sampling_cfg.get("mode", "dense"),
        clips_per_sequence=sampling_cfg.get("clips_per_sequence", None),
        shardshuffle=sampling_cfg.get("shardshuffle", False),
        post_clip_shuffle=sampling_cfg.get("post_clip_shuffle", 0),
        sample_filter=eval_filter,
    )


def setup_model(cfg: DictConfig) -> PoseNet:
    """Instantiate `PoseNet` from the resolved Hydra config."""
    supervision_groups = collect_supervision_dataset_groups(cfg.DATA)
    return PoseNet(
        stage=cfg.MODEL.stage,
        stage1_weight_path=cfg.MODEL.get("stage1_weight"),
        backbone_str=cfg.MODEL.backbone.backbone_str,
        img_size=cfg.MODEL.img_size,
        img_mean=cfg.MODEL.img_mean,
        img_std=cfg.MODEL.img_std,
        infusion_feats_lyr=cfg.MODEL.backbone.infusion_layer,
        drop_cls=cfg.MODEL.backbone.drop_cls,
        backbone_kwargs=cfg.MODEL.backbone.get("kwargs"),
        num_handec_layer=cfg.MODEL.handec.num_layer,
        num_handec_head=cfg.MODEL.handec.num_head,
        ndim_handec_mlp=cfg.MODEL.handec.dim_mlp,
        ndim_handec_head=cfg.MODEL.handec.dim_head,
        prob_handec_dropout=cfg.MODEL.handec.dropout,
        prob_handec_emb_dropout=0.0,
        handec_emb_dropout_type="drop",
        handec_norm=cfg.MODEL.handec.norm,
        ndim_handec_norm_cond_dim=-1,
        ndim_handec_ctx=cfg.MODEL.handec.context_dim,
        handec_skip_token_embed=cfg.MODEL.handec.skip_token_embed,
        handec_mean_init=cfg.MODEL.handec.use_mean_init,
        handec_denorm_output=cfg.MODEL.handec.denorm_output,
        handec_heatmap_resulotion=cfg.MODEL.handec.heatmap_resolution,
        pie_type=cfg.MODEL.persp_info_embed.type,
        num_pie_sample=cfg.MODEL.persp_info_embed.num_sample,
        pie_fusion=cfg.MODEL.persp_info_embed.pie_fusion,
        num_temporal_head=cfg.MODEL.temporal_encoder.num_head,
        num_temporal_layer=cfg.MODEL.temporal_encoder.num_layer,
        trope_scalar=cfg.MODEL.temporal_encoder.trope_scalar,
        zero_linear=cfg.MODEL.temporal_encoder.zero_linear,
        joint_rep_type=cfg.MODEL.joint_type,
        freeze_backbone=cfg.TRAIN.backbone_lr is None,
        norm_by_hand=cfg.MODEL.norm_by_hand,
        handec_cam_head_type=cfg.MODEL.handec.cam_head_type,
        root_z_num_bins=cfg.MODEL.handec.root_z.num_bins,
        root_z_d_min=cfg.MODEL.handec.root_z.d_min,
        root_z_d_max=cfg.MODEL.handec.root_z.d_max,
        root_z_prior_k=cfg.MODEL.handec.root_z.prior_k,
        root_z_geom_hidden_dim=cfg.MODEL.handec.root_z.geom_hidden_dim,
        root_z_dropout=cfg.MODEL.handec.root_z.dropout,
        root_z_use_data_source_embed=cfg.MODEL.handec.root_z.use_data_source_embed,
        lambda_theta=cfg.LOSS.lambda_theta,
        lambda_shape=cfg.LOSS.lambda_shape,
        lambda_trans=cfg.LOSS.lambda_trans,
        lambda_rel=cfg.LOSS.lambda_rel,
        lambda_img=cfg.LOSS.lambda_img,
        lambda_uv_patch=cfg.LOSS.lambda_uv_patch,
        lambda_root_z_cls=cfg.LOSS.lambda_root_z_cls,
        lambda_root_z_res=cfg.LOSS.lambda_root_z_res,
        hm_sigma=cfg.LOSS.heatmap_sigma,
        pred_joint_z_min_mm=cfg.LOSS.pred_joint_z_min_mm,
        reproj_loss_type=cfg.LOSS.reproj_loss_type,
        reproj_loss_delta=cfg.LOSS.reproj_loss_delta,
        ego_datasets=list(supervision_groups["ego"]),
        aux_datasets=list(supervision_groups["aux"]),
        root_min_valid_joints_2d=cfg.LOSS.root_filter.min_valid_joints_2d,
        root_min_hand_bbox_edge_px=cfg.LOSS.root_filter.min_hand_bbox_edge_px,
    )


@torch.no_grad()
def validate(
    cfg: DictConfig,
    accelerator: Accelerator,
    net: torch.nn.Module,
    val_loader: Optional[Iterable],
    global_step: int,
    tracker: Tracker,
    max_eval_steps: Optional[int] = None,
):
    """Run one validation sweep and return aggregated metric values."""
    if val_loader is None:
        return {}
    net.eval()
    meter = StreamingMetricMeter()
    eval_step_cap = int(max_eval_steps if max_eval_steps is not None else cfg.DATA.val.max_val_step)

    for val_step, batch_origin in enumerate(val_loader):
        if eval_step_cap > 0 and val_step >= eval_step_cap:
            break
        batch, _, _ = preprocess_batch(
            batch_origin=batch_origin,
            patch_size=[cfg.MODEL.img_size, cfg.MODEL.img_size],
            patch_expanstion=cfg.TRAIN.expansion_ratio,
            scale_z_range=[1.0, 1.0],
            scale_f_range=[1.0, 1.0],
            persp_rot_max=0.0,
            joint_rep_type=cfg.MODEL.joint_type,
            augmentation_flag=False,
            device=accelerator.device,
            pixel_aug=None,
            perspective_normalization=cfg.TRAIN.get("perspective_normalization", False),
        )
        output_state = net(batch)
        result = output_state["result"]
        local_ego_mask = build_dataset_group_mask(
            batch.get("data_source"),
            accelerator.unwrap_model(net).loss_fn.ego_datasets,
            device=batch["has_mano"].device,
            dtype=batch["has_mano"].dtype,
        )
        local_aux_mask = build_dataset_group_mask(
            batch.get("data_source"),
            accelerator.unwrap_model(net).loss_fn.aux_datasets,
            device=batch["has_mano"].device,
            dtype=batch["has_mano"].dtype,
        )
        joint_cam_gt = accelerator.gather_for_metrics(batch["joint_cam"][:, -1:])
        joint_cam_pred = accelerator.gather_for_metrics(result["joint_cam_pred"][:, -1:])
        verts_cam_gt = accelerator.gather_for_metrics(result["verts_cam_gt"][:, -1:])
        verts_cam_pred = accelerator.gather_for_metrics(result["verts_cam_pred"][:, -1:])
        has_mano = accelerator.gather_for_metrics(batch["has_mano"][:, -1:])
        joint_3d_valid = accelerator.gather_for_metrics(batch["joint_3d_valid"][:, -1:])
        ego_mask = accelerator.gather_for_metrics(local_ego_mask)
        aux_mask = accelerator.gather_for_metrics(local_aux_mask)
        joint_rel_gt = joint_cam_gt - joint_cam_gt[:, :, :1]
        joint_rel_pred = joint_cam_pred - joint_cam_pred[:, :, :1]
        verts_rel_gt = verts_cam_gt - verts_cam_gt[:, :, :1]
        verts_rel_pred = verts_cam_pred - verts_cam_pred[:, :, :1]
        norm_valid = torch.ones_like(has_mano)
        meter.update(
            joint_cam_gt,
            joint_rel_gt,
            verts_cam_gt,
            verts_rel_gt,
            joint_cam_pred,
            joint_rel_pred,
            verts_cam_pred,
            verts_rel_pred,
            has_mano,
            joint_3d_valid,
            norm_valid,
            ego_mask,
            aux_mask,
        )

    metrics = meter.compute()
    tracker.log_scalars(metrics, step=global_step, split="val")
    net.train()
    return metrics


def get_shared_eval_max_steps(
    val_loader: Optional[Iterable],
    cfg_split,
    accelerator: Optional[Accelerator] = None,
) -> int:
    """
    Resolve a safe validation-step cap shared by every rank.

    When an eval iterable exposes `__len__`, we clamp to the minimum number of local batches across
    ranks. This is primarily a safety net; the segmented multi-rank validation path should already
    keep rank lengths aligned, but this avoids future deadlocks if a regression reintroduces a
    mismatch.
    """
    cfg_cap = int(cfg_split.get("max_val_step", 0))
    if val_loader is None:
        return cfg_cap

    try:
        local_steps = int(len(val_loader))
    except TypeError:
        return cfg_cap

    shared_steps = local_steps
    if accelerator is not None and accelerator.num_processes > 1:
        local_tensor = torch.tensor([local_steps], device=accelerator.device, dtype=torch.int64)
        gathered = accelerator.gather(local_tensor).detach().cpu().tolist()
        shared_steps = min(int(x) for x in gathered)
        if accelerator.is_main_process and len(set(int(x) for x in gathered)) > 1:
            logger.warning(
                "Validation dataloader lengths differ across ranks; truncating to %s shared steps from %s",
                shared_steps,
                gathered,
            )

    if cfg_cap > 0:
        return min(shared_steps, cfg_cap)
    return shared_steps


def train(cfg: DictConfig):
    """
    Main training loop.

    Responsibilities:
    - build runtime components (accelerator, model, loaders, optimizer, scheduler, tracker)
    - run step-wise training with non-finite guards
    - emit periodic logs, images, checkpoints, and validation metrics
    """
    accelerator = create_accelerator(cfg)
    set_seed(cfg.GENERAL.seed)

    config_name = HydraConfig.get().job.config_name or cfg.MODEL.stage
    output_dir = build_run_dir(cfg.GENERAL.description)
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        save_config_snapshot(cfg, output_dir, config_name)

    tracker = Tracker(cfg, accelerator, experiment_name=osp.basename(osp.normpath(output_dir)))
    train_loader = build_train_dataloader(cfg)
    val_loader = build_eval_dataloader(
        source_patterns=cfg.DATA.val.source,
        cfg_split=cfg.DATA.val,
        num_frames=cfg.MODEL.num_frame,
        batch_size=int(cfg.DATA.val.get("batch_size", cfg.TRAIN.sample_per_device)),
        num_workers=int(cfg.DATA.val.get("num_workers", 1)),
        prefetch_factor=cfg.GENERAL.prefetch_factor,
        seed=cfg.GENERAL.val_seed,
        accelerator=accelerator,
        infinite=False,
    )
    val_max_steps = get_shared_eval_max_steps(val_loader, cfg.DATA.val, accelerator=accelerator)
    net = setup_model(cfg)

    optimizer = AdamW(
        net.get_optim_param_dict(cfg.TRAIN.lr, cfg.TRAIN.backbone_lr),
        weight_decay=cfg.TRAIN.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.GENERAL.warmup_step,
        num_training_steps=cfg.GENERAL.total_step,
        num_cycles=cfg.GENERAL.cosine_cycle,
    )

    pixel_aug = PixelLevelAugmentation(cfg.TRAIN.get("augmentation"))

    # Prepare all stateful objects through Accelerate together so distributed wrapping stays aligned.
    prepared = [net, optimizer, train_loader, scheduler]
    if val_loader is not None:
        prepared.append(val_loader)
    prepared = accelerator.prepare(*prepared)
    if val_loader is None:
        net, optimizer, train_loader, scheduler = prepared
    else:
        net, optimizer, train_loader, scheduler, val_loader = prepared

    if cfg.GENERAL.resume_path:
        accelerator.load_state(cfg.GENERAL.resume_path)

    best_info = load_best_metric_info(output_dir, "micro_rte_ego", "best_model.json")
    global_step = 0
    train_iter = iter(train_loader)
    net.train()

    while global_step < int(cfg.GENERAL.total_step):
        batch_origin = next(train_iter)
        dropout_rate = get_progressive_dropout(
            step=global_step,
            total_steps=cfg.GENERAL.total_step,
            warmup_steps=cfg.GENERAL.dropout_warmup_step,
            target_dropout=cfg.MODEL.handec.dropout,
        )
        accelerator.unwrap_model(net).set_dropout_rate(dropout_rate)

        batch, trans_2d_mat, _ = preprocess_batch(
            batch_origin=batch_origin,
            patch_size=[cfg.MODEL.img_size, cfg.MODEL.img_size],
            patch_expanstion=cfg.TRAIN.expansion_ratio,
            scale_z_range=cfg.TRAIN.scale_z_range,
            scale_f_range=cfg.TRAIN.scale_f_range,
            persp_rot_max=cfg.TRAIN.persp_rot_max,
            joint_rep_type=cfg.MODEL.joint_type,
            augmentation_flag=True,
            device=accelerator.device,
            pixel_aug=pixel_aug,
            perspective_normalization=cfg.TRAIN.get("perspective_normalization", False),
        )

        with accelerator.accumulate(net):
            output_state = net(batch)
            loss = output_state["loss"]

            local_nonfinite_loss = int(not bool(torch.isfinite(loss.detach()).all().item()))
            global_nonfinite_loss = accelerator.reduce(
                torch.tensor([float(local_nonfinite_loss)], device=accelerator.device),
                reduction="sum",
            )
            if float(global_nonfinite_loss.item()) > 0.0:
                save_nonfinite_step_artifacts(
                    accelerator=accelerator,
                    output_dir=output_dir,
                    step=global_step,
                    trigger_type="forward_loss",
                    loss=loss,
                    batch=batch,
                    trans_2d_mat=trans_2d_mat,
                    output_state=output_state,
                    local_triggered=bool(local_nonfinite_loss),
                    local_nonfinite_summary={},
                )
                raise RuntimeError("Detected non-finite loss before backward")

            accelerator.backward(loss)

            grad_nonfinite_summary = collect_nonfinite_named_tensors(
                (
                    (name, param.grad)
                    for name, param in accelerator.unwrap_model(net).named_parameters()
                )
            )
            local_nonfinite_grad = int(grad_nonfinite_summary["num_nonfinite_tensors"] > 0)
            global_nonfinite_grad = accelerator.reduce(
                torch.tensor([float(local_nonfinite_grad)], device=accelerator.device),
                reduction="sum",
            )
            if float(global_nonfinite_grad.item()) > 0.0:
                save_nonfinite_step_artifacts(
                    accelerator=accelerator,
                    output_dir=output_dir,
                    step=global_step,
                    trigger_type="backward_grad",
                    loss=loss,
                    batch=batch,
                    trans_2d_mat=trans_2d_mat,
                    output_state=output_state,
                    local_triggered=bool(local_nonfinite_grad),
                    local_nonfinite_summary=grad_nonfinite_summary,
                )
                raise RuntimeError("Detected non-finite gradients after backward")

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(net.parameters(), cfg.TRAIN.max_grad)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            param_nonfinite_summary = collect_nonfinite_named_tensors(
                accelerator.unwrap_model(net).named_parameters()
            )
            local_nonfinite_param = int(param_nonfinite_summary["num_nonfinite_tensors"] > 0)
            global_nonfinite_param = accelerator.reduce(
                torch.tensor([float(local_nonfinite_param)], device=accelerator.device),
                reduction="sum",
            )
            if float(global_nonfinite_param.item()) > 0.0:
                save_nonfinite_step_artifacts(
                    accelerator=accelerator,
                    output_dir=output_dir,
                    step=global_step,
                    trigger_type="post_step_param",
                    loss=loss,
                    batch=batch,
                    trans_2d_mat=trans_2d_mat,
                    output_state=output_state,
                    local_triggered=bool(local_nonfinite_param),
                    local_nonfinite_summary=param_nonfinite_summary,
                )
                raise RuntimeError("Detected non-finite parameters after optimizer step")

        if accelerator.sync_gradients:
            global_step += 1
            state = output_state["state"]
            log_payload = {
                "loss_total": loss.detach(),
                **state,
                "lr": scheduler.get_last_lr()[0],
                "dropout_rate": dropout_rate,
            }
            if global_step % int(cfg.GENERAL.log_step) == 0:
                tracker.log_scalars(log_payload, step=global_step, split="train")

            if cfg.TRACKER.log_images and global_step % int(cfg.GENERAL.vis_step) == 0 and accelerator.is_main_process:
                image = vis(batch, trans_2d_mat, output_state["result"], tx=batch["patches"].shape[1] - 1, bx=0)
                tracker.log_image("projection", image, step=global_step, split="train")

            if global_step % int(cfg.GENERAL.checkpoint_step) == 0:
                checkpoint_dir = osp.join(output_dir, "checkpoints", f"checkpoint-{global_step}")
                accelerator.save_state(checkpoint_dir)
                manage_checkpoints(output_dir, keep_last_n=3)
                with StepTimeoutTerminator(
                    timeout_seconds=int(cfg.DATA.val.get("timeout_seconds", 0)),
                    label=f"validation@step{global_step}",
                ):
                    val_metrics = validate(
                        cfg,
                        accelerator,
                        net,
                        val_loader,
                        global_step,
                        tracker,
                        max_eval_steps=val_max_steps,
                    )
                if val_metrics:
                    current_value = float(val_metrics["micro_rte_ego"])
                    if current_value < float(best_info["best_value"]):
                        save_best_model_variant(
                            accelerator=accelerator,
                            output_dir=output_dir,
                            global_step=global_step,
                            val_metrics=val_metrics,
                            config_name=config_name,
                            best_dir_name="best_model",
                            metadata_filename="best_model.json",
                        )
                        best_info = {"best_value": current_value, "step": global_step}

    tracker.finish()
    return output_dir
