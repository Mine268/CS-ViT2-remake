#!/usr/bin/env python3

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SESSION_PREFIX = os.environ.get("SESSION_PREFIX", "csvit2")
DEFAULT_GPU_IDS = os.environ.get("GPU_IDS", "0,1,2,3")
DEFAULT_NUM_PROCESSES = os.environ.get("NUM_PROCESSES", "4")
DEFAULT_MAIN_PROCESS_PORT = os.environ.get("MAIN_PROCESS_PORT", "0")
DEFAULT_RUN_NAME = os.environ.get("RUN_NAME", "")
DEFAULT_DRY_RUN = os.environ.get("DRY_RUN", "0") == "1"

STAGE_OPTIONS = [
    ("stage1", "Stage1"),
    ("stage2", "Stage2"),
]

BACKBONE_OPTIONS = [
    ("baseline", "DINOv2-large"),
    ("dinov3_large", "DINOv3-L/16"),
    ("dinov3_huge", "DINOv3-H+/16"),
]


@dataclass(frozen=True)
class LaunchPreset:
    stage: str
    backbone: str
    ti_enabled: bool
    target: str
    config_name: str
    session_suffix: str
    default_batch: int
    default_heatmap_sigma: float = 4.0
    ti_via_target: bool = False


@dataclass
class LaunchSettings:
    stage: str
    backbone: str
    ti_enabled: bool
    session_prefix: str = DEFAULT_SESSION_PREFIX
    gpu_ids: str = DEFAULT_GPU_IDS
    num_processes: int = int(DEFAULT_NUM_PROCESSES)
    main_process_port: str = DEFAULT_MAIN_PROCESS_PORT
    run_name: str = DEFAULT_RUN_NAME
    batch_size: int = 1
    heatmap_sigma: float = 4.0
    stage1_weight: str = ""
    extra_overrides: str = ""
    dry_run: bool = DEFAULT_DRY_RUN


def resolve_preset(stage: str, backbone: str, ti_enabled: bool) -> LaunchPreset:
    if stage not in {"stage1", "stage2"}:
        raise ValueError(f"Unsupported stage: {stage}")
    if backbone not in {"baseline", "dinov3_large", "dinov3_huge"}:
        raise ValueError(f"Unsupported backbone: {backbone}")
    if stage == "stage2" and ti_enabled:
        raise ValueError("TI is currently only supported for stage1.")

    if stage == "stage1" and backbone == "baseline":
        return LaunchPreset(
            stage=stage,
            backbone=backbone,
            ti_enabled=ti_enabled,
            target="train-stage1",
            config_name="stage1",
            session_suffix="stage1",
            default_batch=42,
        )
    if stage == "stage2" and backbone == "baseline":
        return LaunchPreset(
            stage=stage,
            backbone=backbone,
            ti_enabled=False,
            target="train-stage2",
            config_name="stage2",
            session_suffix="stage2",
            default_batch=6,
        )
    if stage == "stage1" and backbone == "dinov3_large" and ti_enabled:
        return LaunchPreset(
            stage=stage,
            backbone=backbone,
            ti_enabled=True,
            target="train-stage1-dinov3-large-ti",
            config_name="stage1_dinov3_large",
            session_suffix="stage1-dinov3-large-ti",
            default_batch=42,
            ti_via_target=True,
        )
    if stage == "stage1" and backbone == "dinov3_large":
        return LaunchPreset(
            stage=stage,
            backbone=backbone,
            ti_enabled=False,
            target="train-stage1-dinov3-large",
            config_name="stage1_dinov3_large",
            session_suffix="stage1-dinov3-large",
            default_batch=42,
        )
    if stage == "stage2" and backbone == "dinov3_large":
        return LaunchPreset(
            stage=stage,
            backbone=backbone,
            ti_enabled=False,
            target="train-stage2-dinov3-large",
            config_name="stage2_dinov3_large",
            session_suffix="stage2-dinov3-large",
            default_batch=1,
        )
    if stage == "stage1" and backbone == "dinov3_huge":
        return LaunchPreset(
            stage=stage,
            backbone=backbone,
            ti_enabled=ti_enabled,
            target="train-stage1-dinov3",
            config_name="stage1_dinov3",
            session_suffix="stage1-dinov3",
            default_batch=2,
        )
    return LaunchPreset(
        stage=stage,
        backbone=backbone,
        ti_enabled=False,
        target="train-stage2-dinov3",
        config_name="stage2_dinov3",
        session_suffix="stage2-dinov3",
        default_batch=1,
    )


def infer_num_processes(gpu_ids: str) -> int:
    gpu_list = [item.strip() for item in gpu_ids.split(",") if item.strip()]
    return max(1, len(gpu_list))


def format_float(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text or "0"


def build_override_tokens(settings: LaunchSettings, preset: LaunchPreset) -> list[str]:
    tokens: list[str] = []
    if settings.batch_size != preset.default_batch:
        tokens.append(f"TRAIN.sample_per_device={settings.batch_size}")
    if settings.heatmap_sigma != preset.default_heatmap_sigma:
        tokens.append(f"LOSS.heatmap_sigma={format_float(settings.heatmap_sigma)}")
    if settings.ti_enabled and not preset.ti_via_target:
        tokens.append("MODEL.ti.enabled=true")
    return tokens


def build_override_value(settings: LaunchSettings, preset: LaunchPreset) -> str:
    tokens = build_override_tokens(settings, preset)
    extra = settings.extra_overrides.strip()
    if extra:
        tokens.append(extra)
    return " ".join(tokens)


def build_make_command(settings: LaunchSettings, preset: LaunchPreset) -> list[str]:
    command = [
        "make",
        "--no-print-directory",
        preset.target,
        f"SESSION_PREFIX={settings.session_prefix}",
        f"GPU_IDS={settings.gpu_ids}",
        f"NUM_PROCESSES={settings.num_processes}",
        f"MAIN_PROCESS_PORT={settings.main_process_port}",
        f"DRY_RUN={1 if settings.dry_run else 0}",
    ]
    if settings.run_name.strip():
        command.append(f"RUN_NAME={settings.run_name.strip()}")
    override_value = build_override_value(settings, preset)
    if override_value:
        command.append(f"OVERRIDES={override_value}")
    if preset.stage == "stage2":
        if not settings.stage1_weight.strip():
            raise ValueError("STAGE1_WEIGHT is required for stage2.")
        command.append(f"STAGE1_WEIGHT={settings.stage1_weight.strip()}")
    return command


def default_settings_for_preset(preset: LaunchPreset) -> LaunchSettings:
    return LaunchSettings(
        stage=preset.stage,
        backbone=preset.backbone,
        ti_enabled=preset.ti_enabled,
        batch_size=preset.default_batch,
        heatmap_sigma=preset.default_heatmap_sigma,
        num_processes=int(DEFAULT_NUM_PROCESSES),
    )


def prompt_menu(title: str, options: list[str], default_index: int = 0) -> int:
    while True:
        print(f"\n{title}")
        for index, option in enumerate(options, start=1):
            marker = " [default]" if index - 1 == default_index else ""
            print(f"  {index}) {option}{marker}")
        raw = input(f"Choose [1-{len(options)}]: ").strip()
        if not raw:
            return default_index
        if raw.isdigit():
            selected = int(raw) - 1
            if 0 <= selected < len(options):
                return selected
        print("Invalid selection.")


def prompt_text(label: str, default: str = "", allow_empty: bool = True) -> str:
    while True:
        suffix = f" [{default}]" if default else ""
        value = input(f"{label}{suffix}: ").strip()
        if value:
            return value
        if default:
            return default
        if allow_empty:
            return ""
        print("This value is required.")


def prompt_int_menu(label: str, current: int, options: list[int]) -> int:
    unique = []
    for value in [current, *options]:
        if value not in unique:
            unique.append(value)
    menu = [
        f"Keep current ({current})",
        *[str(value) for value in unique if value != current],
        "Custom",
    ]
    choice = prompt_menu(label, menu, default_index=0)
    if choice == 0:
        return current
    if choice == len(menu) - 1:
        while True:
            raw = prompt_text("Custom integer value", allow_empty=False)
            try:
                return int(raw)
            except ValueError:
                print("Please enter a valid integer.")
    return int(menu[choice])


def prompt_float_menu(label: str, current: float, options: list[float]) -> float:
    unique = []
    for value in [current, *options]:
        if value not in unique:
            unique.append(value)
    menu = [
        f"Keep current ({format_float(current)})",
        *[format_float(value) for value in unique if value != current],
        "Custom",
    ]
    choice = prompt_menu(label, menu, default_index=0)
    if choice == 0:
        return current
    if choice == len(menu) - 1:
        while True:
            raw = prompt_text("Custom float value", allow_empty=False)
            try:
                return float(raw)
            except ValueError:
                print("Please enter a valid float.")
    return float(menu[choice])


def prompt_string_menu(label: str, current: str, options: list[str]) -> str:
    unique = []
    for value in [current, *options]:
        if value and value not in unique:
            unique.append(value)
    keep_label = f"Keep current ({current})" if current else "Keep current (empty)"
    menu = [keep_label, *[value for value in unique if value != current], "Custom"]
    choice = prompt_menu(label, menu, default_index=0)
    if choice == 0:
        return current
    if choice == len(menu) - 1:
        return prompt_text("Custom value", default=current)
    return menu[choice]


def prompt_yes_no(label: str, default: bool) -> bool:
    menu = [
        "Yes",
        "No",
    ]
    choice = prompt_menu(label, menu, default_index=0 if default else 1)
    return choice == 0


def choose_stage(current: str = "stage1") -> str:
    labels = [label for _, label in STAGE_OPTIONS]
    default_index = 0 if current == "stage1" else 1
    return STAGE_OPTIONS[prompt_menu("Choose stage", labels, default_index)][0]


def choose_backbone(current: str = "baseline") -> str:
    labels = [label for _, label in BACKBONE_OPTIONS]
    current_index = [key for key, _ in BACKBONE_OPTIONS].index(current)
    return BACKBONE_OPTIONS[prompt_menu("Choose backbone", labels, current_index)][0]


def choose_ti(stage: str, current: bool = False) -> bool:
    if stage == "stage2":
        print("\nTI is currently only available for stage1. Keeping TI disabled.")
        return False
    options = [
        "Disable TI",
        "Enable TI",
    ]
    return prompt_menu("Enable TI?", options, 1 if current else 0) == 1


def print_launch_summary(settings: LaunchSettings, preset: LaunchPreset) -> None:
    override_value = build_override_value(settings, preset) or "<none>"
    session_name = f"{settings.session_prefix}-{preset.session_suffix}"
    print("\nCurrent launch plan")
    print(f"  stage            : {settings.stage}")
    print(f"  backbone         : {settings.backbone}")
    print(f"  ti_enabled       : {settings.ti_enabled}")
    print(f"  make target      : {preset.target}")
    print(f"  hydra config     : {preset.config_name}")
    print(f"  session name     : {session_name}")
    print(f"  GPU_IDS          : {settings.gpu_ids}")
    print(f"  NUM_PROCESSES    : {settings.num_processes}")
    print(f"  MAIN_PROCESS_PORT: {settings.main_process_port}")
    print(f"  sample/device    : {settings.batch_size}")
    print(f"  heatmap_sigma    : {format_float(settings.heatmap_sigma)}")
    print(f"  RUN_NAME         : {settings.run_name or '<auto>'}")
    if settings.stage == "stage2":
        print(f"  STAGE1_WEIGHT    : {settings.stage1_weight or '<required>'}")
    print(f"  extra overrides  : {settings.extra_overrides or '<none>'}")
    print(f"  merged overrides : {override_value}")
    print(f"  dry run          : {settings.dry_run}")


def sync_settings_with_preset(settings: LaunchSettings, reset_common_fields: bool) -> LaunchPreset:
    preset = resolve_preset(settings.stage, settings.backbone, settings.ti_enabled)
    if reset_common_fields:
        settings.batch_size = preset.default_batch
        settings.heatmap_sigma = preset.default_heatmap_sigma
    if preset.stage == "stage1":
        settings.stage1_weight = ""
    return preset


def launch_training_interactive() -> None:
    stage = choose_stage()
    backbone = choose_backbone()
    ti_enabled = choose_ti(stage, current=False)
    preset = resolve_preset(stage, backbone, ti_enabled)
    settings = default_settings_for_preset(preset)

    while True:
        preset = sync_settings_with_preset(settings, reset_common_fields=False)
        print_launch_summary(settings, preset)
        options = [
            "Change stage",
            "Change backbone",
            "Change TI",
            "Set GPU_IDS",
            "Set NUM_PROCESSES",
            "Set sample_per_device",
            "Set heatmap_sigma",
            "Set SESSION_PREFIX",
            "Set RUN_NAME",
            "Set MAIN_PROCESS_PORT",
            "Set extra OVERRIDES",
            "Toggle DRY_RUN",
        ]
        if settings.stage == "stage2":
            options.append("Set STAGE1_WEIGHT")
        options.extend(
            [
                "Preview full command",
                "Launch training",
                "Cancel",
            ]
        )
        choice = prompt_menu("Launch menu", options, default_index=len(options) - 3)

        if choice == 0:
            settings.stage = choose_stage(settings.stage)
            settings.ti_enabled = settings.ti_enabled and settings.stage == "stage1"
            preset = sync_settings_with_preset(settings, reset_common_fields=True)
        elif choice == 1:
            settings.backbone = choose_backbone(settings.backbone)
            preset = sync_settings_with_preset(settings, reset_common_fields=True)
        elif choice == 2:
            settings.ti_enabled = choose_ti(settings.stage, settings.ti_enabled)
            preset = sync_settings_with_preset(settings, reset_common_fields=True)
        elif choice == 3:
            settings.gpu_ids = prompt_string_menu(
                "Choose GPU_IDS",
                settings.gpu_ids,
                ["0", "0,1", "0,1,2,3", "4,5,6,7"],
            )
            inferred = infer_num_processes(settings.gpu_ids)
            if prompt_yes_no(
                f"Set NUM_PROCESSES to match GPU count ({inferred})?",
                settings.num_processes == inferred,
            ):
                settings.num_processes = inferred
        elif choice == 4:
            settings.num_processes = prompt_int_menu(
                "Choose NUM_PROCESSES",
                settings.num_processes,
                [infer_num_processes(settings.gpu_ids), 1, 2, 4, 8],
            )
        elif choice == 5:
            settings.batch_size = prompt_int_menu(
                "Choose TRAIN.sample_per_device",
                settings.batch_size,
                [1, 2, 4, 6, 8, 16, 32, 42],
            )
        elif choice == 6:
            settings.heatmap_sigma = prompt_float_menu(
                "Choose LOSS.heatmap_sigma",
                settings.heatmap_sigma,
                [2.0, 3.0, 4.0, 5.0],
            )
        elif choice == 7:
            settings.session_prefix = prompt_text(
                "SESSION_PREFIX", settings.session_prefix, allow_empty=False
            )
        elif choice == 8:
            settings.run_name = prompt_text("RUN_NAME", settings.run_name, allow_empty=True)
        elif choice == 9:
            settings.main_process_port = prompt_string_menu(
                "Choose MAIN_PROCESS_PORT",
                settings.main_process_port,
                ["0", "29500", "29501", "29502"],
            )
        elif choice == 10:
            settings.extra_overrides = prompt_text(
                "Extra OVERRIDES",
                settings.extra_overrides,
                allow_empty=True,
            )
        elif choice == 11:
            settings.dry_run = not settings.dry_run
        elif settings.stage == "stage2" and choice == 12:
            settings.stage1_weight = prompt_text(
                "STAGE1_WEIGHT",
                settings.stage1_weight,
                allow_empty=False,
            )
        else:
            tail_offset = 13 if settings.stage == "stage2" else 12
            if choice == tail_offset:
                try:
                    command = build_make_command(settings, preset)
                except ValueError as exc:
                    print(f"\n{exc}")
                    continue
                print("\nResolved command")
                print(f"  {shlex.join(command)}")
            elif choice == tail_offset + 1:
                try:
                    command = build_make_command(settings, preset)
                except ValueError as exc:
                    print(f"\n{exc}")
                    continue
                print("\nLaunching")
                print(f"  {shlex.join(command)}")
                if not prompt_yes_no("Execute this command now?", default=True):
                    print("Cancelled.")
                    continue
                subprocess.run(command, cwd=ROOT_DIR, check=False)
                return
            else:
                print("Cancelled.")
                return


def run_tmux_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=ROOT_DIR,
        check=False,
        capture_output=True,
        text=True,
    )


def list_sessions(filter_text: str = "") -> list[str]:
    result = run_tmux_command(["tmux", "ls", "-F", "#S"])
    if result.returncode != 0:
        return []
    sessions = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if filter_text:
        sessions = [session for session in sessions if filter_text in session]
    return sessions


def get_session_log_file(session_name: str) -> str:
    result = run_tmux_command(["tmux", "show-environment", "-t", session_name, "CSVIT2_LOG_FILE"])
    if result.returncode != 0:
        return ""
    for line in result.stdout.splitlines():
        if line.startswith("CSVIT2_LOG_FILE="):
            return line.split("=", 1)[1].strip()
    return ""


def choose_session(action: str) -> str:
    filter_text = prompt_text("Session name filter", DEFAULT_SESSION_PREFIX, allow_empty=True)
    sessions = list_sessions(filter_text)
    if not sessions:
        print(f"No tmux sessions matched filter: {filter_text or '<none>'}")
        return ""
    index = prompt_menu(f"Choose session to {action}", sessions, default_index=0)
    return sessions[index]


def list_sessions_interactive() -> None:
    filter_text = prompt_text("Session name filter", DEFAULT_SESSION_PREFIX, allow_empty=True)
    sessions = list_sessions(filter_text)
    if not sessions:
        print(f"No tmux sessions matched filter: {filter_text or '<none>'}")
        return
    print("\nActive sessions")
    print(f"{'SESSION':<36}  LOG FILE")
    print(f"{'-' * 36}  {'-' * 40}")
    for session_name in sessions:
        print(f"{session_name:<36}  {get_session_log_file(session_name) or '<missing>'}")


def attach_session_interactive() -> None:
    session_name = choose_session("attach")
    if not session_name:
        return
    subprocess.run(["tmux", "attach", "-t", session_name], cwd=ROOT_DIR, check=False)


def tail_log_interactive() -> None:
    session_name = choose_session("tail")
    if not session_name:
        return
    log_file = get_session_log_file(session_name)
    if not log_file:
        print(f"Session {session_name} does not expose CSVIT2_LOG_FILE.")
        return
    subprocess.run(["tail", "-n", "200", "-f", log_file], cwd=ROOT_DIR, check=False)


def stop_session_interactive() -> None:
    session_name = choose_session("stop")
    if not session_name:
        return
    if not prompt_yes_no(f"Stop session {session_name}?", default=False):
        print("Cancelled.")
        return
    subprocess.run(["tmux", "kill-session", "-t", session_name], cwd=ROOT_DIR, check=False)


def print_help() -> None:
    print(
        "\n".join(
            [
                "CS-ViT2 training TUI",
                "",
                "Entry:",
                "  make",
                "  make shell",
                "  make menu",
                "  python3 script/train_tui.py",
                "",
                "Flow:",
                "  1. Choose stage",
                "  2. Choose backbone",
                "  3. Choose whether to enable TI",
                "  4. Adjust common launch parameters",
                "  5. Confirm and launch",
                "",
                "The TUI reuses the existing explicit Make targets internally.",
            ]
        )
    )


def main() -> int:
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        print_help()
        return 0

    if not sys.stdin.isatty() or not sys.stdout.isatty():
        print(
            "Interactive TUI requires a TTY. Use `make train-stage1` or `make help` in non-interactive mode."
        )
        return 0

    while True:
        options = [
            "Launch training",
            "Attach to tmux session",
            "Tail tmux log",
            "Stop tmux session",
            "List tmux sessions",
            "Help",
            "Exit",
        ]
        choice = prompt_menu("CS-ViT2 training TUI", options, default_index=0)
        if choice == 0:
            launch_training_interactive()
        elif choice == 1:
            attach_session_interactive()
        elif choice == 2:
            tail_log_interactive()
        elif choice == 3:
            stop_session_interactive()
        elif choice == 4:
            list_sessions_interactive()
        elif choice == 5:
            print_help()
        else:
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
