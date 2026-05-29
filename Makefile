SHELL := /usr/bin/bash

SESSION_PREFIX ?= csvit2
GPU_IDS ?= 0,1,2,3
NUM_PROCESSES ?= 4
MAIN_PROCESS_PORT ?= 0
DRY_RUN ?= 0
STAGE1_WEIGHT ?=
RUN_NAME ?=
OVERRIDES ?=
CONFIG_NAME ?=
STAGE1_DEFAULT_OVERRIDES ?= TRAIN.sample_per_device=42 LOSS.heatmap_sigma=4.0
STAGE2_DEFAULT_OVERRIDES ?= TRAIN.sample_per_device=6 LOSS.heatmap_sigma=4.0
DINO_STAGE1_LARGE_DEFAULT_OVERRIDES ?= LOSS.heatmap_sigma=4.0
DINO_STAGE1_LARGE_TI_DEFAULT_OVERRIDES ?= LOSS.heatmap_sigma=4.0 MODEL.ti.enabled=true
DINO_STAGE1_DEFAULT_OVERRIDES ?= LOSS.heatmap_sigma=4.0
DINO_STAGE2_DEFAULT_OVERRIDES ?= LOSS.heatmap_sigma=4.0

.DEFAULT_GOAL := shell

.PHONY: shell menu help train-stage1 train-stage2 train-stage1-dinov3-large train-stage1-dinov3-large-ti train-stage2-dinov3-large train-stage1-dinov3 train-stage2-dinov3 attach-stage1 attach-stage2 attach-stage1-dinov3-large attach-stage1-dinov3-large-ti attach-stage2-dinov3-large attach-stage1-dinov3 attach-stage2-dinov3 stop-stage1 stop-stage2 stop-stage1-dinov3-large stop-stage1-dinov3-large-ti stop-stage2-dinov3-large stop-stage1-dinov3 stop-stage2-dinov3 logs-stage1 logs-stage2 logs-stage1-dinov3-large logs-stage1-dinov3-large-ti logs-stage2-dinov3-large logs-stage1-dinov3 logs-stage2-dinov3 tmux-ls

shell:
	@python3 script/train_tui.py

menu: shell

help:
	@printf '%s\n' \
	"CS-ViT2-remake training entrypoints" \
	"" \
	"Targets:" \
	"  make | make shell" \
	"    Launch the interactive training TUI." \
	"  make train-stage1" \
	"    Launch stage1 training in a detached tmux session." \
	"  make train-stage2 STAGE1_WEIGHT=/path/to/stage1/best_model" \
	"    Launch stage2 training in a detached tmux session. STAGE1_WEIGHT is required." \
	"  make train-stage1-dinov3-large" \
	"    Launch DINOv3-L/16 stage1 training with config/stage1_dinov3_large.yaml." \
	"  make train-stage1-dinov3-large-ti" \
	"    Launch DINOv3-L/16 stage1 training with TI enabled." \
	"  make train-stage2-dinov3-large STAGE1_WEIGHT=/path/to/dinov3_large_stage1/best_model" \
	"    Launch DINOv3-L/16 stage2 training with config/stage2_dinov3_large.yaml." \
	"  make train-stage1-dinov3" \
	"    Launch DINOv3-H+/16 stage1 training with config/stage1_dinov3.yaml." \
	"  make train-stage2-dinov3 STAGE1_WEIGHT=/path/to/dinov3_stage1/best_model" \
	"    Launch DINOv3-H+/16 stage2 training with config/stage2_dinov3.yaml." \
	"  make attach-stage1 | make attach-stage2" \
	"    Attach to the running tmux session for that stage." \
	"  make logs-stage1 | make logs-stage2" \
	"    Tail the current tmux log file for that stage." \
	"  make stop-stage1 | make stop-stage2" \
	"    Kill the tmux session for that stage." \
	"  make tmux-ls" \
	"    List active tmux sessions." \
	"" \
	"Make variables (override with VAR=value):" \
	"  SESSION_PREFIX=$(SESSION_PREFIX)" \
	"    Prefix used to build tmux session names: <prefix>-stage1 / <prefix>-stage2." \
	"    Options: any short ASCII token, e.g. csvit2, exp42, ablation-a." \
	"  GPU_IDS=$(GPU_IDS)" \
	"    Comma-separated GPU ids passed to accelerate --gpu_ids." \
	"    Options: 0 | 0,1 | 4,5,6,7 | any visible CUDA device list." \
	"  NUM_PROCESSES=$(NUM_PROCESSES)" \
	"    Number of accelerate worker processes." \
	"    Options: positive integer; usually match the number of GPU ids." \
	"  MAIN_PROCESS_PORT=$(MAIN_PROCESS_PORT)" \
	"    Distributed port passed to accelerate." \
	"    Options: 0 for auto-pick, or a fixed free port such as 29501." \
	"  RUN_NAME=$(if $(RUN_NAME),$(RUN_NAME),<auto>)" \
	"    Explicit run name reused by checkpoint dir, tmux log path, and SwanLab experiment." \
	"    A YYYY-MM-DD prefix is always added unless the name already starts with one." \
	"    Options: empty for auto-generated, or a slug-like name such as stage1-baseline." \
	"  DRY_RUN=$(DRY_RUN)" \
	"    Print the resolved command without starting tmux." \
	"    Options: 0 | 1." \
	"  STAGE1_WEIGHT=$(if $(STAGE1_WEIGHT),$(STAGE1_WEIGHT),<required-for-stage2>)" \
	"    Stage1 checkpoint input for stage2." \
	"    Options: /path/to/best_model directory or /path/to/model.safetensors." \
	"  OVERRIDES=$(if $(OVERRIDES),$(OVERRIDES),<none>)" \
	"    Extra Hydra overrides appended after the default stage overrides." \
	"    Options: any valid Hydra KEY=VALUE sequence inside one quoted string." \
	"  CONFIG_NAME=$(if $(CONFIG_NAME),$(CONFIG_NAME),<target-default>)" \
	"    Hydra config name passed to script/run_train_tmux.sh." \
	"    Options: stage1 | stage2 | stage1_dinov3_large | stage2_dinov3_large | stage1_dinov3 | stage2_dinov3." \
	"" \
	"Default Hydra overrides applied by these targets:" \
	"  train-stage1: $(STAGE1_DEFAULT_OVERRIDES)" \
	"  train-stage2: $(STAGE2_DEFAULT_OVERRIDES)" \
	"  train-stage1-dinov3-large: $(DINO_STAGE1_LARGE_DEFAULT_OVERRIDES)" \
	"  train-stage1-dinov3-large-ti: $(DINO_STAGE1_LARGE_TI_DEFAULT_OVERRIDES)" \
	"  train-stage2-dinov3-large: $(DINO_STAGE2_DEFAULT_OVERRIDES)" \
	"  train-stage1-dinov3: $(DINO_STAGE1_DEFAULT_OVERRIDES)" \
	"  train-stage2-dinov3: $(DINO_STAGE2_DEFAULT_OVERRIDES)" \
	"" \
	"Common OVERRIDES examples:" \
	"  OVERRIDES=\"TRAIN.sample_per_device=32\"" \
	"  OVERRIDES=\"LOSS.heatmap_sigma=4.0 GENERAL.description=stage1-baseline\"" \
	"  OVERRIDES=\"TRAIN.lr=5e-5 TRAIN.backbone_lr=5e-6\"" \
	"  OVERRIDES=\"GENERAL.resume_path=/path/to/checkpoint-5000\"" \
	"  OVERRIDES=\"MODEL.backbone.backbone_str=model/facebook/dinov2-base\"" \
	"  OVERRIDES=\"DATA.train.groups.ego.weight=0.3 DATA.train.groups.aux.weight=0.7\"" \
	"  OVERRIDES=\"DATA.train.groups.ego.datasets.HOT3D=0.6 DATA.train.groups.ego.datasets.AssemblyHands=0.4\"" \
	"" \
	"Examples:" \
	"  make train-stage1" \
	"  make train-stage1 GPU_IDS=0,1 NUM_PROCESSES=2" \
	"  make train-stage1 RUN_NAME=stage1-ablation-a OVERRIDES=\"TRAIN.sample_per_device=32\"" \
	"  make train-stage1 DRY_RUN=1" \
	"  make train-stage2 STAGE1_WEIGHT=/path/to/stage1/best_model RUN_NAME=stage2-baseline" \
	"  make train-stage1-dinov3-large GPU_IDS=0,1 NUM_PROCESSES=2 DRY_RUN=1" \
	"  make train-stage1-dinov3-large-ti GPU_IDS=0,1 NUM_PROCESSES=2 DRY_RUN=1" \
	"  make train-stage2-dinov3-large STAGE1_WEIGHT=/path/to/dinov3_large_stage1/best_model" \
	"  make train-stage1-dinov3 GPU_IDS=0,1,2,3 NUM_PROCESSES=4" \
	"  make logs-stage1"

train-stage1:
	@SESSION_NAME="$(SESSION_PREFIX)-stage1" \
	CONFIG_NAME="$(if $(CONFIG_NAME),$(CONFIG_NAME),stage1)" \
	GPU_IDS="$(GPU_IDS)" \
	NUM_PROCESSES="$(NUM_PROCESSES)" \
	MAIN_PROCESS_PORT="$(MAIN_PROCESS_PORT)" \
	RUN_NAME="$(RUN_NAME)" \
	DRY_RUN="$(DRY_RUN)" \
	bash script/run_train_tmux.sh stage1 $(STAGE1_DEFAULT_OVERRIDES) $(OVERRIDES)

train-stage2:
	@SESSION_NAME="$(SESSION_PREFIX)-stage2" \
	CONFIG_NAME="$(if $(CONFIG_NAME),$(CONFIG_NAME),stage2)" \
	GPU_IDS="$(GPU_IDS)" \
	NUM_PROCESSES="$(NUM_PROCESSES)" \
	MAIN_PROCESS_PORT="$(MAIN_PROCESS_PORT)" \
	STAGE1_WEIGHT="$(STAGE1_WEIGHT)" \
	RUN_NAME="$(RUN_NAME)" \
	DRY_RUN="$(DRY_RUN)" \
	bash script/run_train_tmux.sh stage2 $(STAGE2_DEFAULT_OVERRIDES) $(OVERRIDES)

train-stage1-dinov3-large:
	@SESSION_NAME="$(SESSION_PREFIX)-stage1-dinov3-large" \
	CONFIG_NAME="$(if $(CONFIG_NAME),$(CONFIG_NAME),stage1_dinov3_large)" \
	GPU_IDS="$(GPU_IDS)" \
	NUM_PROCESSES="$(NUM_PROCESSES)" \
	MAIN_PROCESS_PORT="$(MAIN_PROCESS_PORT)" \
	RUN_NAME="$(RUN_NAME)" \
	DRY_RUN="$(DRY_RUN)" \
	bash script/run_train_tmux.sh stage1 $(DINO_STAGE1_LARGE_DEFAULT_OVERRIDES) $(OVERRIDES)

train-stage1-dinov3-large-ti:
	@SESSION_NAME="$(SESSION_PREFIX)-stage1-dinov3-large-ti" \
	CONFIG_NAME="$(if $(CONFIG_NAME),$(CONFIG_NAME),stage1_dinov3_large)" \
	GPU_IDS="$(GPU_IDS)" \
	NUM_PROCESSES="$(NUM_PROCESSES)" \
	MAIN_PROCESS_PORT="$(MAIN_PROCESS_PORT)" \
	RUN_NAME="$(RUN_NAME)" \
	DRY_RUN="$(DRY_RUN)" \
	bash script/run_train_tmux.sh stage1 $(DINO_STAGE1_LARGE_TI_DEFAULT_OVERRIDES) $(OVERRIDES)

train-stage2-dinov3-large:
	@SESSION_NAME="$(SESSION_PREFIX)-stage2-dinov3-large" \
	CONFIG_NAME="$(if $(CONFIG_NAME),$(CONFIG_NAME),stage2_dinov3_large)" \
	GPU_IDS="$(GPU_IDS)" \
	NUM_PROCESSES="$(NUM_PROCESSES)" \
	MAIN_PROCESS_PORT="$(MAIN_PROCESS_PORT)" \
	STAGE1_WEIGHT="$(STAGE1_WEIGHT)" \
	RUN_NAME="$(RUN_NAME)" \
	DRY_RUN="$(DRY_RUN)" \
	bash script/run_train_tmux.sh stage2 $(DINO_STAGE2_DEFAULT_OVERRIDES) $(OVERRIDES)

train-stage1-dinov3:
	@SESSION_NAME="$(SESSION_PREFIX)-stage1-dinov3" \
	CONFIG_NAME="$(if $(CONFIG_NAME),$(CONFIG_NAME),stage1_dinov3)" \
	GPU_IDS="$(GPU_IDS)" \
	NUM_PROCESSES="$(NUM_PROCESSES)" \
	MAIN_PROCESS_PORT="$(MAIN_PROCESS_PORT)" \
	RUN_NAME="$(RUN_NAME)" \
	DRY_RUN="$(DRY_RUN)" \
	bash script/run_train_tmux.sh stage1 $(DINO_STAGE1_DEFAULT_OVERRIDES) $(OVERRIDES)

train-stage2-dinov3:
	@SESSION_NAME="$(SESSION_PREFIX)-stage2-dinov3" \
	CONFIG_NAME="$(if $(CONFIG_NAME),$(CONFIG_NAME),stage2_dinov3)" \
	GPU_IDS="$(GPU_IDS)" \
	NUM_PROCESSES="$(NUM_PROCESSES)" \
	MAIN_PROCESS_PORT="$(MAIN_PROCESS_PORT)" \
	STAGE1_WEIGHT="$(STAGE1_WEIGHT)" \
	RUN_NAME="$(RUN_NAME)" \
	DRY_RUN="$(DRY_RUN)" \
	bash script/run_train_tmux.sh stage2 $(DINO_STAGE2_DEFAULT_OVERRIDES) $(OVERRIDES)

attach-stage1:
	@tmux attach -t "$(SESSION_PREFIX)-stage1"

attach-stage2:
	@tmux attach -t "$(SESSION_PREFIX)-stage2"

attach-stage1-dinov3-large:
	@tmux attach -t "$(SESSION_PREFIX)-stage1-dinov3-large"

attach-stage1-dinov3-large-ti:
	@tmux attach -t "$(SESSION_PREFIX)-stage1-dinov3-large-ti"

attach-stage2-dinov3-large:
	@tmux attach -t "$(SESSION_PREFIX)-stage2-dinov3-large"

attach-stage1-dinov3:
	@tmux attach -t "$(SESSION_PREFIX)-stage1-dinov3"

attach-stage2-dinov3:
	@tmux attach -t "$(SESSION_PREFIX)-stage2-dinov3"

stop-stage1:
	@tmux kill-session -t "$(SESSION_PREFIX)-stage1"

stop-stage2:
	@tmux kill-session -t "$(SESSION_PREFIX)-stage2"

stop-stage1-dinov3-large:
	@tmux kill-session -t "$(SESSION_PREFIX)-stage1-dinov3-large"

stop-stage1-dinov3-large-ti:
	@tmux kill-session -t "$(SESSION_PREFIX)-stage1-dinov3-large-ti"

stop-stage2-dinov3-large:
	@tmux kill-session -t "$(SESSION_PREFIX)-stage2-dinov3-large"

stop-stage1-dinov3:
	@tmux kill-session -t "$(SESSION_PREFIX)-stage1-dinov3"

stop-stage2-dinov3:
	@tmux kill-session -t "$(SESSION_PREFIX)-stage2-dinov3"

logs-stage1:
	@LOG_FILE="$$(tmux show-environment -t "$(SESSION_PREFIX)-stage1" CSVIT2_LOG_FILE 2>/dev/null | sed 's/^CSVIT2_LOG_FILE=//')"; \
	if [[ -z "$$LOG_FILE" ]]; then \
		echo "No active session $(SESSION_PREFIX)-stage1 or log file metadata missing."; \
		exit 1; \
	fi; \
	tail -n 200 -f "$$LOG_FILE"

logs-stage2:
	@LOG_FILE="$$(tmux show-environment -t "$(SESSION_PREFIX)-stage2" CSVIT2_LOG_FILE 2>/dev/null | sed 's/^CSVIT2_LOG_FILE=//')"; \
	if [[ -z "$$LOG_FILE" ]]; then \
		echo "No active session $(SESSION_PREFIX)-stage2 or log file metadata missing."; \
		exit 1; \
	fi; \
	tail -n 200 -f "$$LOG_FILE"

logs-stage1-dinov3-large:
	@LOG_FILE="$$(tmux show-environment -t "$(SESSION_PREFIX)-stage1-dinov3-large" CSVIT2_LOG_FILE 2>/dev/null | sed 's/^CSVIT2_LOG_FILE=//')"; \
	if [[ -z "$$LOG_FILE" ]]; then \
		echo "No active session $(SESSION_PREFIX)-stage1-dinov3-large or log file metadata missing."; \
		exit 1; \
	fi; \
	tail -n 200 -f "$$LOG_FILE"

logs-stage1-dinov3-large-ti:
	@LOG_FILE="$$(tmux show-environment -t "$(SESSION_PREFIX)-stage1-dinov3-large-ti" CSVIT2_LOG_FILE 2>/dev/null | sed 's/^CSVIT2_LOG_FILE=//')"; \
	if [[ -z "$$LOG_FILE" ]]; then \
		echo "No active session $(SESSION_PREFIX)-stage1-dinov3-large-ti or log file metadata missing."; \
		exit 1; \
	fi; \
	tail -n 200 -f "$$LOG_FILE"

logs-stage2-dinov3-large:
	@LOG_FILE="$$(tmux show-environment -t "$(SESSION_PREFIX)-stage2-dinov3-large" CSVIT2_LOG_FILE 2>/dev/null | sed 's/^CSVIT2_LOG_FILE=//')"; \
	if [[ -z "$$LOG_FILE" ]]; then \
		echo "No active session $(SESSION_PREFIX)-stage2-dinov3-large or log file metadata missing."; \
		exit 1; \
	fi; \
	tail -n 200 -f "$$LOG_FILE"

logs-stage1-dinov3:
	@LOG_FILE="$$(tmux show-environment -t "$(SESSION_PREFIX)-stage1-dinov3" CSVIT2_LOG_FILE 2>/dev/null | sed 's/^CSVIT2_LOG_FILE=//')"; \
	if [[ -z "$$LOG_FILE" ]]; then \
		echo "No active session $(SESSION_PREFIX)-stage1-dinov3 or log file metadata missing."; \
		exit 1; \
	fi; \
	tail -n 200 -f "$$LOG_FILE"

logs-stage2-dinov3:
	@LOG_FILE="$$(tmux show-environment -t "$(SESSION_PREFIX)-stage2-dinov3" CSVIT2_LOG_FILE 2>/dev/null | sed 's/^CSVIT2_LOG_FILE=//')"; \
	if [[ -z "$$LOG_FILE" ]]; then \
		echo "No active session $(SESSION_PREFIX)-stage2-dinov3 or log file metadata missing."; \
		exit 1; \
	fi; \
	tail -n 200 -f "$$LOG_FILE"

tmux-ls:
	@tmux ls || true
