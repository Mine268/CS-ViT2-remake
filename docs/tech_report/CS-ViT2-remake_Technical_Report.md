# CS-ViT2-remake 技术报告（完整版）

> **项目**: 精简版 CS-ViT2 重构仓库 — 基于 ViT + MANO 的 3D 手部姿态估计
> **最佳模型**: Stage1 step 95000, `val/micro_rte_ego = 18.07 mm`, `val/micro_mpjpe_ego = 26.06 mm`
> **最后更新**: 2026-05-18

---

## 关于本报告

本报告面向接手该项目的工程师，旨在提供一份可以独立阅读理解全部实现细节的技术文档。报告采用多文件结构，各章节深度覆盖对应模块的**所有函数、参数、数据流、设计决策和工程问题**。

## 章节目录

| 章节 | 文件 | 内容简介 |
|------|------|----------|
| 1 | [01_data_pipeline.md](01_data_pipeline.md) | 数据管线：WebDataset 加载、两级采样、clip-native 导出、schema 标准化、过滤策略、数据源路由 |
| 2 | [02_data_augmentation.md](02_data_augmentation.md) | 数据增强：bbox jitter（约束性/传统）、3D 几何增强（旋转/缩放/透视）、像素级增强、透视归一化 |
| 3 | [03_model_architecture.md](03_model_architecture.md) | 模型架构：ViT Backbone、透视信息嵌入器、MANO Transformer Decoder、Camera Head（Patch UV + Rho Multibin）、几何先验、时序编码器、MANO 手部模型 |
| 4 | [04_loss_and_supervision.md](04_loss_and_supervision.md) | 损失函数与监督路由：ego/aux 分离、8 个损失分量详解、几何先验的 multibin 编解码、dropout 调度 |
| 5 | [05_training_framework.md](05_training_framework.md) | 训练框架：Hydra 配置系统、两阶段训练策略、total_samples→total_step 换算、LR 调度、混合精度、checkpoint 管理、non-finite guard、SwanLab 集成、tmux 后台训练 |
| 6 | [06_inference_and_demo.md](06_inference_and_demo.md) | 推理与 Demo：手部检测器（WiLoR/MediaPipe/GT）、预处理、模型前向、输出格式、wds/image/video 多输入类型支持 |
| 7 | [07_engineering_notes.md](07_engineering_notes.md) | 工程问题与解决方案：clip-native 迁移（10.4x 加速）、OmegaConf struct mode、分布式进程管理、SwanLab 调试污染、torch.compile 失败、AssemblyHands test 占位值、Stage2 数据集排除、多卡验证均衡 |
| 8 | [08_config_reference.md](08_config_reference.md) | 配置参考：所有 Hydra 配置文件的完整参数表、Stage1/Stage2 差异对照、make 变量和命令行覆盖 |

## 项目概览

### 目标

从单张 RGB 图像或视频序列中恢复 3D 手部姿态：MANO 姿态参数（16 个关节的轴角旋转）、形状参数（10 维 PCA 系数）、相机空间中的 3D 平移，最终输出 21 个 3D 关节坐标和 778 个顶点坐标。

### 核心技术栈

| 层次 | 技术选择 |
|------|----------|
| Backbone | DINOv2-large (ViT, patch=14, hidden=1024) |
| 手部模型 | MANO (右手, 10 PCA 形状系数)，通过 SMPL-X 加载 |
| 姿态表示 | Axis-angle (3 维/关节), 也支持 6D rotation 和 quaternion |
| Camera Head | Patch UV heatmap (softargmax) + Rho multibin (分类+残差) |
| 几何先验 | Root-Z/Rho 显式先验: `k × sqrt(fx·fy) / sqrt(bbox_w·bbox_h)` |
| 数据格式 | WebDataset V2 (tar archives), clip-native 预切分 |
| 训练框架 | PyTorch 2.9 + HuggingFace Accelerate + DDP |
| 配置管理 | Hydra 1.3 + OmegaConf 2.3 |
| 实验追踪 | SwanLab (cloud), 同步 console 输出 + tmux log |
| 代码质量 | Ruff format + lint (line_length=100) |
| 包管理 | uv + Python 3.12 |

### 关键数值

| 指标 | 值 |
|------|-----|
| 图像尺寸 | 224×224 |
| Patch 尺寸 | 14 (DINOv2) |
| 可见 patch 数 | 16×16 = 256 |
| MANO 关节数 | 16 (运动链) |
| 输出关节数 | 21 (含指尖) |
| MANO 顶点数 | 778 |
| 形状参数维度 | 10 |
| 坐标单位 | 毫米 (mm) |
| 坐标系统 | 右手系 (x 右, y 下, z 前) |

### 训练数据统计

| 组 | 数据集 | Stage1 | Stage2 |
|----|--------|--------|--------|
| ego | HOT3D | ✓ | ✓ |
| ego | AssemblyHands | ✓ | ✓ |
| aux | InterHand2.6M | ✓ | ✓ |
| aux | MTC | ✓ | ✓ |
| aux | DexYCB | ✓ | ✓ |
| aux | HO3D_v3 | ✓ | ✓ |
| aux | FreiHAND | ✓ | ✗ (无 T≥7 clips) |
| aux | RHD | ✓ | ✗ (无 T≥7 clips) |

### 文件结构

```
CS-ViT2-remake/
├── config/                    # Hydra 配置（6 个 yaml）
├── src/
│   ├── data/                  # 数据加载、预处理、schema、采样、导出
│   ├── model/                 # backbone、heads、temporal、loss、root_z、net
│   ├── train/                 # engine、checkpoint、tracker、nan_guard
│   └── utils/                 # mano、metric、proj、rot、vis、misc、train_utils
├── script/                    # train.py、test.py、demo_stage1.py、export 脚本
├── hand_bbox_module/          # WiLoR 手部检测器
├── model/                     # 预训练权重、MANO 资产
├── tests/                     # 12 个 pytest 测试文件
├── docs/                      # 文档（含本报告）
├── Makefile                   # 训练入口 + tmux 管理
└── pyproject.toml
```
