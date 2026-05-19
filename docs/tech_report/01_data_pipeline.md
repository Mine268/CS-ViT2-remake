# 第 1 章：数据管线

> 本章覆盖 WebDataset 加载、两级随机采样、clip-native 导出、schema 标准化、过滤策略和数据源路由。

---

## 1.1 概述

CS-ViT2-remake 的数据管线完全基于 WebDataset V2 格式，所有数据以 tar 归档存储。管线经历了从 "sequence 格式" 到 "clip-native 格式" 的重构，加载吞吐获得了 10.4 倍加速。

## 1.2 数据存储格式

### 1.2.1 WebDataset 成员结构

每个样本（clip）在 tar 中包含以下成员（`src/data/export.py:20-34`）：

| 成员命名模式 | 内容 | 格式 |
|-------------|------|------|
| `{key}.imgs_path.json` | 图像路径列表 | JSON list |
| `{key}.handedness.json` | 左右手标签 | JSON scalar |
| `{key}.data_source.json` | 数据集来源 | JSON scalar |
| `{key}.source_split.json` | 数据分割名 | JSON scalar |
| `{key}.intr_type.json` | 内参类型 | JSON scalar |
| `{key}.additional_desc.json` | 额外描述 | JSON list |
| `{key}.source_index.json` | 帧来源索引 | JSON list |
| `{key}.img_bytes.pickle` | 图像原始字节 | Pickle (protocol=4) |
| `{key}.hand_bbox.npy` | 手部 bbox | NumPy (4,) |
| `{key}.joint_img.npy` | 2D 关节坐标 | NumPy (21, 2) |
| `{key}.joint_cam.npy` | 3D 相机坐标 | NumPy (21, 3) |
| `{key}.joint_rel.npy` | 相对根关节坐标 | NumPy (21, 3) |
| `{key}.joint_2d_valid.npy` | 2D 有效性掩码 | NumPy (21,) |
| `{key}.joint_3d_valid.npy` | 3D 有效性掩码 | NumPy (21,) |
| `{key}.has_mano.npy` | MANO 参数有效 | NumPy scalar |
| `{key}.mano_pose.npy` | MANO 姿态 | NumPy (48,) |
| `{key}.mano_shape.npy` | MANO 形状 | NumPy (10,) |
| `{key}.has_intr.npy` | 内参有效 | NumPy scalar |
| `{key}.timestamp.npy` | 时间戳 | NumPy scalar |
| `{key}.focal.npy` | 焦距 | NumPy (2,) |
| `{key}.princpt.npy` | 主点 | NumPy (2,) |

字段分为三类序列化方式（`src/data/export.py:20-34`）：
- `JSON_LIST_FIELDS`: `imgs_path`, `additional_desc`, `source_index` → JSON 列表
- `JSON_SCALAR_FIELDS`: `handedness`, `data_source`, `source_split`, `intr_type` → JSON 标量
- `PICKLE_FIELDS`: `imgs_bytes` → Pickle 序列化
- `NPY_FIELDS`: 所有 `PER_FRAME_ARRAY_SPECS` 中的字段 → NumPy `.npy` 格式

### 1.2.2 V1 vs V2 格式

`src/data/schema.py:10-42` 中定义了两套 numpy key 集合：

**LEGACY_NUMPY_KEYS** (12 keys)：旧格式使用统一的 `joint_valid.npy` 和 `mano_valid.npy`

**V2_NUMPY_KEYS** (16 keys)：新格式新增了 `joint_2d_valid.npy`、`joint_3d_valid.npy`、`has_mano.npy`、`has_intr.npy`，将有效性判断细化，同时保留 `joint_valid.npy` 向后兼容。

## 1.3 Schema 标准化

### 1.3.1 `PER_FRAME_ARRAY_SPECS`

`src/data/schema.py:44-61` 定义了所有逐帧数组字段的规范形状和默认填充值：

| 字段 | 尾形状 | 默认值 | 说明 |
|------|--------|--------|------|
| `hand_bbox` | `(4,)` | 0.0 | xyxy 格式 |
| `joint_img` | `(21, 2)` | 0.0 | 图像坐标 |
| `joint_hand_bbox` | `(21, 2)` | 0.0 | 相对 hand bbox 坐标 |
| `joint_cam` | `(21, 3)` | 0.0 | 相机坐标 |
| `joint_rel` | `(21, 3)` | 0.0 | 根相对坐标 |
| `joint_2d_valid` | `(21,)` | **1.0** | 2D 默认有效 |
| `joint_3d_valid` | `(21,)` | **1.0** | 3D 默认有效 |
| `joint_valid` | `(21,)` | **1.0** | 兼容字段 |
| `mano_pose` | `(48,)` | 0.0 | 16×3 axis-angle |
| `mano_shape` | `(10,)` | 0.0 | 10-dim PCA |
| `has_mano` | `()` | 0.0 | 默认无 MANO |
| `mano_valid` | `()` | 0.0 | 兼容字段 |
| `has_intr` | `()` | 0.0 | 默认无内参 |
| `timestamp` | `()` | 0.0 | 归一化时间 |
| `focal` | `(2,)` | 0.0 | (fx, fy) |
| `princpt` | `(2,)` | 0.0 | (cx, cy) |

注意：有效性相关字段的默认值为 1.0（假设默认有效），而实际数据字段默认值为 0.0。

### 1.3.2 `normalize_decoded_clip_sample`

`src/data/schema.py:236-483` — 这是整个数据管线的中心规范化入口。

**输入**：WebDataset 解码后的原始样本字典

**处理流程**：

1. **推断帧数** (`infer_num_frames`, `src/data/schema.py:164-178`)：按优先级尝试 `img_bytes.pickle` 长度 → `imgs_path.json` 长度 → 遍历 legacy/v2 numpy keys 找第一个匹配数组的 shape[0] → 报错

2. **规范化图像路径** (`src/data/schema.py:254-258`)：若缺失则格式化为 `"{sample_key}::{idx:04d}"`

3. **数据源解析** (`src/data/schema.py:267-283`)：
   - 训练流使用 `force_data_source=True`，数据源名称由配置直接指定
   - 否则从 shard 的 `data_source.json` 读取，回退到 `default_data_source` 或 key-based 推断
   - 始终通过 `canonicalize_data_source_name` 规范化（应用别名映射）

4. **遗留兼容** (`src/data/schema.py:288-417`)：处理 `joint_valid`/`mano_valid` 的缺失、`has_intr` 的默认推断、2D/3D 有效性回退

5. **逐帧数组规范化** (`_coerce_frame_array`, `src/data/schema.py:131-161`)：自动为单帧数据添加 batch 维度、验证形状、缺失时填充默认值

**输出字典**: 24 个键的无歧义规范表示

### 1.3.3 `slice_normalized_clip_sample`

`src/data/schema.py:486-509` — 从多帧 clip 中切片出子 clip。

- `LIST_SLICE_KEYS` (`imgs_path`, `imgs_bytes`, `additional_desc`, `source_index`)：`list[start:end]`
- `SCALAR_COPY_KEYS` (`handedness`, `data_source`, `source_split`, `intr_type`)：逐字复制
- `PER_FRAME_ARRAY_SPECS` 中的所有字段：`ndarray[start:end]`

## 1.4 数据源路由

### 1.4.1 别名规范化

`src/utils/data_source.py:11-14` 中的 `normalize_data_source_alias_key` 将任何数据集名称转换为查找键：
```
"InterHand2.6M" → "interhand26m"
"HOT3D" → "hot3d"
"interhand_26m" → "interhand26m"
```
规则：小写 + 去除非字母数字字符。

### 1.4.2 规范名称映射

`src/data/config.py:38-58` 中的 `build_data_source_alias_map` 从 `DATA.datasets` 配置构建别名→规范名称映射：
1. 每个数据集的规范名称本身就是别名
2. 加上 `aliases` 列表中的额外别名
3. 所有别名通过 `normalize_data_source_alias_key` 标准化
4. 歧义检测：同一标准化键映射到两个规范名时抛异常

### 1.4.3 名称规范化

`src/utils/data_source.py:33-52` 中的 `canonicalize_data_source_name`：
1. 字符串化 + strip
2. 在 alias_map 中查找标准化键
3. 命中则返回规范名称，否则返回原始值

## 1.5 训练数据计划

### 1.5.1 `TrainDataPlan`

`src/data/config.py:19-35` — 冻结数据类，包含完整的训练采样配置：

- `dataset_alias_map: Dict[str, str]` — 别名映射表
- `group_weights: OrderedDict[str, float]` — `{"ego": 0.2, "aux": 0.8}` (归一化后)
- `group_dataset_weights: Dict[str, OrderedDict[str, float]]` — 组内数据集权重 (归一化后)
- `group_dataset_sources: Dict[str, OrderedDict[str, List[str]]]` — 组内数据集的具体文件路径
- `ego_datasets: List[str]` — ego 组成员
- `aux_datasets: List[str]` — aux 组成员

### 1.5.2 `build_train_data_plan`

`src/data/config.py:183-236` — 完整解析 `DATA` 配置为 `TrainDataPlan`：

1. 验证 `DATA.datasets` 非空
2. 收集 ego/aux 组成员（验证无重叠）
3. 读取训练 split 名称（默认 `"train"`）
4. 遍历每个组的每个数据集：
   - 尝试解析 split 对应的 glob patterns
   - **如果数据集没有该 split（如 FreiHAND 无 train_stage2），静默跳过**
   - 存储权重和展开后的文件路径
5. 归一化：每个组内数据集权重和为 1.0，组间权重和为 1.0

### 1.5.3 `collect_supervision_dataset_groups`

`src/data/config.py:104-141` — 仅验证配置结构，不接触文件系统：
- `DATA.datasets` 必须非空
- ego 和 aux 组必须都在配置中声明
- 每个数据集只能属于一个组（重叠检测）
- 引用的数据集必须在 `DATA.datasets` 中存在

## 1.6 WebDataset 加载

### 1.6.1 Sequence → Clip 转换

`src/data/wds.py:60-113` 中的 `clip_to_t_frames` 是将 sequence 格式样本转换为固定长度 clip 的生成器：

1. 对每个 decoded sequence，调用 `normalize_decoded_clip_sample` 规范化
2. 如果总帧数 < `num_frames`，跳过
3. 按照 `sampling_mode` 选择 clip 起始索引：
   - `"dense"`: 所有可能的 clip `[0, total_clips)`，stride 滑窗
   - `"random_clip"`: 随机无重复采样 `clips_per_sequence` 个起始索引
4. 对每个索引起始位置，调用 `slice_normalized_clip_sample`，应用可选的 `sample_filter`
5. RNG 种子偏移 `seed + worker_id`，确保多 worker 不重复

### 1.6.2 Clip 数量估算

`src/data/wds.py:116-135` 中的 `estimate_wds_shard_clip_counts`：
- 只读取 tar 中的 `imgs_path.json` 成员（不解码图像）
- 根据 `count_sample_clips(total_frames, num_frames, stride)` 计算
- 公式：`max(0, (total_frames - num_frames) // stride + 1)`

### 1.6.3 平衡段分配 (平衡评估)

`src/data/wds.py:138-179` 中的 `build_balanced_clip_segments` 将全局 shard clip 范围划分为 per-rank 连续段：

1. 计算所有 shard 的 `total_clips`
2. 计算每个 rank 的全局范围：`[total * rank / parts, total * (rank+1) / parts)`
3. 将全局边界映射回每个 shard 的本地范围
4. 为每个 (rank, shard) 重叠创建 `ClipSegment(tar_path, start_clip, end_clip)`

`src/data/wds.py:218-236` 中的 `equalize_rank_clip_segments` 将所有 rank 修剪到相同 clip 数，防止 NCCL gather 因长度不同死锁。

### 1.6.4 预切分 Clip 加载器

`src/data/wds.py:343-389` 中的 `_build_precut_clip_webdataset` 用于 clip-native 数据（已预切分，无需运行时切片）：

```python
wds.WebDataset(urls, resampled=infinite, seed=seed)
  .shuffle(shardshuffle)                    # shard 级洗牌
  .then(wds.split_by_node)                   # 按节点分片
  .then(wds.split_by_worker)                 # 按 worker 分片
  .shuffle(shuffle_buffer, initial=seed, seed=seed)  # seed=seed 保证确定性
  .decode()                                  # 默认解码（npy/json/pickle）
  .compose(partial(_normalize_and_filter, ...)) # 规范化 + 过滤
  .map(preprocess_frame)                     # 解码图像，转 tensor
  .shuffle(post_clip_shuffle, initial=seed, seed=seed)  # seed=seed 保证确定性
```

> **种子确定性修复**: `webdataset` 的 `_shuffle` 函数中，`initial` 参数控制 buffer 填充阈值，而非随机种子。
> RNG 由独立的 `seed` 参数控制。若不传 `seed=seed`，RNG 会用 `os.getpid() + time.time()` 初始化，
> 导致相同 seed 的两次运行产出不同样本。`src/data/wds.py` 已修复所有 `.shuffle()` 和 `WebDataset()` 调用点。

### 1.6.5 两级 RandomMix 训练加载器

`src/data/wds.py:445-512` 中的 `get_group_reweight_precut_clip_dataloader` 是生产训练的核心加载器。

**两级混合结构**：

```
Level 1 输出: wds.RandomMix([ego_stream, aux_stream], weights=[0.2, 0.8])
  |
  ├── ego_stream: wds.RandomMix([
  │     HOT3D_stream (来自 _build_precut_clip_webdataset, seed+0),
  │     AssemblyHands_stream (seed+1009)
  │   ], weights=[0.5, 0.5])
  │
  └── aux_stream: wds.RandomMix([
        InterHand2.6M_stream (seed+0),
        FreiHAND_stream (seed+1009),
        MTC_stream (seed+2018),
        ... (每个数据集 seed 偏移 = group_idx * 100003 + dataset_idx * 1009)
      ], weights=[0.25, 0.1875, ...])
```

- `infinite=True`: `resampled=True`, `longest=False`（概率混合，无限循环）
- `infinite=False`: `resampled=False`, `longest=True`（线性混合，一次遍历）

### 1.6.6 分段评估加载器

`src/data/wds.py:515-611`:
- `WDSClipSegmentDataset`: `IterableDataset`，按分配好的 `ClipSegment` 列表顺序读取 tar，只 yield 属于该 rank 的 clip 窗口
- `get_segmented_wds_dataloader`: 包装为 DataLoader，`__len__` 返回精确的 clip 数

### 1.6.7 Collation

`src/data/wds.py:275-291` 中的 `collate_fn`:
- 过滤 None 条目
- `COLLATE_LIST_KEYS = {"imgs"}`: 保持为 list（不 stack——因为图像尺寸可能不同）
- `torch.Tensor`: `torch.stack`
- 其他（字符串、int）: 保持为 list

## 1.7 采样过滤

`src/data/sampler.py:10-56` 中的 `build_clip_sample_filter_fn` 构建训练时的 clip 级质量过滤器：

**参数**：
- `min_valid_joints_2d` (默认 16): 最少有效 2D 关节点数
- `min_hand_bbox_edge_px` (默认 8): 最小 hand bbox 边长
- `frame_policy`:
  - `"all"` (默认): 所有帧必须都通过
  - `"last"`: 只有最后一帧需要通过
  - `"any"`: 至少一帧通过

过滤在 `normalize_decoded_clip_sample` 之后、`preprocess_frame`（图像解码）之前执行，节省计算。

## 1.8 Clip 导出管线

### 1.8.1 导出入口

`script/export_train_clips.py:183` 行的主脚本。

**Stage-aware 参数**：

| Stage | Clip Length | Stride | Output Split |
|-------|------------|--------|-------------|
| stage1 | 1 | 1 | train_stage1 |
| stage2 | 7 | 4 | train_stage2 |

### 1.8.2 核心导出函数

`src/data/export.py:169-243` 中的 `export_sequence_shards_to_clip_shards`：

1. 调用 `iter_sequence_clip_samples` 遍历 sequence shard：
   - 打开 WebDataset（`shardshuffle=false`, 无节点/worker 分割）
   - 规范化 + 固定长度滑动窗口切 clip
2. 对每个 clip 序列化：
   - JSON 字段 → `json.dumps(ensure_ascii=False)`
   - Pickle 字段 → `pickle.dumps(protocol=4)`
   - NPY 字段 → `np.save` 到内存 buffer
3. 按 `max_tar_size_bytes`（默认 1GB）轮转 tar 文件
4. 导出 `_export_stats.json` 统计信息

## 1.9 附录：完整数据流图

```
训练路径:
  build_train_data_plan(config.DATA)
    → collect_supervision_dataset_groups() 验证 ego/aux 成员
    → _resolve_dataset_split_sources() 展开 glob → 文件列表
    → 归一化权重
    → TrainDataPlan

  get_group_reweight_precut_clip_dataloader(plan)
    → 构建两级 RandomMix
    → 每级内部 _build_precut_clip_webdataset()
      → 解码 → normalize_decoded_clip_sample() → sample_filter → preprocess_frame()
    → collate_fn → DataLoader
    → preprocess_batch() 应用增强 → 模型

评估路径:
  estimate_wds_shard_clip_counts() → build_balanced_clip_segments() → equalize_rank_clip_segments()
  → WDSClipSegmentDataset → get_segmented_wds_dataloader()
  → preprocess_batch(无增强) → 模型

导出路径:
  export_sequence_shards_to_clip_shards()
  → iter_sequence_clip_samples() 滑窗
  → serialize_clip_sample_members() → tar 轮转
```
