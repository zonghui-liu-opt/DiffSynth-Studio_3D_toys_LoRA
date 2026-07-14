# DirectDistill/BSA：将已有多 seed teacher latent 安全收敛到 seed=1

本文适用于以下情况：teacher latent 已按 seed `1,2,3,4` 全部提取完成，但后续 DirectDistill/BSA 训练和推理只希望使用 seed `1`。

## 结论

- 不需要重新提取 seed=1 teacher latent。
- 只过滤 metadata 中 `seed=1` 的完整数据行；已有 latent 文件保持不动。
- 绝对不能把 seed 2/3/4 的 `seed` 字段直接改成 `1`。这些 latent 对应不同随机轨迹，改标签会破坏 latent、seed 和 provenance 的一致性。
- 建议创建全新的 seed1 teacher root，通过软链接共享 `teacher_latents/` 和 `first_frames/`。这不是物理备份：两个 root 指向同一批大文件，必须把共享目录当作只读数据使用。
- 当前 BSA 训练入口会在加载 5B 模型前拒绝包含 seed 2/3/4 的 metadata；BSA Test 推理入口固定使用 seed=1。

## 1. 先确认原始 metadata 的 seed 分布

```bash
export ALL_SEED_ROOT=/data/teacher/figurine360-all-seeds

python3 - "$ALL_SEED_ROOT/metadata_direct_distill_train.csv" <<'PY'
import csv
import pathlib
import sys
from collections import Counter

path = pathlib.Path(sys.argv[1])
with path.open("r", encoding="utf-8-sig", newline="") as file:
    rows = list(csv.DictReader(file))

print("rows:", len(rows))
print("seeds:", Counter((row.get("seed") or "").strip() for row in rows))
PY
```

如果每个对象都有四个 seed，过滤后 seed1 行数通常约为原来的四分之一。

## 2. 创建不复制大文件的 seed1 teacher root

下面的脚本不会修改或删除原目录，只会：

1. 创建新的 seed1 root；
2. 软链接已有 `teacher_latents/` 和 `first_frames/`；
3. 原子写入只包含 seed1 行的新 metadata。

```bash
export ALL_SEED_ROOT=/data/teacher/figurine360-all-seeds
export SEED1_ROOT=/data/teacher/figurine360-seed1

python3 - "$ALL_SEED_ROOT" "$SEED1_ROOT" <<'PY'
import csv
import os
import pathlib
import sys

source = pathlib.Path(sys.argv[1]).expanduser().resolve()
target = pathlib.Path(sys.argv[2]).expanduser().resolve()
if target == source:
    raise SystemExit("SEED1_ROOT 不能与 ALL_SEED_ROOT 相同")
if target.exists() and any(target.iterdir()):
    raise SystemExit(f"SEED1_ROOT 必须是全新目录或空目录: {target}")
target.mkdir(parents=True, exist_ok=True)

for directory in ("teacher_latents", "first_frames"):
    source_dir = source / directory
    target_dir = target / directory
    if not source_dir.is_dir():
        raise SystemExit(f"缺少目录: {source_dir}")
    if target_dir.exists() or target_dir.is_symlink():
        if target_dir.resolve() != source_dir.resolve():
            raise SystemExit(f"目标已存在且指向其他位置: {target_dir}")
    else:
        target_dir.symlink_to(source_dir, target_is_directory=True)

for name in (
    "metadata_direct_distill.csv",
    "metadata_direct_distill_train.csv",
    "metadata_direct_distill_validation.csv",
):
    source_csv = source / name
    if not source_csv.is_file():
        print(f"跳过不存在的文件: {source_csv}")
        continue

    with source_csv.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fields = reader.fieldnames or []
        if "seed" not in fields:
            raise SystemExit(f"{source_csv} 缺少 seed 列")
        rows = list(reader)

    seed1_rows = [
        row for row in rows
        if (row.get("seed") or "").strip() == "1"
    ]
    if name == "metadata_direct_distill_train.csv" and not seed1_rows:
        raise SystemExit(f"{source_csv} 没有 seed=1 训练样本")
    if name == "metadata_direct_distill_validation.csv" and not seed1_rows:
        print("警告: validation metadata 没有 seed1 行；请改用 BSA Test 测试集推理")

    output_csv = target / name
    temporary = output_csv.with_suffix(output_csv.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(seed1_rows)
    os.replace(temporary, output_csv)
    print(f"{name}: {len(rows)} -> {len(seed1_rows)} rows")
PY
```

不要在确认新训练链路前删除原多 seed root。软链接共享的不是副本：不要通过 `SEED1_ROOT` 再运行 teacher prepare、清理或删除 `teacher_latents/`、`first_frames/` 中的文件，否则原 root 中的同一文件也会受到影响。

## 3. 验证 seed、正式几何和引用文件

正式 BSA 配置要求 `height=480,width=832,num_frames=81`。运行：

```bash
python3 - "$SEED1_ROOT" <<'PY'
import csv
import pathlib
import sys
from collections import Counter

root = pathlib.Path(sys.argv[1]).expanduser().resolve()
metadata = root / "metadata_direct_distill_train.csv"
with metadata.open("r", encoding="utf-8-sig", newline="") as file:
    rows = list(csv.DictReader(file))

if not rows:
    raise SystemExit("seed1 训练 metadata 为空")

seeds = Counter((row.get("seed") or "").strip() for row in rows)
if set(seeds) != {"1"}:
    raise SystemExit(f"发现非 seed1 数据: {seeds}")

expected_shape = (480, 832, 81)
missing = []
bad_shape = []
for row_number, row in enumerate(rows, start=2):
    actual_shape = tuple(int(row[key]) for key in ("height", "width", "num_frames"))
    if actual_shape != expected_shape:
        bad_shape.append((row_number, actual_shape))
    for key in ("input_image", "teacher_latent"):
        path = pathlib.Path(row[key])
        path = path if path.is_absolute() else root / path
        if not path.is_file():
            missing.append((row_number, key, str(path)))

if bad_shape:
    raise SystemExit(f"正式几何不匹配: {bad_shape[:8]}")
if missing:
    raise SystemExit(f"metadata 引用文件不存在: {missing[:8]}")

print(f"检查通过: rows={len(rows)}, seeds={dict(seeds)}, shape={expected_shape}")
PY
```

## 4. 使用 seed1 metadata 启动新的 BSA 实验

不要从使用多 seed 数据训练过的 BSA checkpoint 继续训练。若目标是严格的 seed1 实验，应从 dense DirectDistill warm-start 开启新 BSA 输出目录：

```bash
cd /data/code/DiffSynth-Studio_3D_toys

export TEACHER_ROOT="$SEED1_ROOT"
export DENSE_WARMSTART_LORA=/data/weights/direct-distill-step-19600.safetensors
export BSA_TRAIN_OUTPUT=/data/outputs/bsa-directdistill-seed1-conservative-v1
export BSA_SPARSITY_SCHEDULE=conservative_epoch_v1
export NUM_EPOCHS=60
export GRADIENT_ACCUMULATION_STEPS=1
unset BSA_CHECKPOINT

SCRIPT=examples/wanvideo/model_training/special/direct_distill/Wan2.2-TI2V-5B-Figurine360-BSA.sh
bash "$SCRIPT" doctor
test ! -e "$BSA_TRAIN_OUTPUT"

nohup bash "$SCRIPT" train \
  >"${BSA_TRAIN_OUTPUT}.launch.log" 2>&1 </dev/null &
tail -F "${BSA_TRAIN_OUTPUT}.launch.log"
```

Warmup 会根据过滤后的 metadata 行数、卡数和 gradient accumulation 自动重算。若过滤后为 721 行，6 卡、batch=1/rank、GA=1 时仍为 121 steps/epoch、60 epochs 共 7260 optimizer steps。

## 5. seed1 推理

BSA Test 不读取 teacher latent，只需要 `input_image,prompt` 测试集。`input_image` 必须是存在的绝对路径；相对路径会被 `doctor` 拒绝。metadata 可以没有 `seed` 列；如果包含，则所有行必须为 `1`。

```bash
export TEST_METADATA=/data/test/metadata-seed1.csv
export TEST_OUTPUT=/data/outputs/bsa-directdistill-seed1-test
export BSA_CHECKPOINT=/data/outputs/bsa-directdistill-seed1-conservative-v1/checkpoint-step-...

TEST_SCRIPT=examples/wanvideo/model_training/special/direct_distill/Wan2.2-TI2V-5B-Figurine360-BSA-Test.sh
bash "$TEST_SCRIPT" doctor
SAMPLE_INDEX=0 bash "$TEST_SCRIPT" one
START_INDEX=0 END_INDEX=12 bash "$TEST_SCRIPT" all
```

目标三联视频为：

```text
$TEST_OUTPUT/sample-0/teacher_vs_student_vs_student_bsa.mp4
```

## 6. 关于已有 19600-step DirectDistill 权重

- 如果该权重训练时使用的 metadata 本来就只有 seed1，则可以直接作为 BSA warm-start。
- 如果该权重实际使用了 seed 1/2/3/4 混合数据，那么只过滤 BSA metadata 只能保证后续 BSA 阶段是 seed1，不能把已有 DirectDistill 阶段追溯性地称为 seed1-only。
- 若实验结论要求整个 DirectDistill → BSA 链路都严格 seed1，应使用本文生成的 seed1 metadata 重新训练 DirectDistill，再启动新的 BSA 实验。
