import csv
import json
import os
import shutil
import sys
import time

import numpy as np
from scipy.stats import qmc


# ==========================================
# 0. CST 路径与数据保存路径
# ==========================================
cst_lib_path = r"D:\Program Files (x86)\CST Studio Suite 2024\AMD64\python_cst_libraries"
if cst_lib_path not in sys.path:
    sys.path.append(cst_lib_path)

import cst.interface


path = os.getcwd()

# 使用新的目录，避免把之前逐样本改变 H 的数据与本次分段固定 H 的数据混在一起。
# 如果本次任务中断后续跑，请保持此路径不变。
data_save_path = r"D:\python study\shoujishuju\data_LHS_H_blocks_1to5"
os.makedirs(data_save_path, exist_ok=True)


# ==========================================
# 1. LHS 抽样参数
# ==========================================
SAMPLES_PER_H = 1111
BATCH_SIZE = 20

# H 的 9 个离散取值：1.0、1.5、...、5.0 mm。
# 每个 H 水平生成 1000 个样本，因此本方案总样本数为 10000。
H_LEVELS = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
TOTAL_SAMPLES = SAMPLES_PER_H * len(H_LEVELS)
R_FIXED = 20.0

LHS_SEED = 42
MATRIX_THRESHOLD = 0.5

# 续跑时使用相同 seed 和相同参数，可重新生成完全一致的 25 维矩阵 LHS 序列。
print(">>> 正在生成 25 维 LHS 抽样空间（用于 5x5 表面图案）...")
sampler = qmc.LatinHypercube(d=25, seed=LHS_SEED)
lhs_samples = sampler.random(n=TOTAL_SAMPLES)

# 按样本编号分段固定 H：每个 H 水平连续对应 1000 个样本。
h_samples = np.repeat(H_LEVELS, SAMPLES_PER_H)

print(f">>> 共 {TOTAL_SAMPLES} 个样本，每个 H 水平 {SAMPLES_PER_H} 个：")
for h_value in H_LEVELS:
    print(f"    H={h_value:.1f} mm: {SAMPLES_PER_H} 个")


# ==========================================
# 2. 断点续传与抽样信息导出
# ==========================================
existing_folders = [
    int(folder_name)
    for folder_name in os.listdir(data_save_path)
    if folder_name.isdigit()
    and os.path.isdir(os.path.join(data_save_path, folder_name))
]
start_idx = max(existing_folders) if existing_folders else 0
print(f">>> 已发现最大历史样本编号：{start_idx}。")

if start_idx > TOTAL_SAMPLES:
    raise RuntimeError(
        f"历史数据编号已到 {start_idx}，超过 TOTAL_SAMPLES={TOTAL_SAMPLES}。"
        "请检查数据目录或调整 TOTAL_SAMPLES。"
    )

config_path = os.path.join(data_save_path, "sampling_config.json")
sampling_config = {
    "total_samples": TOTAL_SAMPLES,
    "samples_per_h": SAMPLES_PER_H,
    "lhs_seed": LHS_SEED,
    "lhs_dimension": 25,
    "h_levels_mm": H_LEVELS.tolist(),
    "R_fixed_ohm": R_FIXED,
    "matrix_threshold": MATRIX_THRESHOLD,
    "h_assignment": (
        f"{SAMPLES_PER_H} consecutive sample IDs per H level, in ascending H order"
    ),
    "matrix_layout": "5x5 binary pattern rotated into 10x10 quadrants",
}

if os.path.exists(config_path):
    with open(config_path, "r", encoding="utf-8") as config_file:
        saved_config = json.load(config_file)
    if saved_config != sampling_config:
        raise RuntimeError(
            "当前抽样参数与数据目录中的 sampling_config.json 不一致。"
            "为避免续跑时样本编号与 H 值错位，请恢复原参数或使用新的 data_save_path。"
        )
elif start_idx > 0:
    raise RuntimeError(
        "数据目录中已有历史样本，但没有 sampling_config.json，无法确认历史 H 与新抽样序列一致。"
        "请将 data_save_path 指向一个新目录，避免把旧版固定 H 数据混入。"
    )
else:
    with open(config_path, "w", encoding="utf-8") as config_file:
        json.dump(sampling_config, config_file, ensure_ascii=False, indent=2)

# 保存完整抽样计划，Excel 可直接打开；每个样本文件夹内也会单独保存 H.txt。
h_schedule_path = os.path.join(data_save_path, "H_sampling.csv")
with open(h_schedule_path, "w", newline="", encoding="utf-8-sig") as schedule_file:
    writer = csv.writer(schedule_file)
    writer.writerow(["sample_id", "H_mm"])
    writer.writerows(
        (sample_id, f"{h_value:.1f}")
        for sample_id, h_value in enumerate(h_samples, start=1)
    )

# 续跑前核对已完成样本的 H 值，防止修改 seed 或离散水平后误接旧数据。
for sample_id in existing_folders:
    if sample_id < 1 or sample_id > TOTAL_SAMPLES:
        continue

    sample_folder = os.path.join(data_save_path, str(sample_id))
    h_file = os.path.join(sample_folder, "H.txt")
    s_file = os.path.join(sample_folder, "s.txt")

    if not os.path.exists(h_file):
        if os.path.exists(s_file) and os.path.getsize(s_file) > 100:
            raise RuntimeError(
                f"历史样本 {sample_id} 有 S 参数结果但缺少 H.txt，无法确认其基板厚度。"
                "请使用新的 data_save_path，或补齐该样本的 H.txt。"
            )
        continue

    with open(h_file, "r", encoding="utf-8") as h_value_file:
        recorded_h = float(h_value_file.read().strip())
    expected_h = float(h_samples[sample_id - 1])
    if not np.isclose(recorded_h, expected_h, atol=1e-9):
        raise RuntimeError(
            f"历史样本 {sample_id} 记录的 H={recorded_h} mm，"
            f"但当前 LHS 序列要求 H={expected_h:.1f} mm。"
            "请恢复原抽样设置或使用新的 data_save_path。"
        )

# 只有存在有效 S 参数文件的样本才算完成。失败的文件夹（例如本次 ID 1–4）会自动重试。
completed_ids = set()
for sample_id in existing_folders:
    if sample_id < 1 or sample_id > TOTAL_SAMPLES:
        continue
    s_file = os.path.join(data_save_path, str(sample_id), "s.txt")
    if os.path.exists(s_file) and os.path.getsize(s_file) > 100:
        completed_ids.add(sample_id)

pending_ids = [sample_id for sample_id in range(1, TOTAL_SAMPLES + 1)
               if sample_id not in completed_ids]
if not pending_ids:
    print(">>> 数据已全部生成完毕！")
    sys.exit()
first_pending_id = pending_ids[0]


# ==========================================
# 3. CST 环境与建模基础功能
# ==========================================
cst_app = cst.interface.DesignEnvironment()


def setup_static_environment(modeler, h_value):
    # H 只在一个全新的 CST 项目中初始化；同一项目里的样本不再改 sub_h。
    p = 30
    frq = [2, 18]
    line_break = "\n"

    def add_vba(name, command):
        modeler.add_to_history(name, command)

    add_vba(
        "init_params",
        f'StoreParameter "sub_h", "{h_value:.1f}"\n'
        f'StoreParameter "sheet_res", "{R_FIXED}"',
    )
    add_vba(
        "define frequency",
        f'Solver.FrequencyRange "{frq[0]}","{frq[1]}"',
    )

    cmd_mat1 = [
        "With Material",
        ".Reset",
        '.Name "material1"',
        '.Epsilon "2.8"',
        '.Type "Normal"',
        ".Create",
        "End With",
    ]
    add_vba("define material1", line_break.join(cmd_mat1))

    cmd_res = [
        "With Material",
        ".Reset",
        '.Name "ResistorMaterial"',
        '.Type "Lossy metal"',
        '.OhmicSheetImpedance "sheet_res", "0"',
        '.Colour "1", "0", "0"',
        ".Create",
        "End With",
    ]
    add_vba("define material_resistor", line_break.join(cmd_res))

    add_vba("new component", 'Component.New "static_comp"')
    cmd_sub = [
        "With Brick",
        ".Reset",
        '.Name "substrate"',
        '.Component "static_comp"',
        '.Material "material1"',
        f'.Xrange "{-p / 2}", "{p / 2}"',
        f'.Yrange "{-p / 2}", "{p / 2}"',
        '.Zrange "0", "sub_h"',
        ".Create",
        "End With",
    ]
    add_vba("define substrate", line_break.join(cmd_sub))

    cmd_bound = [
        "With Boundary",
        '.Xmin "unit cell"',
        '.Xmax "unit cell"',
        '.Ymin "unit cell"',
        '.Ymax "unit cell"',
        '.Zmin "electric"',
        '.Zmax "expanded open"',
        '.OpenAddSpaceFactor "0.5"',
        '.SetPeriodicBoundaryAngles "0.0", "0.0"',
        '.SetPeriodicBoundaryAnglesDirection "inward"',
        '.UnitCellFitToBoundingBox "True"',
        "End With",
    ]
    add_vba("define boundary", line_break.join(cmd_bound))

    cmd_port = [
        "With FloquetPort",
        ".Reset",
        '.Port "Zmax"',
        '.SetNumberOfModesConsidered "2"',
        '.SetDistanceToReferencePlane "0.0"',
        "End With",
    ]
    add_vba("define port", line_break.join(cmd_port))
    add_vba("solver type", 'ChangeSolverType "HF Frequency Domain"')


# ==========================================
# 4. CST 环境重启与临时文件清理
# ==========================================
def restart_and_clean_cst(current_modeler, batch_idx, h_value, last_cst_path=None):
    if current_modeler is not None:
        print("\n  [Cleanup] 关闭当前 CST 进程，等待释放文件锁...")
        try:
            current_modeler.add_to_history("Quit App", "Quit")
        except Exception:
            pass
        time.sleep(3)

    if last_cst_path and os.path.exists(last_cst_path):
        print("  [Cleanup] 清理临时项目文件...")
        cst_folder = os.path.splitext(last_cst_path)[0]
        try:
            if os.path.exists(cst_folder):
                shutil.rmtree(cst_folder, ignore_errors=True)
            if os.path.exists(last_cst_path):
                os.remove(last_cst_path)
        except Exception as error:
            print(f"  [Warning] 文件清理部分失败: {error}")

    run_tag = time.strftime("%Y%m%d_%H%M%S")
    new_cst_name = (
        f"Temp_LHS_{run_tag}_{os.getpid()}_Batch_{batch_idx:04d}_H{h_value:.1f}.cst"
    )
    new_cst_path = os.path.join(path, new_cst_name)

    print(
        f"  [Init] 初始化新 Batch ({batch_idx}) 的 CST 环境，"
        f"固定 H={h_value:.1f} mm..."
    )
    new_mws = cst_app.new_mws()
    new_mws.save(new_cst_path)
    new_modeler = new_mws.model3d
    setup_static_environment(new_modeler, h_value)

    return new_mws, new_modeler, new_cst_path


# ==========================================
# 5. 主循环：每个 H 区段固定厚度，5x5 LHS 图案旋转拼接为 10x10
# ==========================================
print(
    f">>> 开始任务：每 {SAMPLES_PER_H} 个样本切换一次 H "
    f"({H_LEVELS[0]:.1f}–{H_LEVELS[-1]:.1f} mm)，R={R_FIXED} Ω；"
    f"总计 {TOTAL_SAMPLES} 个样本，从待处理 ID {first_pending_id} 开始..."
)

mws = None
modeler = None
current_cst_path = None
current_h = None
samples_in_batch = 0
batch_number = 0

for global_id in range(1, TOTAL_SAMPLES + 1):
    if global_id in completed_ids:
        continue

    i = global_id - 1
    h_value = float(h_samples[i])

    # 取 25 维 LHS 样本并二值化为 5x5 矩阵。
    matrix_5x5 = (
        lhs_samples[i] > MATRIX_THRESHOLD
    ).astype(int).reshape((5, 5))

    # 顺时针旋转并拼接为 10x10 矩阵。
    matrix_cw90 = np.rot90(matrix_5x5, k=-1)
    matrix_cw180 = np.rot90(matrix_5x5, k=2)
    matrix_cw270 = np.rot90(matrix_5x5, k=1)

    top_row = np.hstack((matrix_5x5, matrix_cw90))
    bottom_row = np.hstack((matrix_cw270, matrix_cw180))
    matrix_10x10 = np.vstack((top_row, bottom_row))

    # 每 50 个待处理样本重启 CST；H 切换时也必须先关闭旧项目再新建项目。
    # 每个新项目在建模前直接初始化对应 sub_h，不在项目运行中改 H 或调用 F7。
    h_changed = current_h is None or not np.isclose(h_value, current_h)
    if mws is None or h_changed or samples_in_batch >= BATCH_SIZE:
        batch_number += 1
        mws, modeler, current_cst_path = restart_and_clean_cst(
            modeler, batch_number, h_value, current_cst_path
        )
        current_h = h_value
        samples_in_batch = 0

    current_save_folder = os.path.join(data_save_path, str(global_id))
    os.makedirs(current_save_folder, exist_ok=True)

    # 先记录样本参数和矩阵，即使 CST 求解失败也能追溯样本。
    with open(
        os.path.join(current_save_folder, "H.txt"), "w", encoding="utf-8"
    ) as h_file:
        h_file.write(f"{h_value:.1f}")

    with open(
        os.path.join(current_save_folder, "R.txt"), "w", encoding="utf-8"
    ) as r_file:
        r_file.write(str(R_FIXED))

    np.savetxt(
        os.path.join(current_save_folder, "matrix_5x5.txt"),
        matrix_5x5,
        fmt="%d",
        delimiter="\t",
    )
    np.savetxt(
        os.path.join(current_save_folder, "matrix_10x10.txt"),
        matrix_10x10,
        fmt="%d",
        delimiter="\t",
    )

    print(
        f"  ID {global_id}/{TOTAL_SAMPLES} | H={h_value:.1f} mm, "
        f"R={R_FIXED} Ω ... ",
        end="",
        flush=True,
    )

    try:
        uid = f"ID{global_id}"

        modeler.add_to_history(
            f"clean_pixels_{uid}",
            'On Error Resume Next\n'
            'Component.Delete "pixel_comp"\n'
            "On Error GoTo 0",
        )
        modeler.add_to_history(
            f"new_comp_{uid}", 'Component.New "pixel_comp"'
        )

        pixel_cmds = []
        p = 30.0
        pixel_size = p / 10.0
        start_x = -p / 2
        start_y = p / 2

        for row in range(10):
            for col in range(10):
                if matrix_10x10[row, col] == 1:
                    x_min = start_x + col * pixel_size
                    x_max = start_x + (col + 1) * pixel_size
                    y_max = start_y - row * pixel_size
                    y_min = start_y - (row + 1) * pixel_size

                    command = (
                        "With Brick\n"
                        ".Reset\n"
                        f'.Name "p_{row}_{col}"\n'
                        '.Component "pixel_comp"\n'
                        '.Material "ResistorMaterial"\n'
                        f'.Xrange "{x_min}", "{x_max}"\n'
                        f'.Yrange "{y_min}", "{y_max}"\n'
                        '.Zrange "sub_h", "sub_h"\n'
                        ".Create\n"
                        "End With"
                    )
                    pixel_cmds.append(command)

        if pixel_cmds:
            modeler.add_to_history(
                f"draw_pixels_{uid}", "\n".join(pixel_cmds)
            )

        modeler.add_to_history(f"clear_results_{uid}", "DeleteResults")
        modeler.run_solver()

        s_file = os.path.join(current_save_folder, "s.txt")
        export_cmd = [
            r'SelectTreeItem("1D Results\S-Parameters\SZmax(1),Zmax(1)")',
            "With ASCIIExport",
            ".Reset",
            f'.FileName "{s_file}"',
            ".Execute",
            "End With",
        ]
        modeler.add_to_history(
            f"export_{uid}", "\n".join(export_cmd)
        )

        # 等待导出文件落盘。
        time.sleep(1.0)

        if os.path.exists(s_file) and os.path.getsize(s_file) > 100:
            print("OK.")
        else:
            print("Failed! S-parameter file missing or empty.")

    except Exception as error:
        print(f"Error at ID {global_id}: {error}")
        time.sleep(2)
        continue
    finally:
        # 失败的 ID 下次运行时会重试；本次仍计入 CST 批次长度。
        samples_in_batch += 1

# 任务结束时也关闭并清理最后一个临时 CST 项目。
if modeler is not None:
    print("\n  [Cleanup] 任务结束，关闭最后一个 CST 项目...")
    try:
        modeler.add_to_history("Quit App", "Quit")
    except Exception:
        pass
    time.sleep(3)

if current_cst_path and os.path.exists(current_cst_path):
    last_cst_folder = os.path.splitext(current_cst_path)[0]
    try:
        if os.path.exists(last_cst_folder):
            shutil.rmtree(last_cst_folder, ignore_errors=True)
        os.remove(current_cst_path)
    except Exception as error:
        print(f"  [Warning] 最后一个临时项目清理失败: {error}")

