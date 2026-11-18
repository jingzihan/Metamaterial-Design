import sys
import os
import json
import csv
import time
import shutil
from collections import defaultdict

import numpy as np
from scipy.stats import qmc


# 0. CST path and data path
cst_lib_path = r"D:\Program Files (x86)\CST Studio Suite 2024\AMD64\python_cst_libraries"
if cst_lib_path not in sys.path:
    sys.path.append(cst_lib_path)

import cst
import cst.interface


path = os.getcwd()
data_save_path = r"D:\python study\shoujishuju\data_LHS_H_discrete_10000"
os.makedirs(data_save_path, exist_ok=True)


# 1. Sampling parameters (manuscript-aligned)
TOTAL_SAMPLES = 10000
BATCH_SIZE    = 20

H_LEVELS = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
N_H      = len(H_LEVELS)
R_FIXED  = 20.0
LHS_SEED          = 42
MATRIX_THRESHOLD  = 0.5
TOPOLOGY_DIM      = 25
CU_THICKNESS_MM   = 0


# 2. Discrete Latin hypercube for substrate thickness h
counts = np.full(N_H, TOTAL_SAMPLES // N_H, dtype=int)
counts[-1] += TOTAL_SAMPLES - counts.sum()

h_layer_idx = np.repeat(np.arange(N_H), counts)
rng_h = np.random.default_rng(LHS_SEED + 1)
rng_h.shuffle(h_layer_idx)
h_samples = H_LEVELS[h_layer_idx]


# 3. 25-D Latin hypercube for the 5x5 topology pattern
print(">>> Generating 25-D LHS for 5x5 topology ...")
sampler_topo = qmc.LatinHypercube(d=TOPOLOGY_DIM, seed=LHS_SEED)
lhs_samples = sampler_topo.random(n=TOTAL_SAMPLES)

print(f">>> Total samples: {TOTAL_SAMPLES}")
print(f">>> h levels and counts (discrete LHS):")
for h_val, cnt in zip(H_LEVELS, counts):
    print(f"    H={h_val:.1f} mm : {cnt} samples")


# 4. Resume logic and sampling metadata
existing_folders = [
    int(f) for f in os.listdir(data_save_path)
    if f.isdigit() and os.path.isdir(os.path.join(data_save_path, f))
]
start_idx = max(existing_folders) if existing_folders else 0
print(f">>> Found max historical sample ID: {start_idx}")

config_path = os.path.join(data_save_path, "sampling_config.json")
sampling_config = {
    "total_samples": int(TOTAL_SAMPLES),
    "h_levels_mm": H_LEVELS.tolist(),
    "counts_per_h": counts.tolist(),
    "h_assignment": "discrete Latin hypercube, layer counts 1111/1112, shuffled by seed",
    "R_fixed_ohm": float(R_FIXED),
    "lhs_seed_topology": int(LHS_SEED),
    "lhs_seed_h": int(LHS_SEED + 1),
    "topology_dim": int(TOPOLOGY_DIM),
    "matrix_threshold": float(MATRIX_THRESHOLD),
    "matrix_layout": "5x5 binary pattern rotated into 10x10 quadrants",
    "cu_thickness_mm": float(CU_THICKNESS_MM),
    "adaptive_mesh_convergence": 0.02,
}

if os.path.exists(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        saved = json.load(f)
    if saved != sampling_config:
        raise RuntimeError(
            "sampling_config.json mismatch. Use a fresh data_save_path "
            "or restore the original parameters."
        )
elif start_idx > 0:
    raise RuntimeError(
        "Historical data exist but sampling_config.json is missing. "
        "Use a fresh data_save_path."
    )
else:
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(sampling_config, f, ensure_ascii=False, indent=2)


h_schedule_path = os.path.join(data_save_path, "H_sampling.csv")
with open(h_schedule_path, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(["sample_id", "H_mm"])
    for sid, h_val in enumerate(h_samples, start=1):
        writer.writerow([sid, f"{h_val:.1f}"])


for sid in existing_folders:
    if sid < 1 or sid > TOTAL_SAMPLES:
        continue
    sample_dir = os.path.join(data_save_path, str(sid))
    h_file = os.path.join(sample_dir, "H.txt")
    s_file = os.path.join(sample_dir, "s.txt")
    if not os.path.exists(h_file):
        if os.path.exists(s_file) and os.path.getsize(s_file) > 100:
            raise RuntimeError(
                f"Sample {sid} has S-params but no H.txt. "
                "Use a fresh data_save_path or restore H.txt."
            )
        continue
    with open(h_file, "r", encoding="utf-8") as f:
        recorded_h = float(f.read().strip())
    expected_h = float(h_samples[sid - 1])
    if not np.isclose(recorded_h, expected_h, atol=1e-9):
        raise RuntimeError(
            f"Sample {sid}: recorded H={recorded_h}, expected H={expected_h}. "
            "Restore original settings or use a fresh data_save_path."
        )

completed_ids = set()
for sid in existing_folders:
    if sid < 1 or sid > TOTAL_SAMPLES:
        continue
    s_file = os.path.join(data_save_path, str(sid), "s.txt")
    if os.path.exists(s_file) and os.path.getsize(s_file) > 100:
        completed_ids.add(sid)


pending_by_h = defaultdict(list)
for sid in range(1, TOTAL_SAMPLES + 1):
    if sid in completed_ids:
        continue
    pending_by_h[float(h_samples[sid - 1])].append(sid)

if not any(pending_by_h.values()):
    print(">>> All samples completed.")
    sys.exit()

# 5. CST environment
cst_app = cst.interface.DesignEnvironment()


def setup_static_environment(modeler, h_value):
    p = 30
    frq = [2, 18]
    lb = "\n"

    def add_vba(name, command):
        modeler.add_to_history(name, command)


    add_vba(
        "init_params",
        f'StoreParameter "sub_h", "{h_value:.1f}"\n'
        f'StoreParameter "sheet_res", "{R_FIXED}"\n'
        f'StoreParameter "cu_t", "{CU_THICKNESS_MM}"',
    )
    add_vba(
        "define_frequency",
        f'Solver.FrequencyRange "{frq[0]}","{frq[1]}"',
    )


    cmd_mat_pet = [
        "With Material",
        ".Reset",
        '.Name "material1"',
        '.Epsilon "2.8"',
        '.Mue "1.0"',
        '.Sigma "0.0"',
        '.TanD "0.015"',
        '.Type "Normal"',
        ".Create",
        "End With",
    ]
    add_vba("define_material_pet", lb.join(cmd_mat_pet))


    cmd_mat_res = [
        "With Material",
        ".Reset",
        '.Name "ResistorMaterial"',
        '.Type "Lossy metal"',
        '.OhmicSheetImpedance "sheet_res", "0"',
        '.Colour "1", "0", "0"',
        ".Create",
        "End With",
    ]
    add_vba("define_material_res", lb.join(cmd_mat_res))


    add_vba("new_component", 'Component.New "static_comp"')


    cmd_sub = [
        "With Brick",
        ".Reset",
        '.Name "substrate"',
        '.Component "static_comp"',
        '.Material "material1"',
        f'.Xrange "{-p/2}", "{p/2}"',
        f'.Yrange "{-p/2}", "{p/2}"',
        '.Zrange "0", "sub_h"',
        ".Create",
        "End With",
    ]
    add_vba("define_substrate", lb.join(cmd_sub))


    cmd_gnd = [
        "With Brick",
        ".Reset",
        '.Name "copper_ground"',
        '.Component "static_comp"',
        '.Material "Copper (annealed)"',
        f'.Xrange "{-p/2}", "{p/2}"',
        f'.Yrange "{-p/2}", "{p/2}"',
        '.Zrange "-cu_t", "0"',
        ".Create",
        "End With",
    ]
    add_vba("define_copper_ground", lb.join(cmd_gnd))


    cmd_bnd = [
        "With Boundary",
        '.Xmin "unit cell"',
        '.Xmax "unit cell"',
        '.Ymin "unit cell"',
        '.Ymax "unit cell"',
        '.Zmin "expanded open"',
        '.Zmax "expanded open"',
        '.OpenAddSpaceFactor "0.5"',
        '.SetPeriodicBoundaryAngles "0.0", "0.0"',
        '.SetPeriodicBoundaryAnglesDirection "inward"',
        '.UnitCellFitToBoundingBox "True"',
        "End With",
    ]
    add_vba("define_boundary", lb.join(cmd_bnd))


    cmd_port = [
        "With FloquetPort",
        ".Reset",
        '.Port "Zmax"',
        '.SetNumberOfModesConsidered "2"',
        '.SetDistanceToReferencePlane "0.0"',
        "End With",
    ]
    add_vba("define_floquet_port", lb.join(cmd_port))

    # --- solver type ---
    add_vba("solver_type", 'ChangeSolverType "HF Frequency Domain"')


    sCommand = ['With MeshAdaption3D',
                '.SetType "HighFrequencyTet"',
                '.SetAdaptionStrategy "ExpertSystem"',
                '.MinPasses "3"',
                '.MaxPasses "6"',
                '.ClearStopCriteria',
                '.MaxDeltaS "0.02"',
                '.NumberOfDeltaSChecks "3"',
                '.EnableInnerSParameterAdaptation "True"',
                '.PropagationConstantAccuracy "0.005"',
                '.NumberOfPropConstChecks "2"',
                '.EnablePortPropagationConstantAdaptation "True"',
                '.RemoveAllUserDefinedStopCriteria',
                '.AddStopCriterion "All S-Parameters", "0.02", "3", "True"',
                '.AddStopCriterion "Reflection S-Parameters", "0.02", "1", "False"',
                '.AddStopCriterion "Transmission S-Parameters", "0.02", "1", "False"',
                '.AddStopCriterion "Portmode kz/k0", "0.005", "2", "True"',
                '.AddStopCriterion "All Probes", "0.05", "2", "False"',
                '.AddSParameterStopCriterion "True", "", "", "0.02", "1", "False"',
                '.MinimumAcceptedCellGrowth "0.5"',
                '.RefThetaFactor ""',
                '.SetMinimumMeshCellGrowth "5"',
                '.ErrorEstimatorType "Automatic"',
                '.RefinementType "Automatic"',
                '.SnapToGeometry "True"',
                '.SubsequentChecksOnlyOnce "False"',
                '.WavelengthBasedRefinement "True"',
                '.EnableLinearGrowthLimitation "True"',
                '.SetLinearGrowthLimitation ""',
                '.SingularEdgeRefinement "2"',
                '.DDMRefinementType "Automatic"',
                'End With']
    modeler.add_to_history("set_3d_mesh_adaptation", lb.join(sCommand))



def restart_and_clean_cst(current_modeler, batch_idx, h_value, last_cst_path=None):
    if current_modeler is not None:
        print("\n  [Cleanup] Closing current CST ...")
        try:
            current_modeler.add_to_history("Quit App", "Quit")
        except Exception:
            pass
        time.sleep(3)

    if last_cst_path and os.path.exists(last_cst_path):
        print("  [Cleanup] Removing temporary CST project ...")
        cst_folder = os.path.splitext(last_cst_path)[0]
        try:
            if os.path.exists(cst_folder):
                shutil.rmtree(cst_folder, ignore_errors=True)
            if os.path.exists(last_cst_path):
                os.remove(last_cst_path)
        except Exception as e:
            print(f"  [Warning] cleanup failed: {e}")

    tag = time.strftime("%Y%m%d_%H%M%S")
    new_name = (
        f"Temp_LHS_{tag}_{os.getpid()}_Batch_{batch_idx:04d}_H{h_value:.1f}.cst"
    )
    new_path = os.path.join(path, new_name)

    print(f"  [Init] New CST project (batch {batch_idx}, H={h_value:.1f} mm) ...")
    new_mws = cst_app.new_mws()
    new_mws.save(new_path)
    new_modeler = new_mws.model3d
    setup_static_environment(new_modeler, h_value)
    return new_mws, new_modeler, new_path



# 7. Main loop: iterate by H group, then by sample ID
print(
    f">>> Start: {sum(len(v) for v in pending_by_h.values())} pending samples, "
    f"grouped by H across {N_H} levels."
)

batch_number = 0
mws = None
modeler = None
current_cst_path = None

for h_val in sorted(pending_by_h.keys()):
    pending_ids = sorted(pending_by_h[h_val])
    if not pending_ids:
        continue


    batch_number += 1
    mws, modeler, current_cst_path = restart_and_clean_cst(
        modeler, batch_number, h_val, current_cst_path
    )
    samples_in_batch = 0

    for global_id in pending_ids:
        if samples_in_batch >= BATCH_SIZE:
            batch_number += 1
            mws, modeler, current_cst_path = restart_and_clean_cst(
                modeler, batch_number, h_val, current_cst_path
            )
            samples_in_batch = 0

        i = global_id - 1
        matrix_5x5 = (lhs_samples[i] > MATRIX_THRESHOLD).astype(int).reshape((5, 5))
        A_cw90  = np.rot90(matrix_5x5, k=-1)
        A_cw180 = np.rot90(matrix_5x5, k=2)
        A_cw270 = np.rot90(matrix_5x5, k=1)
        top_row    = np.hstack((matrix_5x5, A_cw90))
        bottom_row = np.hstack((A_cw270, A_cw180))
        matrix_10x10 = np.vstack((top_row, bottom_row))

        sample_dir = os.path.join(data_save_path, str(global_id))
        os.makedirs(sample_dir, exist_ok=True)


        with open(os.path.join(sample_dir, "H.txt"), "w", encoding="utf-8") as f:
            f.write(f"{h_val:.1f}")
        with open(os.path.join(sample_dir, "R.txt"), "w", encoding="utf-8") as f:
            f.write(str(R_FIXED))
        np.savetxt(
            os.path.join(sample_dir, "matrix_5x5.txt"),
            matrix_5x5, fmt="%d", delimiter="\t",
        )
        np.savetxt(
            os.path.join(sample_dir, "matrix_10x10.txt"),
            matrix_10x10, fmt="%d", delimiter="\t",
        )

        print(
            f"  ID {global_id}/{TOTAL_SAMPLES} | H={h_val:.1f} mm, R={R_FIXED} ... ",
            end="", flush=True,
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

            p = 30.0
            pixel_size = p / 10.0
            start_x = -p / 2
            start_y = p / 2
            pixel_cmds = []
            for row in range(10):
                for col in range(10):
                    if matrix_10x10[row, col] == 1:
                        x_min = start_x + col * pixel_size
                        x_max = start_x + (col + 1) * pixel_size
                        y_max = start_y - row * pixel_size
                        y_min = start_y - (row + 1) * pixel_size
                        pixel_cmds.append(
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
            if pixel_cmds:
                modeler.add_to_history(
                    f"draw_pixels_{uid}", "\n".join(pixel_cmds)
                )

            modeler.add_to_history(f"clear_results_{uid}", "DeleteResults")
            modeler.run_solver()

            s_file = os.path.join(sample_dir, "s.txt")
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

            time.sleep(1.0)

            if os.path.exists(s_file) and os.path.getsize(s_file) > 100:
                print("OK.")
            else:
                print("Failed! S-parameter file missing or empty.")

        except Exception as e:
            print(f"Error at ID {global_id}: {e}")
            time.sleep(2)
            continue
        finally:
            samples_in_batch += 1

if modeler is not None:
    print("\n  [Cleanup] Closing final CST project ...")
    try:
        modeler.add_to_history("Quit App", "Quit")
    except Exception:
        pass
    time.sleep(3)

if current_cst_path and os.path.exists(current_cst_path):
    last_folder = os.path.splitext(current_cst_path)[0]
    try:
        if os.path.exists(last_folder):
            shutil.rmtree(last_folder, ignore_errors=True)
        os.remove(current_cst_path)
    except Exception as e:
        print(f"  [Warning] final cleanup failed: {e}")