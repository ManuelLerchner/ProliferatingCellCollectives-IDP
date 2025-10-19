import os
import shutil
import subprocess

# from data_loader import load_latest_iteration
# from render_particles import draw_particles

BIN_FOLDER = f"../code/cpp/build/src"

END_RADIUS = 100


base_physics_config = {
    'LAMBDA': 1e-3,
}


def run_simulation(config, mode, LAMBDA, thr, proc):
    print(config)

    args = " ".join([f"-{key} {value}" for key, value in config.items()])

    args += f" -end_radius {END_RADIUS}"

    process = subprocess.run(
        f"export OMP_NUM_THREADS={thr} && make -j && mpirun -np {proc} ./cellcollectives -mode {mode} {args}", shell=True, cwd=BIN_FOLDER)

    target_folder = f"{BIN_FOLDER}/hybrid/vtk_output_{mode}/{thr}/{proc}"

    if os.path.exists(target_folder):
        shutil.rmtree(target_folder)

    shutil.copytree(f"{BIN_FOLDER}/vtk_output_{mode}",
                    target_folder)


for mode in ["hard"]:
    for LAMBDA in [1e-3]:
        for proc in reversed(range(1, 21)):
            for thr in reversed(range(1, 21)):
                total = thr * proc
                if total > 18:
                    continue

                print(
                    f"Running simulation with LAMBDA={LAMBDA}, mode={mode}, thr={thr}, proc={proc}")

                config = base_physics_config.copy()
                config["LAMBDA"] = LAMBDA
                run_simulation(config, mode, LAMBDA, thr, proc)
                print("Finished simulation")
