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
        f"export OMP_NUM_THREADS={thr} && make -j && mpirun -np {proc} ./cellcollectives -mode {mode} -log_every_colony_radius_delta 5 {args}", shell=True, cwd=BIN_FOLDER)

    target_folder = f"{BIN_FOLDER}/hybrid/vtk_output_{mode}/{thr}/{proc}"

    if os.path.exists(target_folder):
        shutil.rmtree(target_folder)

    shutil.copytree(f"{BIN_FOLDER}/vtk_output_{mode}",
                    target_folder)


max_threads = os.cpu_count()

for mode in ["hard"]:
    for LAMBDA in [1e-3]:
        for p in reversed(range(0, 10, 1)):
            for t in reversed(range(0, 10, 1)):
                proc = 2 ** p
                thr = 2 ** t

                total = thr * proc
                if total > max_threads or total < 1:
                    continue

                print(
                    f"Running simulation with LAMBDA={LAMBDA}, mode={mode}, thr={thr}, proc={proc}")

                config = base_physics_config.copy()
                config["LAMBDA"] = LAMBDA
                run_simulation(config, mode, LAMBDA, thr, proc)
                print("Finished simulation")
