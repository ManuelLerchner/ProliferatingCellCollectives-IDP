import os
import shutil
import subprocess
import time

from data_loader import load_latest_iteration
from render_particles import draw_particles

BIN_FOLDER = f"../code/cpp/build/src"

END_RADIUS = 50


base_physics_config = {
    'LAMBDA': 1e-3,
}


def run_simulation(config, mode, LAMBDA):
    print(config)

    args = " ".join([f"-{key} {value}" for key, value in config.items()])

    args += f" -end_radius {END_RADIUS}"

    process = subprocess.Popen(
        f"make -j && mpirun -np 16 ./cellcollectives -mode {mode} {args}", shell=True, cwd=BIN_FOLDER, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    t_last = 0
    while process.stdout.readable():

        try:
            if time.time() - t_last > 50:
                t_last = time.time()
                data = load_latest_iteration(
                    f"{BIN_FOLDER}/vtk_output_{mode}/data")

                draw_particles(data["particles"])

        except Exception as e:
            print(e)

        line = process.stdout.readline()

        if not line:
            break

        print(str(line.strip() )+ (" " * 100), end="\r")

    # copy vtk_output_{mode} folder to growth_comparison_data/vtk_output_{mode}_{lambda}
    # delete old folder if it exists
    if os.path.exists(f"{BIN_FOLDER}/growth_comparison_data/vtk_output_{mode}_{LAMBDA}"):
        shutil.rmtree(
            f"{BIN_FOLDER}/growth_comparison_data/vtk_output_{mode}_{LAMBDA:1e}")

    shutil.copytree(f"{BIN_FOLDER}/vtk_output_{mode}",
                    f"{BIN_FOLDER}/growth_comparison_data/vtk_output_{mode}_{LAMBDA:1e}/")


for mode in ["soft", "hard"]:
    for LAMBDA in [1e-4, 1e-3, 1e-2, 1e-1]:
        config = base_physics_config.copy()
        config["LAMBDA"] = LAMBDA
        run_simulation(config, mode, LAMBDA)
