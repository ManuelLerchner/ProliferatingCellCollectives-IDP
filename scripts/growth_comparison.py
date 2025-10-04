import os
import shutil
import subprocess
import time

# from data_loader import load_latest_iteration
# from render_particles import draw_particles

BIN_FOLDER = f"../code/cpp/build/src"

END_RADIUS = 100


base_physics_config = {
    'LAMBDA': 1e-3,
}


def run_simulation(config, mode, LAMBDA):
    print(config)

    args = " ".join([f"-{key} {value}" for key, value in config.items()])

    args += f" -end_radius {END_RADIUS}"

    process = subprocess.run(
        f"make -j && srun ./cellcollectives -mode {mode} {args}", shell=True, cwd=BIN_FOLDER)

    # copy vtk_output_{mode} folder to growth_comparison_data/vtk_output_{mode}_{lambda}
    # delete old folder if it exists
    if os.path.exists(f"{BIN_FOLDER}/growth_comparison_data/vtk_output_{mode}_{LAMBDA}"):
        shutil.rmtree(
            f"{BIN_FOLDER}/growth_comparison_data/vtk_output_{mode}_{LAMBDA:1e}")

    shutil.copytree(f"{BIN_FOLDER}/vtk_output_{mode}",
                    f"{BIN_FOLDER}/growth_comparison_data/vtk_output_{mode}_{LAMBDA:1e}/")


for mode in ["hard","soft"]:
    for LAMBDA in [1e-2,1e-3,1e-4]:
        config = base_physics_config.copy()
        config["LAMBDA"] = LAMBDA
        run_simulation(config, mode, LAMBDA)
        print("Finished simulation")
