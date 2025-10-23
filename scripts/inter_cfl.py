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


def run_simulation(config, mode, cfl_factor, thr, proc):
    print(config)

    args = " ".join([f"-{key} {value}" for key, value in config.items()])

    args += f" -end_radius {END_RADIUS}"

    process = subprocess.run(
        f"export OMP_NUM_THREADS=1 && make -j && mpirun -np {proc} ./cellcollectives -mode {mode} -log_every_colony_radius_delta 5 -cfl_factor {cfl_factor} {args}", shell=True, cwd=BIN_FOLDER)

    target_folder = f"{BIN_FOLDER}/cfl/vtk_output_{mode}/{cfl_factor}"

    if os.path.exists(target_folder):
        shutil.rmtree(target_folder)

    shutil.copytree(f"{BIN_FOLDER}/vtk_output_{mode}",
                    target_folder)


max_threads = os.cpu_count()

for mode in ["hard"]:
    for proc in [18]:
        for thr in [1]:
            for cfl_factor in ([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]):

                print(
                    f"Running simulation with CFL_FACTOR={cfl_factor}, mode={mode}, thr={thr}, proc={proc}")

                config = base_physics_config.copy()
                config["CFL_FACTOR"] = cfl_factor
                run_simulation(config, mode, cfl_factor, thr, proc)
                print("Finished simulation")
