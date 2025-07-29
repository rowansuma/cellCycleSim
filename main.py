import taichi as ti
import numpy as np
from env import Env
import time
import csv

ti.init(arch=ti.gpu)

defaults = [0.002, 150000, 60, 0.05]
display_phase = True
write_csv = True

# Simulation Parameters
print("\nSimulation Parameters:\n")
use_defaults = input("Use Defaults? (y/n): ")
if use_defaults == "y":
    radius = defaults[0]
    max_cells = defaults[1]
    freq = defaults[2]
    scalpel_size = defaults[3]
    print(f"Using defaults:\nRadius = {radius}\nMax = {max_cells}\nDuration = {freq}\nScalpel Size = {scalpel_size}")
elif use_defaults == "n":
    radius = float(input(f"Cell Radius (default: {defaults[0]}): "))
    max_cells = int(input(f"Max Cells (default: {defaults[1]}): "))
    freq = int(input(f"Cell Cycle Duration (default: {defaults[2]}): "))
    scalpel_size = float(input(f"Scalpel Size (default: {defaults[3]}): "))
else:
    print("Invalid; quitting...")
    quit()

time.sleep(0.5)

env = Env(radius, max_cells, freq, scalpel_size)

gui = ti.GUI("Cell Cycle Sim", res=env.SCREEN_SIZE)
env.initialize_board()

LMB_down = False

hour = 0

fieldnames = ["step", "population", "ecm", "gene0", "gene1", "gene2", "gene3", "gene4", "gene5", "gene6", "gene7", "gene8", "gene9", "gene10"]

with open('data.csv', 'w') as csv_file:
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()

# Main Loop
while gui.running:
    for _ in range(env.SUBSTEPS):  # Do multiple steps per frame for stability
        env.verlet_step()
        env.border_constraints()
        env.clear_grid()
        env.insert_into_grid()
        env.apply_locomotion()
        env.handle_collisions()
    env.clear_ecm_grid()
    env.insert_ecm_into_grid()
    env.handle_cell_cycle()
    env.clamp_cell_count()
    print("ECM:", env.ecmCount[None])

    # Mouse Button Handling
    mouse_pos = gui.get_cursor_pos()

    for e in gui.get_events():
        if e.key == ti.GUI.LMB:
            if e.type == ti.GUI.PRESS:
                LMB_down = True
            elif e.type == ti.GUI.RELEASE:
                LMB_down = False
        if e.key == ti.GUI.RMB and e.type == ti.GUI.PRESS:
            env.create_cell(mouse_pos[0], mouse_pos[1])

    # Deletion
    if LMB_down and mouse_pos is not None:
        env.mark_for_deletion(mouse_pos[0], mouse_pos[1], env.SCALPEL_RADIUS)
        env.write_buffer_cells()
        env.copy_back_buffer()
        if not gui.is_pressed(ti.GUI.SHIFT):
            env.mark_ecm_for_deletion(mouse_pos[0], mouse_pos[1], env.SCALPEL_RADIUS)
            env.write_buffer_ecm()
            env.copy_back_buffer_ecm()


    gui.circles(
        env.ecmPosField.to_numpy()[:env.ecmCount[None]],
        radius=env.CELL_RADIUS * env.SCREEN_SIZE[0] * env.CELL_RADIUS_SCALAR,
        color=0x252345
    )

    if display_phase:
        positions = env.posField.to_numpy()[:env.cellsAlive[None]]
        phases = env.phaseField.to_numpy()[:env.cellsAlive[None]]
        colors = env.PHASE_COLORS[phases]

        gui.circles(
            positions,
            radius=env.CELL_RADIUS * env.SCREEN_SIZE[0] * env.CELL_RADIUS_SCALAR,
            color=colors
        )
    else:
        gui.circles(env.posField.to_numpy(), radius=env.CELL_RADIUS * env.SCREEN_SIZE[0] * env.CELL_RADIUS_SCALAR, color=0x66ccff)

    # gui.arrows(
    #     orig=env.posField.to_numpy()[:env.cellsAlive[None]], direction=env.repulseField.to_numpy()[:env.cellsAlive[None]],
    #     radius=1,
    #     color=0xffffff
    # )
    gui.show()

    # Write to CSV
    if write_csv:
        with open('data.csv', 'a') as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            info = {
                "step": env.step[None],
                "population": env.cellsAlive[None],
                "ecm": env.ecmCount[None],
                "gene0": env.geneField[0][0],
                "gene1": env.geneField[0][1],
                "gene2": env.geneField[0][2],
                "gene3": env.geneField[0][3],
                "gene4": env.geneField[0][4],
                "gene5": env.geneField[0][5],
                "gene6": env.geneField[0][6],
                "gene7": env.geneField[0][7],
                "gene8": env.geneField[0][8],
                "gene9": env.geneField[0][9],
                "gene10": env.geneField[0][10],
            }
            csv_writer.writerow(info)




    warn = ""
    if env.cellsAlive[None] == env.MAX_CELL_COUNT:
        warn = " | Warning: Max Cell Count Reached!"
    if env.step[None] % 10 == 0:
        print("Step: " + str(env.step[None]) + " | Hour: " + str(round(hour)) + " | Cells: " + str(env.cellsAlive[None]) + warn)

    hour += 24/env.CELL_CYCLE_DURATION[None]

    env.step[None] += 1
