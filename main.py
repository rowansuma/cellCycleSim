import taichi as ti
from env import Env
import time
import csv

ti.init(arch=ti.gpu)

# Simulation Parameters
print("\nSimulation Parameters:\n")
defaults = input("Use Defaults? (y/n): ")
if defaults == "y":
    radius = 0.002
    max = 100000
    freq = 60
    print(f"Using defaults:\nRadius = {radius}\nMax = {max}\nDuration = {freq}")
elif defaults == "n":
    radius = float(input("Cell Radius (lower = larger scale sim): "))
    max = int(input("Max Cells (larger = more memory):"))
    freq = int(input("Cell Cycle Duration (in steps): "))
else:
    print("Invalid; quitting...")
    quit()

time.sleep(2)

env = Env(radius, max, freq)

gui = ti.GUI("Cell Cycle Sim", res=env.SCREEN_SIZE)
env.initialize_board()

LMB_down = False

fieldnames = ["step", "population"]

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
        env.handle_collisions()
    env.handle_cell_cycle()
    env.clamp_cell_count()

    # Mouse Button Handling
    mouse_pos = gui.get_cursor_pos()

    for e in gui.get_events():
        if e.key == ti.GUI.LMB:
            if e.type == ti.GUI.PRESS:
                LMB_down = True
            elif e.type == ti.GUI.RELEASE:
                LMB_down = False

        if LMB_down:

            pass

    gui.circles(env.posField.to_numpy(), radius=env.CELL_RADIUS * env.SCREEN_SIZE[0] * env.CELL_RADIUS_SCALAR, color=0x66ccff)
    # print(env.posField.to_numpy())
    gui.show()

    # Write to CSV
    with open('data.csv', 'a') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        info = {
            "step": env.step[None],
            "population": env.cellsAlive[None]
        }
        csv_writer.writerow(info)



    # print(env.step)
    # print(env.cellsAlive[None])
    warn = ""
    if env.cellsAlive[None] == env.MAX_CELL_COUNT:
        warn = " | Warning: Max Cell Count Reached!"
    print("Step: " + str(env.step[None]) + " | Cells: " + str(env.cellsAlive[None]) + warn)
    env.step[None] += 1