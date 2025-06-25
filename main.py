import taichi as ti
from env import Env
import time

ti.init(arch=ti.gpu)

# Simulation Parameters
print("\nSimulation Parameters:\n")
defaults = input("Use Defaults? (y/n): ")
if defaults == "y":
    friction = 0.98
    radius = 0.001
    max = 1000000
    print(f"Using defaults:\nFriction = {friction}\nRadius = {radius}\nMax = {max}",)
elif defaults == "n":
    friction = float(input("Friction Multiplier (0 = no movement, 1 = no friction): "))
    radius = float(input("Cell Radius (lower = larger scale sim): "))
    max = int(input("Max Cells: "))
else:
    print("Invalid; quitting...")
    quit()

time.sleep(2)

env = Env(friction, radius, max)

gui = ti.GUI("Cell Cycle Sim", res=env.SCREEN_SIZE)
env.initialize_board()

# Main Loop
while gui.running:
    for _ in range(env.SUBSTEPS):  # Do multiple steps per frame for stability
        env.verlet_step()
        env.border_constraints()
        env.clear_grid()
        env.insert_into_grid()
        env.handle_collisions()
    env.handle_cell_cycle()
    gui.circles(env.posField.to_numpy(), radius=env.CELL_RADIUS*env.CELL_RADIUS_SCALAR, color=0x66ccff)
    # print(env.posField.to_numpy())
    gui.show()
    # print(env.step)
    # print(env.cellsAlive[None])
    print("Step: " + str(env.step[None]) + " | Cells: " + str(env.cellsAlive[None]))
    env.step[None] += 1