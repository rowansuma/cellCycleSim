import taichi as ti
from env import Env

ti.init(arch=ti.gpu)

env = Env()

gui = ti.GUI("Cell Cycle Sim", res=env.SCREEN_SIZE)
env.initialize_board()

step = 0
while gui.running:
    for _ in range(env.SUBSTEPS):  # Do multiple steps per frame for stability
        env.verlet_step()
        env.border_constraints()
        env.solve_collisions()
    gui.circles(env.posField.to_numpy(), radius=env.CELL_RADIUS*env.CELL_RADIUS_SCALAR, color=0x66ccff)
    gui.show()
    print(step)
    step += 1