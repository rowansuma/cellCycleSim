import taichi as ti
import tomli
import os
import shutil
from env import Env
import csv
import socket
import threading
import subprocess
import atexit
import sys
import numpy as np

# Start plot.py as a subprocess
plot_proc = subprocess.Popen([sys.executable, "plot.py"])

# Ensure it gets killed when main.py exits
def cleanup():
    print("Cleaning up: killing plot.py")
    plot_proc.terminate()  # Send SIGTERM
    try:
        plot_proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        print("plot.py didn't terminate in time. Forcing kill.")
        plot_proc.kill()

def command_server():
    global display_phase
    global display_cells
    global display_ecm
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('localhost', 65432))  # Choose your port, must match client
    server.listen()

    print("Command server started, waiting for connections...")

    while True:
        conn, addr = server.accept()
        with conn:
            data = conn.recv(1024)
            if not data:
                continue
            cmd = data.decode('utf-8').strip()
            if cmd == "toggle_display_phase":
                display_phase = not display_phase
            if cmd == "toggle_display_cells":
                display_cells = not display_cells
            if cmd == "toggle_display_ecm":
                display_ecm = not display_ecm
            # You can add more commands here
            conn.sendall(b'OK')


atexit.register(cleanup)

ti.init(arch=ti.gpu)


if not os.path.exists("config.toml"):
    shutil.copyfile("defaultconfig.toml", "config.toml")
    print(f"Created config.toml from defaultconfig.toml")

with open('config.toml', 'rb') as f:
    config = tomli.load(f)

display_phase = True
display_cells = True
display_ecm = True


env = Env(config)
threading.Thread(target=command_server, daemon=True).start()

gui = ti.GUI("Cell Cycle Sim", res=env.SCREEN_SIZE)
env.initialize_board()

LMB_down = False

hour = 0

fieldnames = ["step", "population", "ecm", "g0", "g1", "s", "g2/m", "gene0", "gene1", "gene2", "gene3", "gene4", "gene5", "gene6", "gene7", "gene8", "gene9", "gene10"]

with open('data.csv', 'w') as csv_file:
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()

# Main Loop
# Write to CSV
with open('data.csv', 'a') as csv_file:
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

        if display_ecm:
            gui.circles(
                env.ecmPosField.to_numpy()[:env.ecmCount[None]],
                radius=env.CELL_RADIUS * env.SCREEN_SIZE[0] * env.CELL_RADIUS_SCALAR,
                color=0x252345
            )
        phases = env.phaseField.to_numpy()[:env.cellsAlive[None]]
        if display_cells:
            if display_phase:
                positions = env.posField.to_numpy()[:env.cellsAlive[None]]
                colors = env.PHASE_COLORS[phases]

                gui.circles(
                    positions,
                    radius=env.CELL_RADIUS * env.SCREEN_SIZE[0] * env.CELL_RADIUS_SCALAR,
                    color=colors
                )
            else:
                gui.circles(env.posField.to_numpy(), radius=env.CELL_RADIUS * env.SCREEN_SIZE[0] * env.CELL_RADIUS_SCALAR, color=0xffffff)

        # gui.arrows(
        #     orig=env.posField.to_numpy()[:env.cellsAlive[None]], direction=env.repulseField.to_numpy()[:env.cellsAlive[None]],
        #     radius=1,
        #     color=0xffffff
        # )
        gui.show()


        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)


        info = {
            "step": env.step[None],
            "population": env.cellsAlive[None],
            "ecm": env.ecmCount[None],
            "g0": np.count_nonzero(phases == 0)*100/env.cellsAlive[None],
            "g1": np.count_nonzero(phases == 1)*100/env.cellsAlive[None],
            "s": np.count_nonzero(phases == 2)*100/env.cellsAlive[None],
            "g2/m": (np.count_nonzero(phases == 3)+np.count_nonzero(phases == 4))*100/env.cellsAlive[None],
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
