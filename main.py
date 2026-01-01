import taichi as ti
import tomli
import os
import shutil
import csv
import socket
import threading
import subprocess
import atexit
import sys

from env import Env

plot_proc = subprocess.Popen([sys.executable, "plot.py"])

def cleanup():
    plot_proc.terminate()
    try:
        plot_proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        plot_proc.kill()

def command_server():
    global display_phase
    global display_cells
    global display_ecm
    global cycle_scalpel
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('localhost', 65432))
    server.listen()

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
            if cmd == "cycle_scalpel":
                cycle_scalpel += 1
                if cycle_scalpel == 3:
                    cycle_scalpel = 0
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
cycle_scalpel = 0

env = Env(config)
threading.Thread(target=command_server, daemon=True).start()

gui = ti.GUI("Cell Cycle Sim", res=env.SCREEN_SIZE)

# Data output location
os.makedirs("data", exist_ok=True)

LMB_down = False

hour = 0

fieldnames = ["step", "population", "ecm"]

with open('data/data.csv', 'w') as csv_file:
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()

env.experimental_setup()

# Main Loop
with open('data/data.csv', 'a') as csv_file:
    while gui.running and (env.END_STEP == -1 or env.step[None] < env.END_STEP):
        # Input Handling
        mouse_pos = gui.get_cursor_pos()

        for e in gui.get_events():
            if e.type == ti.GUI.PRESS:
                if e.key == ti.GUI.LMB:
                    LMB_down = True
                if e.key == ti.GUI.RMB:
                    env.create_cell_kernel(mouse_pos[0], mouse_pos[1])
                if e.key == ti.GUI.SPACE:
                    env.paused = not env.paused
                if e.key == ti.GUI.SHIFT:
                    env.save_state()
            elif e.type == ti.GUI.RELEASE:
                if e.key == ti.GUI.LMB:
                    LMB_down = False

        # Deletion
        if env.paused and LMB_down and mouse_pos is not None:
            env.delete_cells_kernel(gui.get_cursor_pos()[0], gui.get_cursor_pos()[1], env.SCALPEL_RADIUS, cycle_scalpel)
            if not gui.is_pressed(ti.GUI.SHIFT):
                env.delete_ecm_kernel(gui.get_cursor_pos()[0], gui.get_cursor_pos()[1], env.SCALPEL_RADIUS, cycle_scalpel)

        if display_ecm:
            gui.circles(
                env.ecmHandler.posField.to_numpy()[:env.ecmHandler.count[None]],
                radius=env.CELL_RADIUS * env.SCREEN_SIZE[0] * env.CELL_RADIUS_SCALAR,
                color=0x252345
            )
        if display_cells:
            positions = env.fibroHandler.posField.to_numpy()[:env.fibroHandler.count[None]]
            if display_phase:
                gui.circles(
                    positions,
                    radius=env.CELL_RADIUS * env.SCREEN_SIZE[0] * env.CELL_RADIUS_SCALAR,
                    color=env.PHASE_COLORS[env.fibroHandler.phaseField.to_numpy()[:env.fibroHandler.count[None]]]
                )
            else:
                gui.circles(positions, radius=env.CELL_RADIUS * env.SCREEN_SIZE[0] * env.CELL_RADIUS_SCALAR, color=0xffffff)

        gui.show()

        if env.paused:
            continue

        # env.clear_topo_field()
        for _ in range(env.SUBSTEPS):  # Do multiple steps per frame for stability
            env.verlet_step_cells_kernel()
            env.border_constraints_cell_kernel()
            env.rebuild_grid_cells_kernel()
            env.handle_collisions_cells_kernel()

        env.update_kernel()

        env.rebuild_grid_ecm_kernel()

        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        info = {
            "step": env.step[None],
            "population": env.fibroHandler.count[None],
            "ecm": env.ecmHandler.count[None]
        }
        csv_writer.writerow(info)

        warn = ""
        if env.fibroHandler.count[None] == env.MAX_CELL_COUNT:
            warn = " | Warning: Max Cell Count Reached!"
        if env.step[None] % 10 == 0:
            print("Step: " + str(env.step[None]) + " | Hour: " + str(round(hour)) + " | Cells: " + str(env.fibroHandler.count[None]) + warn)

        hour += 24/env.CELL_CYCLE_DURATION[None]
        env.step[None] += 1

gui.close()