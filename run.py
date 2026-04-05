import taichi as ti
import tomli
import os
import shutil
import csv

from FISS.env import Env

class Trial:
    def __init__(self, display_visuals=False, display_cells=True, display_phase=True, display_ecm=True, scalpel="circle"):
        ti.init(arch=ti.gpu)

        if not os.path.exists("FISS/config.toml"):
            shutil.copyfile("FISS/defaultconfig.toml", "FISS/config.toml")
            print(f"Created config.toml from defaultconfig.toml")
        
        with open('FISS/config.toml', 'rb') as f:
            config = tomli.load(f)
            
        self.env = Env(config)
        self.gui = ti.GUI("Cell Cycle Sim", res=self.env.SCREEN_SIZE)
        self.hour = 0
        self.LMB_down = False

        self.display_visuals = display_visuals
        self.display_cells = display_cells
        self.display_phase = display_phase
        self.display_ecm = display_ecm
        self.scalpel = scalpel

        self.END_ON_HEAL = config["experiment"]["end_on_wound_heal"]
        self.end_timer = -1
        self.END_TIMER_DURATION = config["experiment"]["end_timer_duration"]

        self.data_fieldnames = ["step", "fibroblast_count", "ecm_count", "wound_area", "wound_width"]

        os.makedirs("FISS/data", exist_ok=True)

        with open('FISS/data/data.csv', 'w') as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=self.data_fieldnames)
            csv_writer.writeheader()

        self.scalpel_dict = {
            "circle": 0,
            "square": 1,
            "triangle": 2,
            "line": 3
        }


    def run_trial(self):
        self.env.experimental_setup()

        # Main Loop
        with open('FISS/data/data.csv', 'a') as csv_file:
            while self.env.END_STEP == -1 or self.env.step[None] < self.env.END_STEP:
                if self.env.INITIAL_WOUND_AREA is None:
                    self.env.INITIAL_WOUND_AREA = self.env.statisticHandler.get_wound_area()

                if self.display_visuals:
                    # Input Handling
                    mouse_pos = self.gui.get_cursor_pos()

                    for e in self.gui.get_events():
                        if e.type == ti.GUI.PRESS:
                            if e.key == ti.GUI.LMB:
                                self.LMB_down = True
                            if e.key == ti.GUI.RMB:
                                self.env.create_cell_kernel(mouse_pos[0], mouse_pos[1])
                            if e.key == ti.GUI.SPACE:
                                self.env.paused = not self.env.paused
                            if e.key == ti.GUI.ALT:
                                self.env.saveHandler.save_state()
                            if e.key == ti.GUI.ESCAPE:
                                self.gui.running = False
                        elif e.type == ti.GUI.RELEASE:
                            if e.key == ti.GUI.LMB:
                                self.LMB_down = False

                    # Deletion
                    if self.env.paused and self.gui.is_pressed(ti.GUI.SHIFT) and self.LMB_down and mouse_pos is not None:
                        self.env.delete_cells_kernel(self.gui.get_cursor_pos()[0], self.gui.get_cursor_pos()[1], self.env.WOUND_WIDTH, self.scalpel_dict[self.scalpel])
                        self.env.delete_ecm_kernel(self.gui.get_cursor_pos()[0], self.gui.get_cursor_pos()[1], self.env.WOUND_WIDTH, self.scalpel_dict[self.scalpel])

                    if self.display_ecm:
                        if self.env.DRAW_ECM_LINES:
                            self.gui.lines(self.env.ecmHandler.ecmConnectPosField.to_numpy()[:self.env.ecmHandler.count[None]],
                                      self.env.ecmHandler.posField.to_numpy()[:self.env.ecmHandler.count[None]],
                                      radius=1,
                                      color=0x353355)
                        else:
                            self.gui.circles(
                                self.env.ecmHandler.posField.to_numpy()[:self.env.ecmHandler.count[None]],
                                radius=self.env.CELL_RADIUS * self.env.SCREEN_SIZE[0] * self.env.CELL_RADIUS_SCALAR,
                                color=0x353355
                            )

                    if self.display_cells:
                        positions = self.env.fibroHandler.posField.to_numpy()[:self.env.fibroHandler.count[None]]
                        if self.display_phase:
                            self.gui.circles(
                                positions,
                                radius=self.env.CELL_RADIUS * self.env.SCREEN_SIZE[0] * self.env.CELL_RADIUS_SCALAR,
                                color=self.env.PHASE_COLORS[self.env.fibroHandler.phaseField.to_numpy()[:self.env.fibroHandler.count[None]]]
                            )
                        else:
                            self.gui.circles(positions, radius=self.env.CELL_RADIUS * self.env.SCREEN_SIZE[0] * self.env.CELL_RADIUS_SCALAR, color=0xffffff)

                    self.gui.show()
        
                if self.env.paused:
                    continue
        
                # self.env.clear_topo_field()
                for _ in range(self.env.SUBSTEPS):  # Do multiple steps per frame for stability
                    self.env.verlet_step_cells_kernel()
                    self.env.border_constraints_cell_kernel()
                    self.env.rebuild_grid_cells_kernel()
                    self.env.handle_collisions_cells_kernel()
        
                self.env.update_kernel()
        
                self.env.rebuild_grid_ecm_kernel()
                if self.env.step[None] % 30 == 0:
                    sum = 0
                    count = int(self.env.GRID_RES/10)
                    for i in range(count):
                        sum += self.env.statisticHandler.get_wound_width(i)
                    avg = sum/count
        
                    csv_writer = csv.DictWriter(csv_file, fieldnames=self.data_fieldnames)
                    info = {
                        "step": self.env.step[None],
                        "fibroblast_count": self.env.fibroHandler.count[None],
                        "ecm_count": self.env.ecmHandler.count[None],
                        "wound_area": self.env.statisticHandler.get_wound_area(),
                        "wound_width": avg
                    }
                    csv_writer.writerow(info)
                    csv_file.flush()
                    os.fsync(csv_file.fileno())
        
                if self.env.CAPTURE_DATA and self.env.step[None] % 60 == 0:
                    self.env.imagingHandler.capture_image(f"{self.env.DATA_PATH}/images/experiment_{self.env.EXPERIMENT_TIMESTAMP}")
        
                warn = ""
                if self.env.fibroHandler.count[None] == self.env.MAX_CELL_COUNT:
                    warn = " | Warning: Max Cell Count Reached!"
                if self.env.step[None] % 10 == 0:
                    print("Step: " + str(self.env.step[None]) + " | Hour: " + str(round(self.hour)) + " | Cells: " + str(self.env.fibroHandler.count[None]) + warn)
                    
                if self.END_ON_HEAL:
                    if self.env.INITIAL_WOUND != "none" and self.env.statisticHandler.get_wound_area() == 0 and self.end_timer == -1:
                        self.end_timer = self.END_TIMER_DURATION
            
                    if self.end_timer > 0:
                        self.end_timer -= 1
                    elif self.end_timer == 0:
                        break
        
                self.hour += 24/self.env.CELL_CYCLE_DURATION[None]
                self.env.step[None] += 1
        
        if self.env.SAVE_VIDEO:
            self.env.imagingHandler.save_video(f"{self.env.DATA_PATH}/images/experiment_{self.env.EXPERIMENT_TIMESTAMP}")
        
        self.gui.close()