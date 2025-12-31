import taichi as ti
import numpy as np
from datetime import datetime
from pathlib import Path

from particle.ecm import ECMHandler
from particle.cell import CellHandler
from particle.fibroblast import FibroblastHandler


@ti.data_oriented
class Env:
    def __init__(self, config):
        self.SCREEN_SIZE = (1000, 1000)

        # Constants
        self.INITIAL_MODE = config["experiment"]["initial_mode"]
        self.INITIAL_WOUND = config["experiment"]["initial_wound"]

        self.MAX_CELL_COUNT = config["cells"]["max_cell_count"]
        self.CELL_RADIUS = config["cells"]["cell_radius"]
        if self.CELL_RADIUS <= 0.0002: self.CELL_RADIUS_SCALAR = 0.00024/self.CELL_RADIUS
        self.CELL_REPULSION = config["cells"]["cell_repulsion"]
        self.REPRODUCTION_OFFSET = config["cells"]["reproduction_offset"]
        self.MAX_CELL_SPEED = config["cells"]["max_cell_speed"]*self.CELL_RADIUS
        self.CELL_CYCLE_DURATION = ti.field(dtype=ti.i32, shape=())
        self.CCDPlaceholder = config["cells"]["cell_cycle_duration"]

        self.INHIBITION_RADIUS = config["inhibition"]["inhibition_radius"]
        self.INHIBITION_THRESHOLD = config["inhibition"]["inhibition_threshold"]
        self.INHIBITION_EXIT_THRESHOLD = config["inhibition"]["inhibition_exit_threshold"]
        self.INHIBITION_FACTOR = config["inhibition"]["inhibition_factor"]

        self.SUBSTEPS = config["environment"]["substeps"]
        self.GRID_SCALE_FACTOR = config["environment"]["grid_scale_factor"]
        self.GRID_RES = int(1 / (self.CELL_RADIUS * 2 * self.GRID_SCALE_FACTOR))
        self.MAX_PARTICLES_PER_GRID_CELL = config["environment"]["max_particles_per_grid_cell"]
        self.FRICTION = config["environment"]["friction"]

        self.MIN_ECM_PERIOD = config["ecm"]["min_ecm_period"]
        self.MAX_ECM_COUNT = config["ecm"]["max_ecm_count"]
        self.ECM_DETECTION_RADIUS = config["ecm"]["ecm_detection_radius"]*self.CELL_RADIUS
        self.ECM_THRESHOLD = config["ecm"]["ecm_threshold"]
        self.ECM_AVOIDANCE_STRENGTH = config["ecm"]["ecm_avoidance_strength"]

        self.PHASE_COLORS = np.array(config["display"]["phase_colors"], dtype=np.uint32)
        self.CELL_RADIUS_SCALAR = config["display"]["cell_radius_scalar"]

        self.GENE_POINTS = ti.Vector.field(2, dtype=ti.f32, shape=(11, 2))
        self.GENE_POINTS.from_numpy(np.array(config["genes"]["gene_points"], dtype=np.float32))
        self.GENE_VARIATION = config["genes"]["gene_variation"]
        self.GENE_CYCLE_LENGTH = config["genes"]["gene_cycle_length"]

        self.SCALPEL_RADIUS = config["tools"]["scalpel_radius"]

        self.EPSILON = 1e-5

        # Taichi counters
        self.step = ti.field(dtype=ti.i32, shape=()) # 0

        # Data Collection
        self.topoField = ti.field(dtype=ti.f32, shape=(self.GRID_RES, self.GRID_RES))

        self.paused = False
        self.loaded = False

        # Handlers
        self.fibroHandler = FibroblastHandler(self)
        self.ecmHandler = ECMHandler(self)

        self.initialize_board()

    @ti.kernel
    def initialize_board(self): # Board Init, assign taichi fields
        self.CELL_CYCLE_DURATION[None] = self.CCDPlaceholder
        self.step[None] = 0

        self.fibroHandler.clear_fields()
        self.ecmHandler.clear_fields()

    def experimental_setup(self):
        if self.INITIAL_MODE == "single":
            self.create_cell_kernel(0.5, 0.5)
        elif self.INITIAL_MODE == "full":
            self.load_state("defaultstates/full_state")
        else:
            raise Exception("Invalid initial mode: " + self.INITIAL_MODE)

        if self.INITIAL_MODE == "single" and self.INITIAL_WOUND != "none":
            raise Exception("Wounds are not supported on the single cell initial setup.")

        if self.INITIAL_WOUND not in ["none", "circle", "triangle", "square"]:
            raise Exception("Invalid wound configuration: " + self.INITIAL_WOUND)

        @ti.kernel
        def initial_wound_kernel():
            shape = {"circle": 0, "square": 1, "triangle": 2}[self.INITIAL_WOUND]

            self.fibroHandler.mark_for_deletion(0.5, 0.5, self.SCALPEL_RADIUS, shape)
            self.fibroHandler.write_buffer()
            self.fibroHandler.copy_back_buffer()
            self.fibroHandler.rebuild_grid()

            self.ecmHandler.mark_for_deletion(0.5, 0.5, self.SCALPEL_RADIUS, shape)
            self.ecmHandler.write_buffer()
            self.ecmHandler.copy_back_buffer()
            self.ecmHandler.rebuild_grid()

        if self.INITIAL_WOUND != "none":
            initial_wound_kernel()

    def save_state(self):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        save_dir = Path("savestates") / f"save_{timestamp}"
        save_dir.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            save_dir / f"fibroblast_state.npz",
            **self.fibroHandler.export_state()
        )
        np.savez_compressed(
            save_dir / f"ecm_state.npz",
            **self.ecmHandler.export_state()
        )

    def load_state(self, path):
        self.fibroHandler.load_state(np.load(path+"/fibroblast_state.npz"))
        self.ecmHandler.load_state(np.load(path+"/ecm_state.npz"))

    @ti.kernel
    def verlet_step_cells_kernel(self):
        self.fibroHandler.verlet_step()

    @ti.kernel
    def border_constraints_cell_kernel(self):
        self.fibroHandler.border_constraints()

    @ti.kernel
    def rebuild_grid_cells_kernel(self):
        self.fibroHandler.rebuild_grid()

    @ti.kernel
    def handle_collisions_cells_kernel(self): # Collisions
        self.fibroHandler.handle_collisions()

    @ti.kernel
    def update_kernel(self): # Collisions
        self.fibroHandler.update()
        self.ecmHandler.update()

    @ti.kernel
    def rebuild_grid_ecm_kernel(self):
        self.ecmHandler.rebuild_grid()

    @ti.kernel
    def mark_cells_for_deletion_kernel(self, mouse_x: ti.f32, mouse_y: ti.f32, size: ti.f32, shape: ti.i32):
        self.fibroHandler.mark_for_deletion(mouse_x, mouse_y, size, shape)

    @ti.kernel
    def mark_ecm_for_deletion_kernel(self, mouse_x: ti.f32, mouse_y: ti.f32, size: ti.f32, shape: ti.i32):
        self.ecmHandler.mark_for_deletion(mouse_x, mouse_y, size, shape)

    # @ti.kernel
    # def clear_topo_field(self):
    #     for i, j in self.topoField:
    #         self.topoField[i, j] = 0.0
    #
    # @ti.kernel
    # def accumulate_density(self):
    #     for i in range(self.fibroHandler.count[None]):
    #         pos = self.fibroHandler.posField[i]
    #         x = ti.min(ti.max(int(pos[0] * self.GRID_RES), 0), self.GRID_RES - 1)
    #         y = ti.min(ti.max(int(pos[1] * self.GRID_RES), 0), self.GRID_RES - 1)
    #         ti.atomic_add(self.topoField[x, y], 1.0)

    @ti.kernel
    def write_buffer_cells_kernel(self): # Write to-be-swapped cells to buffer
        self.fibroHandler.write_buffer()

    @ti.kernel
    def write_buffer_ecm_kernel(self):
        self.ecmHandler.write_buffer()

    @ti.kernel
    def copy_back_buffer_cells_kernel(self): # Write the buffer cells back to main fields
        self.fibroHandler.copy_back_buffer()

    @ti.kernel
    def copy_back_buffer_ecm_kernel(self):
        self.ecmHandler.copy_back_buffer()

    @ti.kernel
    def create_cell_kernel(self, posX: ti.f32, posY: ti.f32):
        self.fibroHandler.create(posX, posY)

    @ti.kernel
    def create_ecm_kernel(self, posX: ti.f32, posY: ti.f32):
        self.ecmHandler.create(posX, posY)