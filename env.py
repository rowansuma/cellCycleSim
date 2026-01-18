import taichi as ti
import numpy as np
from datetime import datetime

from tools.imaging_handler import ImagingHandler
from tools.save_handler import SaveHandler
from particle.ecm import ECMHandler
from particle.fibroblast import FibroblastHandler
from tools.statistic_handler import StatisticHandler


@ti.data_oriented
class Env:
    def __init__(self, config):
        self.SCREEN_SIZE = (1000, 1000)

        # Constants
        self.DOMAIN_SIZE = config["experiment"]["domain_size"]
        self.INITIAL_MODE = config["experiment"]["initial_mode"]
        self.INITIAL_WOUND = config["experiment"]["initial_wound"]
        self.WOUND_WIDTH = config["experiment"]["wound_width"]
        self.END_STEP = config["experiment"]["end_step"]

        self.CAPTURE_DATA = config["data_collection"]["capture_data"]
        self.DATA_PATH = config["data_collection"]["data_path"]
        self.MAX_IMAGE_PIXEL_CELLS = config["data_collection"]["max_image_pixel_cells"]
        self.SAVE_VIDEO = config["data_collection"]["save_video"]
        self.VIDEO_FRAME_RATE = config["data_collection"]["video_frame_rate"]

        self.MAX_CELL_COUNT = config["cells"]["max_cell_count"]
        self.CELL_RADIUS_UM = config["cells"]["cell_radius"]
        self.CELL_RADIUS = self.CELL_RADIUS_UM/config["experiment"]["domain_size"]
        if self.CELL_RADIUS <= 0.0002: self.CELL_RADIUS_SCALAR = 0.00024/self.CELL_RADIUS
        self.CELL_REPULSION = config["cells"]["cell_repulsion"]
        self.REPRODUCTION_OFFSET = config["cells"]["reproduction_offset"]
        self.MAX_CELL_SPEED = config["cells"]["max_cell_speed"]*self.CELL_RADIUS
        self.CELL_TURN_SPEED = config["cells"]["cell_turn_speed"]
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
        self.DRAW_ECM_LINES = config["display"]["draw_ecm_lines"]

        self.EPSILON = 1e-5

        self.EXPERIMENT_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.INITIAL_WOUND_AREA = None

        # Taichi counters
        self.step = ti.field(dtype=ti.i32, shape=()) # 0

        # Data Collection
        self.topoField = ti.field(dtype=ti.f32, shape=(self.GRID_RES, self.GRID_RES))

        self.paused = False

        # Handlers
        self.fibroHandler = FibroblastHandler(self)
        self.ecmHandler = ECMHandler(self)

        self.saveHandler = SaveHandler({"fibroblast": self.fibroHandler, "ecm": self.ecmHandler})
        self.imagingHandler = ImagingHandler(self)
        self.statisticHandler = StatisticHandler(self)

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
            self.saveHandler.load_state("defaultstates/full_state")
        else:
            raise Exception("Invalid initial mode: " + self.INITIAL_MODE)

        if self.INITIAL_MODE == "single" and self.INITIAL_WOUND != "none":
            raise Exception("Wounds are not supported on the single cell initial setup.")

        if self.INITIAL_WOUND not in ["none", "circle", "triangle", "square", "line"]:
            raise Exception("Invalid wound configuration: " + self.INITIAL_WOUND)

        @ti.kernel
        def initial_wound_kernel():
            shape = {"circle": 0, "square": 1, "triangle": 2, "line": 3}[self.INITIAL_WOUND]

            self.fibroHandler.mark_for_deletion(0.5, 0.5, self.WOUND_WIDTH, shape)
            self.fibroHandler.write_buffer()
            self.fibroHandler.copy_back_buffer()
            self.fibroHandler.rebuild_grid()

            self.ecmHandler.mark_for_deletion(0.5, 0.5, self.WOUND_WIDTH, shape)
            self.ecmHandler.write_buffer()
            self.ecmHandler.copy_back_buffer()
            self.ecmHandler.rebuild_grid()

        if self.INITIAL_WOUND != "none":
            initial_wound_kernel()

    # CELL KERNELS

    @ti.kernel
    def verlet_step_cells_kernel(self):
        self.fibroHandler.verlet_step()

    @ti.kernel
    def border_constraints_cell_kernel(self):
        self.fibroHandler.border_constraints()

    @ti.kernel
    def handle_collisions_cells_kernel(self): # Collisions
        self.fibroHandler.handle_collisions()

    @ti.kernel
    def rebuild_grid_cells_kernel(self):
        self.fibroHandler.rebuild_grid()

    @ti.kernel
    def create_cell_kernel(self, posX: ti.f32, posY: ti.f32):
        self.fibroHandler.create(posX, posY)

    @ti.kernel
    def delete_cells_kernel(self, mouse_x: ti.f32, mouse_y: ti.f32, size: ti.f32, shape: ti.i32):
        self.fibroHandler.mark_for_deletion(mouse_x, mouse_y, size, shape)
        self.fibroHandler.write_buffer()
        self.fibroHandler.copy_back_buffer()
        self.fibroHandler.rebuild_grid()

    # ECM KERNELS

    @ti.kernel
    def rebuild_grid_ecm_kernel(self):
        self.ecmHandler.rebuild_grid()

    @ti.kernel
    def create_ecm_kernel(self, posX: ti.f32, posY: ti.f32):
        self.ecmHandler.create(posX, posY)

    @ti.kernel
    def delete_ecm_kernel(self, mouse_x: ti.f32, mouse_y: ti.f32, size: ti.f32, shape: ti.i32):
        self.ecmHandler.mark_for_deletion(mouse_x, mouse_y, size, shape)
        self.ecmHandler.write_buffer()
        self.ecmHandler.copy_back_buffer()
        self.ecmHandler.rebuild_grid()

    # LOGIC KERNELS

    @ti.kernel
    def update_kernel(self):
        self.fibroHandler.update()
        self.ecmHandler.update()