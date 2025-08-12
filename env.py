import taichi as ti
import numpy as np

@ti.data_oriented
class Env:
    def __init__(self, config):
        self.SCREEN_SIZE = (1000, 1000)

        # Constants
        self.MAX_CELL_COUNT = config["cells"]["max_cell_count"]
        self.CELL_RADIUS = config["cells"]["cell_radius"]
        if self.CELL_RADIUS <= 0.0002: self.CELL_RADIUS_SCALAR = 0.00024/self.CELL_RADIUS
        self.CELL_REPULSION = config["cells"]["cell_repulsion"]
        self.REPRODUCTION_OFFSET = config["cells"]["reproduction_offset"]
        self.MAX_CELL_SPEED = config["cells"]["max_cell_speed"]*self.CELL_RADIUS
        self.CELL_CYCLE_DURATION = ti.field(dtype=ti.i32, shape=())
        self.CCDPlaceholder = config["cells"]["cell_cycle_duration"]

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

        # Taichi counters
        self.step = ti.field(dtype=ti.i32, shape=()) # 0
        self.cellsAlive = ti.field(dtype=ti.i32, shape=()) # 1
        self.ecmCount = ti.field(dtype=ti.i32, shape=()) # 0

        # Grid
        self.grid = ti.field(dtype=ti.i32, shape=(self.GRID_RES, self.GRID_RES, self.MAX_PARTICLES_PER_GRID_CELL))
        self.gridCount = ti.field(dtype=ti.i32, shape=(self.GRID_RES, self.GRID_RES))  # how many particles per cell

        # ECM Grid
        self.ecmGrid = ti.field(dtype=ti.i32, shape=(self.GRID_RES, self.GRID_RES, self.MAX_PARTICLES_PER_GRID_CELL))
        self.ecmGridCount = ti.field(dtype=ti.i32, shape=(self.GRID_RES, self.GRID_RES))

        # Cell Info
        self.posField = ti.Vector.field(2, dtype=ti.f32, shape=self.MAX_CELL_COUNT) # Current Pos
        self.prevPosField = ti.Vector.field(2, dtype=ti.f32, shape=self.MAX_CELL_COUNT) # Previous Pos
        self.lastDivField = ti.field(dtype=ti.i32, shape=self.MAX_CELL_COUNT)
        self.inhibitionField = ti.field(dtype=ti.f32, shape=self.MAX_CELL_COUNT)
        self.phaseField = ti.field(dtype=ti.i32, shape=self.MAX_CELL_COUNT)
        self.mvmtField = ti.Vector.field(3, dtype=ti.f32, shape=self.MAX_CELL_COUNT)
        self.cycleDurField = ti.field(dtype=ti.i32, shape=self.MAX_CELL_COUNT)  # Cycle duration
        self.geneField = ti.Vector.field(11, dtype=ti.f32, shape=self.MAX_CELL_COUNT)
        self.lastECMField = ti.field(dtype=ti.i32, shape=self.MAX_CELL_COUNT)
        self.ecmPeriodField = ti.field(dtype=ti.f32, shape=self.MAX_CELL_COUNT)

        # Cell Info Buffers
        self.posFieldBuffer = ti.Vector.field(2, dtype=ti.f32, shape=self.MAX_CELL_COUNT)
        self.prevPosFieldBuffer = ti.Vector.field(2, dtype=ti.f32, shape=self.MAX_CELL_COUNT)
        self.lastDivFieldBuffer = ti.field(dtype=ti.i32, shape=self.MAX_CELL_COUNT)
        self.inhibitionFieldBuffer = ti.field(dtype=ti.f32, shape=self.MAX_CELL_COUNT)
        self.phaseFieldBuffer = ti.field(dtype=ti.i32, shape=self.MAX_CELL_COUNT)
        self.mvmtFieldBuffer = ti.Vector.field(3, dtype=ti.f32, shape=self.MAX_CELL_COUNT)
        self.cycleDurFieldBuffer = ti.field(dtype=ti.i32, shape=self.MAX_CELL_COUNT)
        self.geneFieldBuffer = ti.Vector.field(11, dtype=ti.f32, shape=self.MAX_CELL_COUNT)
        self.lastECMFieldBuffer = ti.field(dtype=ti.i32, shape=self.MAX_CELL_COUNT)
        self.ecmPeriodFieldBuffer = ti.field(dtype=ti.f32, shape=self.MAX_CELL_COUNT)

        # ECM Info
        self.ecmPosField = ti.Vector.field(2, dtype=ti.f32, shape=self.MAX_ECM_COUNT) # Current Pos
        self.ecmPosFieldBuffer = ti.Vector.field(2, dtype=ti.f32, shape=self.MAX_ECM_COUNT)
        self.bufferCountECM = ti.field(dtype=ti.i32, shape=())

        # Deletion
        self.toDelete = ti.field(dtype=ti.i32, shape=self.MAX_CELL_COUNT)
        self.toDeleteECM = ti.field(dtype=ti.i32, shape=self.MAX_ECM_COUNT)
        self.bufferCount = ti.field(dtype=ti.i32, shape=())

        # Data Collection
        self.topoField = ti.field(dtype=ti.f32, shape=(self.GRID_RES, self.GRID_RES))

        self.initialize_board()

    @ti.kernel
    def initialize_board(self): # Board Init, assign taichi fields
        self.cellsAlive[None] = 1
        self.ecmCount[None] = 0
        self.CELL_CYCLE_DURATION[None] = self.CCDPlaceholder
        self.step[None] = 0

        for index in range(self.MAX_CELL_COUNT): # Create Placeholders
            self.posField[index] = [-1, -1]
            self.prevPosField[index] = self.posField[index]
            self.lastDivField[index] = -1
            self.inhibitionField[index] = -1
            self.phaseField[index] = -1
            self.mvmtField[index] = [-1, -1, -1]
            self.cycleDurField[index] = -1
            self.geneField[index] = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
            self.lastECMField[index] = -1
            self.ecmPeriodField[index] = -1

        self.posField[0] = [0.5, 0.5] # Init First Cell
        self.prevPosField[0] = self.posField[0]
        self.lastDivField[0] = 0
        self.inhibitionField[0] = 0
        self.phaseField[0] = 1
        self.mvmtField[0] = [0, -1, self.MAX_CELL_SPEED]
        self.cycleDurField[0] = self.CELL_CYCLE_DURATION[None] + int((ti.random() - 0.5) * 10)
        self.lastECMField[0] = 0
        self.ecmPeriodField[0] = 0
        self.calc_starting_genes(0)

    @ti.func
    def calc_starting_genes(self, idx: ti.i32):
        for gene_idx in range(11):
            firstPoint = self.GENE_POINTS[gene_idx, 0]
            lastPoint = self.GENE_POINTS[gene_idx, 1]
            beforePoint = (lastPoint[0]-self.GENE_CYCLE_LENGTH, lastPoint[1])
            m = (firstPoint[1]-beforePoint[1]) / (firstPoint[0]-beforePoint[0])
            b = firstPoint[1] - m * firstPoint[0]
            self.geneField[idx][gene_idx] = b
        ti.static_print(self.geneField[idx])

    @ti.func
    def update_genes(self, idx: ti.i32):
        cycleTime = (self.step[None] - self.lastDivField[idx]) / self.CELL_CYCLE_DURATION[None]
        for gene_idx in range(11):
            # Find last and next points
            last_x = -float('inf')
            next_x = float('inf')
            last_point = ti.Vector([0.0, 0.0])
            next_point = ti.Vector([0.0, 0.0])
            max_x = -float('inf')
            min_x = float('inf')
            max_point = ti.Vector([0.0, 0.0])
            min_point = ti.Vector([0.0, 0.0])

            # Normalize points
            for point_idx in range(2):
                point = self.GENE_POINTS[gene_idx, point_idx]
                x = point[0] / 50
                y = point[1]
                if cycleTime >= x > last_x:
                    last_x = x
                    last_point = ti.Vector([x, y])
                if cycleTime <= x < next_x:
                    next_x = x
                    next_point = ti.Vector([x, y])
                if x > max_x:
                    max_x = x
                    max_point = ti.Vector([x, y])
                if x < min_x:
                    min_x = x
                    min_point = ti.Vector([x, y])

            # Edge cases
            if last_x == -float('inf'):
                last_point = ti.Vector([max_x - 1.0, max_point[1]])
            if next_x == float('inf'):
                next_point = ti.Vector([min_x + 1.0, min_point[1]])

            x0 = last_point[0]
            y0 = last_point[1]
            x1 = next_point[0]
            y1 = next_point[1]

            if x1 != x0 and x0 <= cycleTime <= x1:
                increment = (y1 - y0) / (x1 - x0) / self.CELL_CYCLE_DURATION[None]
                # Add small random variation
                noise = (ti.random() - 0.5) * self.GENE_VARIATION
                self.geneField[idx][gene_idx] += increment + noise
                if (y0 < y1 < self.geneField[idx][gene_idx]) or (y0 > y1 > self.geneField[idx][gene_idx]):
                    self.geneField[idx][gene_idx] = y1
            elif cycleTime < x0:
                self.geneField[idx][gene_idx] = y0
            elif cycleTime > x1:
                self.geneField[idx][gene_idx] = y1

    @ti.kernel
    def verlet_step(self): # Verlet Calculations for Motion (velocity calc)
        for index in range(self.cellsAlive[None]):
            tempVar = self.posField[index]
            self.posField[index] += (self.posField[index] - self.prevPosField[index])*self.FRICTION
            self.prevPosField[index] = tempVar

    @ti.kernel
    def border_constraints(self): # Stop cells from exiting view
        for i in range(self.cellsAlive[None]):
            at_border = False
            for j in ti.static(range(2)):
                if self.posField[i][j] < 0:
                    self.posField[i][j] = 0
                    v = self.posField[i][j] - self.prevPosField[i][j]
                    self.prevPosField[i][j] = self.posField[i][j] + v * 0.5
                    at_border = True

                if self.posField[i][j] > 1:
                    self.posField[i][j] = 1
                    v = self.posField[i][j] - self.prevPosField[i][j]
                    self.prevPosField[i][j] = self.posField[i][j] + v * 0.5
                    at_border = True

    @ti.kernel
    def apply_locomotion(self):
        for i in range(self.cellsAlive[None]):
            repulse_vec = ti.Vector([0.0, 0.0])
            ecm_count = 0
            if self.phaseField[i] != 0:
                gridcell_x = ti.min(ti.max(int(self.posField[i][0] * self.GRID_RES), 0), self.GRID_RES - 1)
                gridcell_y = ti.min(ti.max(int(self.posField[i][1] * self.GRID_RES), 0), self.GRID_RES - 1)
                ecm_centroid = ti.Vector([0.0, 0.0])
                for offset in ti.static(ti.grouped(ti.ndrange((-2, 3), (-2, 3)))):
                    cx = gridcell_x + offset[0]
                    cy = gridcell_y + offset[1]
                    if 0 <= cx < self.GRID_RES and 0 <= cy < self.GRID_RES:
                        count = self.ecmGridCount[cx, cy]
                        for j in range(count):
                            ecm_idx = self.ecmGrid[cx, cy, j]
                            dx = self.posField[i] - self.ecmPosField[ecm_idx]
                            dist = dx.norm()
                            if dist < self.ECM_DETECTION_RADIUS:
                                ecm_centroid += self.ecmPosField[ecm_idx]
                                ecm_count += 1
                if ecm_count > 0:
                    ecm_avg_pos = ecm_centroid/ecm_count
                    delta = self.posField[i] - ecm_avg_pos
                    if ti.math.length(delta) > 0.005:
                        repulse_vec = ti.math.normalize(delta)*self.ECM_AVOIDANCE_STRENGTH

            if ti.random() < 0.3:
                r = ti.random()
                val = 0
                if r < 1/3:
                    val = -1
                elif r < 2/3:
                    val = 0
                else:
                    val = 1
                self.mvmtField[i][1] = val
            self.mvmtField[i][0] += self.mvmtField[i][1] * 0.01
            angle = self.mvmtField[i][0] * 2 * ti.math.pi
            mvmtVector = self.mvmtField[i][2] * ti.Vector([ti.cos(angle), ti.sin(angle)])
            self.posField[i] += (mvmtVector+repulse_vec)/(ti.math.log(ecm_count+5)-0.6)

    @ti.kernel
    def handle_collisions(self): # Collisions
        for i in range(self.cellsAlive[None]):
            pos_i = self.posField[i]
            gridcell_x = ti.min(ti.max(int(pos_i[0] * self.GRID_RES), 0), self.GRID_RES - 1)
            gridcell_y = ti.min(ti.max(int(pos_i[1] * self.GRID_RES), 0), self.GRID_RES - 1)

            has_neighbors = False
            for offset in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2)))):
                cx = gridcell_x + offset[0]
                cy = gridcell_y + offset[1]
                if 0 <= cx < self.GRID_RES and 0 <= cy < self.GRID_RES:
                    count = self.gridCount[cx, cy]
                    for j in range(count):
                        other = self.grid[cx, cy, j]
                        if other != i:
                            dx = self.posField[i] - self.posField[other]
                            dist = dx.norm()
                            min_dist = 2 * self.CELL_RADIUS
                            if min_dist > dist > 1e-5:
                                movementOffset = self.CELL_RADIUS * self.CELL_REPULSION * ((min_dist - dist) / min_dist) * dx.normalized()
                                self.posField[i] += movementOffset
                                self.posField[other] -= movementOffset
                            if 1.99*self.CELL_RADIUS > dist > 1e-5:
                                self.inhibitionField[i] += self.INHIBITION_FACTOR
                                if self.inhibitionField[i] > self.INHIBITION_THRESHOLD:
                                    self.inhibitionField[i] = self.INHIBITION_THRESHOLD
                                has_neighbors = True

            # Only reduce inhibition once per cell per step if no neighbors found
            if not has_neighbors:
                self.inhibitionField[i] -= self.INHIBITION_FACTOR

    @ti.kernel
    def handle_cell_cycle(self):
        # Use per-cell cycle duration
        for i in range(self.cellsAlive[None]):
            cycle_length = self.cycleDurField[i]
            g1_end = max(1, int(0.4 * cycle_length))
            s_end = max(g1_end + 1, g1_end + max(1, int(0.33 * cycle_length)))
            g2_end = max(s_end + 1, s_end + max(1, int(0.17 * cycle_length)))
            if g2_end > cycle_length:
                g2_end = cycle_length
            early_g1_end = max(2, g1_end // 20)

            # Phase Switching
            cycleTime = self.step[None] - self.lastDivField[i]
            prev_phase = self.phaseField[i]
            if prev_phase == 0:  # If in G0, stay in G0 until contact inhibition is relieved
                if self.inhibitionField[i] < self.INHIBITION_EXIT_THRESHOLD:
                    # Leaving G0, reset cycle and enter G1
                    self.phaseField[i] = 1
                    self.lastDivField[i] = self.step[None]
                    cycleTime = 0
                else:
                    self.phaseField[i] = 0  # Stay in G0
            else:
                # Only allow entry to G0 during early G1
                if cycleTime < early_g1_end and self.inhibitionField[i] >= self.INHIBITION_THRESHOLD:
                    self.phaseField[i] = 0  # Enter G0
                elif cycleTime < g1_end:
                    self.phaseField[i] = 1  # G1
                elif cycleTime < s_end:
                    self.phaseField[i] = 2  # S
                elif cycleTime < g2_end:
                    self.phaseField[i] = 3  # G2
                else:
                    self.phaseField[i] = 4  # M

            # Cell Movement
            if self.phaseField[i] == 3 or self.phaseField[i] == 0:
                self.mvmtField[i][2] -= self.MAX_CELL_SPEED/40
                if self.mvmtField[i][2] < 0:
                    self.mvmtField[i][2] = 0

            if self.phaseField[i] == 1:
                self.mvmtField[i][2] += self.MAX_CELL_SPEED/10
                if self.mvmtField[i][2] > self.MAX_CELL_SPEED:
                    self.mvmtField[i][2] = self.MAX_CELL_SPEED
                
            # ECM PERIOD CALCULATIONS
            ecm_nearby_count = 0
            pos_i = self.posField[i]
            gridcell_x = ti.min(ti.max(int(pos_i[0] * self.GRID_RES), 0), self.GRID_RES - 1)
            gridcell_y = ti.min(ti.max(int(pos_i[1] * self.GRID_RES), 0), self.GRID_RES - 1)
            for offset in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2)))):
                cx = gridcell_x + offset[0]
                cy = gridcell_y + offset[1]
                if 0 <= cx < self.GRID_RES and 0 <= cy < self.GRID_RES:
                    count = self.ecmGridCount[cx, cy]
                    for j in range(count):
                        ecm_idx = self.ecmGrid[cx, cy, j]
                        dx = pos_i - self.ecmPosField[ecm_idx]
                        dist = dx.norm()
                        if dist < self.ECM_DETECTION_RADIUS:
                            ecm_nearby_count += 1
            self.ecmPeriodField[i] = self.MIN_ECM_PERIOD+ecm_nearby_count
            if ecm_nearby_count > self.ECM_THRESHOLD:
                self.ecmPeriodField[i] = 99999999

            # ECM Deposition
            ecmTime = self.step[None] - self.lastECMField[i]
            if ecmTime >= self.ecmPeriodField[i]:
                current = ti.atomic_add(self.ecmCount[None], 0)
                if current + 1 < self.MAX_ECM_COUNT:  # Safe Addition
                    ecm_idx = ti.atomic_add(self.ecmCount[None], 1)
                    if ecm_idx < self.MAX_ECM_COUNT:
                        ecm_pos = self.posField[i]
                        self.initialize_ecm_fields(ecm_idx, ecm_pos)
                        self.lastECMField[i] = self.step[None]

            # Cell Division
            if self.phaseField[i] == 4 and cycleTime >= cycle_length:
                current = ti.atomic_add(self.cellsAlive[None], 0)
                if current + 1 >= self.MAX_CELL_COUNT:  # Safe Addition
                    continue
                new_idx = ti.atomic_add(self.cellsAlive[None], 1)
                if new_idx < self.MAX_CELL_COUNT:
                    offset_range = self.REPRODUCTION_OFFSET * self.CELL_RADIUS
                    offset = ti.Vector([
                        ti.random() * offset_range - offset_range * 0.5,
                        ti.random() * offset_range - offset_range * 0.5
                    ])
                    new_pos = self.posField[i] + offset
                    self.initialize_cell_fields(new_idx, new_pos)
                    self.lastDivField[i] = self.step[None]

            # Update Genes
            self.update_genes(i)

    @ti.kernel
    def clear_topo_field(self):
        for i, j in self.topoField:
            self.topoField[i, j] = 0.0

    @ti.kernel
    def accumulate_density(self):
        for i in range(self.cellsAlive[None]):
            pos = self.posField[i]
            x = ti.min(ti.max(int(pos[0] * self.GRID_RES), 0), self.GRID_RES - 1)
            y = ti.min(ti.max(int(pos[1] * self.GRID_RES), 0), self.GRID_RES - 1)
            ti.atomic_add(self.topoField[x, y], 1.0)

    @ti.kernel
    def create_cell(self, posX: ti.f32, posY: ti.f32):
        current = ti.atomic_add(self.cellsAlive[None], 0)
        if current + 1 < self.MAX_CELL_COUNT:  # Safe Addition
            new_idx = ti.atomic_add(self.cellsAlive[None], 1)
            if new_idx < self.MAX_CELL_COUNT:
                new_pos = [posX, posY]
                self.initialize_cell_fields(new_idx, new_pos)

    @ti.func
    def initialize_cell_fields(self, idx: ti.i32, pos: ti.template()):
        self.posField[idx] = pos
        self.prevPosField[idx] = pos
        self.lastDivField[idx] = self.step[None]
        self.inhibitionField[idx] = 0
        self.phaseField[idx] = 1
        self.mvmtField[idx] = [ti.random(), 0, self.MAX_CELL_SPEED]
        self.cycleDurField[idx] = self.CELL_CYCLE_DURATION[None] + int((ti.random() - 0.5) * 10)
        self.lastECMField[idx] = self.step[None]
        self.ecmPeriodField[idx] = 0
        self.calc_starting_genes(idx)

    @ti.func
    def initialize_ecm_fields(self, idx: ti.i32, pos: ti.template()):
        self.ecmPosField[idx] = pos

    @ti.kernel
    def clear_grid(self): # Clear the grid
        for i in ti.grouped(self.gridCount):
            self.gridCount[i] = 0

    @ti.kernel
    def insert_into_grid(self): # Place the cell into the correct gridcell
        for i in range(self.cellsAlive[None]):
            pos = self.posField[i]
            cell_x = ti.min(ti.max(int(pos[0] * self.GRID_RES), 0), self.GRID_RES - 1)
            cell_y = ti.min(ti.max(int(pos[1] * self.GRID_RES), 0), self.GRID_RES - 1)

            index = ti.atomic_add(self.gridCount[cell_x, cell_y], 1)
            if index < self.MAX_PARTICLES_PER_GRID_CELL:
                self.grid[cell_x, cell_y, index] = i

    @ti.kernel
    def clamp_cell_count(self):
        if self.cellsAlive[None] > self.MAX_CELL_COUNT:
            self.cellsAlive[None] = self.MAX_CELL_COUNT

    @ti.kernel
    def mark_for_deletion(self, mouse_x: ti.f32, mouse_y: ti.f32, scalpel_radius: ti.f32): # Mark cells for deletion
        for i in range(self.cellsAlive[None]):
            dx = self.posField[i][0] - mouse_x
            dy = self.posField[i][1] - mouse_y
            dist = (dx * dx + dy * dy) ** 0.5
            if dist < scalpel_radius:
                self.toDelete[i] = 1
            else:
                self.toDelete[i] = 0

    @ti.kernel
    def mark_ecm_for_deletion(self, mouse_x: ti.f32, mouse_y: ti.f32, scalpel_radius: ti.f32):
        for i in range(self.ecmCount[None]):
            dx = self.ecmPosField[i][0] - mouse_x
            dy = self.ecmPosField[i][1] - mouse_y
            dist = (dx * dx + dy * dy) ** 0.5
            if dist < scalpel_radius:
                self.toDeleteECM[i] = 1
            else:
                self.toDeleteECM[i] = 0

    @ti.kernel
    def write_buffer_cells(self): # Write to-be-swapped cells to buffer
        self.bufferCount[None] = 0
        for i in range(self.cellsAlive[None]):
            if self.toDelete[i] == 0:
                idx = ti.atomic_add(self.bufferCount[None], 1)
                self.posFieldBuffer[idx] = self.posField[i]
                self.prevPosFieldBuffer[idx] = self.prevPosField[i]
                self.lastDivFieldBuffer[idx] = self.lastDivField[i]
                self.inhibitionFieldBuffer[idx] = self.inhibitionField[i]
                self.phaseFieldBuffer[idx] = self.phaseField[i]
                self.mvmtFieldBuffer[idx] = self.mvmtField[i]
                self.cycleDurFieldBuffer[idx] = self.cycleDurField[i]
                self.geneFieldBuffer[idx] = self.geneField[i]
                self.lastECMFieldBuffer[idx] = self.lastECMField[i]
                self.ecmPeriodFieldBuffer[idx] = self.ecmPeriodField[i]

    @ti.kernel
    def write_buffer_ecm(self):
        self.bufferCountECM[None] = 0
        for i in range(self.ecmCount[None]):
            if self.toDeleteECM[i] == 0:
                idx = ti.atomic_add(self.bufferCountECM[None], 1)
                self.ecmPosFieldBuffer[idx] = self.ecmPosField[i]

    @ti.kernel
    def copy_back_buffer(self): # Write the buffer cells back to main fields
        n = self.bufferCount[None]
        for i in range(n):
            self.posField[i] = self.posFieldBuffer[i]
            self.prevPosField[i] = self.prevPosFieldBuffer[i]
            self.lastDivField[i] = self.lastDivFieldBuffer[i]
            self.inhibitionField[i] = self.inhibitionFieldBuffer[i]
            self.phaseField[i] = self.phaseFieldBuffer[i]
            self.mvmtField[i] = self.mvmtFieldBuffer[i]
            self.cycleDurField[i] = self.cycleDurFieldBuffer[i]
            self.geneField[i] = self.geneFieldBuffer[i]
            self.lastECMField[i] = self.lastECMFieldBuffer[i]
            self.ecmPeriodField[i] = self.ecmPeriodFieldBuffer[i]
        self.cellsAlive[None] = n

    @ti.kernel
    def copy_back_buffer_ecm(self):
        n = self.bufferCountECM[None]
        for i in range(n):
            self.ecmPosField[i] = self.ecmPosFieldBuffer[i]
        self.ecmCount[None] = n

    @ti.kernel
    def clear_ecm_grid(self):
        for i in ti.grouped(self.ecmGridCount):
            self.ecmGridCount[i] = 0

    @ti.kernel
    def insert_ecm_into_grid(self):
        for i in range(self.ecmCount[None]):
            pos = self.ecmPosField[i]
            cell_x = ti.min(ti.max(int(pos[0] * self.GRID_RES), 0), self.GRID_RES - 1)
            cell_y = ti.min(ti.max(int(pos[1] * self.GRID_RES), 0), self.GRID_RES - 1)
            index = ti.atomic_add(self.ecmGridCount[cell_x, cell_y], 1)
            if index < self.MAX_PARTICLES_PER_GRID_CELL:
                self.ecmGrid[cell_x, cell_y, index] = i
