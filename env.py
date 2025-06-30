import taichi as ti

@ti.data_oriented
class Env:
    def __init__(self, radius, max_cells, freq, scalpel_size):
        self.SCREEN_SIZE = (1000, 1000)
        self.SUBSTEPS = 3

        # Constants
        self.MAX_CELL_COUNT = max_cells
        self.CELL_RADIUS = radius
        self.CELL_RADIUS_SCALAR = 1.0 # Increase for better visibility at larger scale
        self.CELL_REPULSION = 0.000004 # how aggressively the cells repulse each other
        self.INHIBITION_DISTANCE = 2.0
        self.INHIBITION_COUNT = 3
        self.REPRODUCTION_OFFSET = 1.5
        self.SCALPEL_RADIUS = scalpel_size

        self.GRID_SCALE_FACTOR = 1.5
        self.GRID_RES = int(1 / (self.CELL_RADIUS * 2 * self.GRID_SCALE_FACTOR))
        self.MAX_PARTICLES_PER_GRID_CELL = 64  # or 64
        self.FRICTION = 0.95

        self.CELL_CYCLE_DURATION = ti.field(dtype=ti.i32, shape=())
        self.CCDPlaceholder = freq

        self.PHASE_COLORS = [
            0x858585,  # G0 (gray)
            0x66ccff,  # G1 (blue)
            0xffcc66,  # S  (yellow)
            0x66ff66,  # G2 (green)
            0xff6699,  # M  (pink)
        ]

        # Taichi counters
        self.step = ti.field(dtype=ti.i32, shape=()) # 0
        self.cellsAlive = ti.field(dtype=ti.i32, shape=()) # 1

        # Cell Info
        self.posField = ti.Vector.field(2, dtype=ti.f32, shape=self.MAX_CELL_COUNT) # Current Pos
        self.prevPosField = ti.Vector.field(2, dtype=ti.f32, shape=self.MAX_CELL_COUNT) # Previous Pos
        self.lastDivField = ti.field(dtype=ti.i32, shape=self.MAX_CELL_COUNT)
        self.neighborsField = ti.field(dtype=ti.i32, shape=self.MAX_CELL_COUNT)
        self.phaseField = ti.field(dtype=ti.i32, shape=self.MAX_CELL_COUNT)

        # Grid
        self.grid = ti.field(dtype=ti.i32, shape=(self.GRID_RES, self.GRID_RES, self.MAX_PARTICLES_PER_GRID_CELL))
        self.gridCount = ti.field(dtype=ti.i32, shape=(self.GRID_RES, self.GRID_RES))  # how many particles per cell

        # Cell Info Buffers
        self.posFieldBuffer = ti.Vector.field(2, dtype=ti.f32, shape=self.MAX_CELL_COUNT)
        self.prevPosFieldBuffer = ti.Vector.field(2, dtype=ti.f32, shape=self.MAX_CELL_COUNT)
        self.lastDivFieldBuffer = ti.field(dtype=ti.i32, shape=self.MAX_CELL_COUNT)
        self.neighborsFieldBuffer = ti.field(dtype=ti.i32, shape=self.MAX_CELL_COUNT)
        self.phaseFieldBuffer = ti.field(dtype=ti.i32, shape=self.MAX_CELL_COUNT)

        # Deletion
        self.toDelete = ti.field(dtype=ti.i32, shape=self.MAX_CELL_COUNT)
        self.bufferCount = ti.field(dtype=ti.i32, shape=())

        self.initialize_board()

    @ti.kernel
    def initialize_board(self): # Board Init, assign taichi fields
        self.cellsAlive[None] = 1
        self.CELL_CYCLE_DURATION[None] = self.CCDPlaceholder
        self.step[None] = 0

        for index in range(self.MAX_CELL_COUNT): # Create Placeholders
            self.posField[index] = [-1, -1]
            self.prevPosField[index] = self.posField[index]
            self.lastDivField[index] = -1
            self.neighborsField[index] = -1
            self.phaseField[index] = -1
        self.posField[0] = [0.5, 0.5] # Init First Cell
        self.prevPosField[0] = self.posField[0]
        self.lastDivField[0] = 0
        self.neighborsField[0] = 0
        self.phaseField[0] = 1


    @ti.kernel
    def verlet_step(self): # Verlet Calculations for Motion (velocity calc)
        for index in range(self.cellsAlive[None]):
            tempVar = self.posField[index]
            self.posField[index] += (self.posField[index] - self.prevPosField[index])*self.FRICTION
            self.prevPosField[index] = tempVar

    @ti.kernel
    def border_constraints(self): # Stop cells from exiting view
        for i in range(self.cellsAlive[None]):
            self.neighborsField[i] = 0
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
            if at_border:
                self.neighborsField[i] += 1


    @ti.kernel
    def handle_collisions(self): # Collisions
        for i in range(self.cellsAlive[None]):
            pos_i = self.posField[i]
            gridcell_x = ti.min(ti.max(int(pos_i[0] * self.GRID_RES), 0), self.GRID_RES - 1)
            gridcell_y = ti.min(ti.max(int(pos_i[1] * self.GRID_RES), 0), self.GRID_RES - 1)

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
                                movementOffset = self.CELL_REPULSION * ((min_dist - dist) / min_dist) * dx.normalized()
                                self.posField[i] += movementOffset
                                self.posField[other] -= movementOffset
                            if dist <= self.INHIBITION_DISTANCE * self.CELL_RADIUS:
                                self.neighborsField[i] += 1


    @ti.kernel
    def handle_cell_cycle(self):
        offset_range = self.REPRODUCTION_OFFSET * self.CELL_RADIUS
        g1_end = self.CELL_CYCLE_DURATION[None] * 10 // 24
        s_end = self.CELL_CYCLE_DURATION[None] * 18 // 24
        g2_end = self.CELL_CYCLE_DURATION[None] * 22 // 24

        for i in range(self.cellsAlive[None]):
            # Cycle Phase Assignment
            cycleTime = self.step[None] - self.lastDivField[i]
            if self.neighborsField[i] >= self.INHIBITION_COUNT:
                self.phaseField[i] = 0  # G0
            elif cycleTime < g1_end:
                self.phaseField[i] = 1  # G1
            elif cycleTime < s_end:
                self.phaseField[i] = 2  # S
            elif cycleTime < g2_end:
                self.phaseField[i] = 3  # G2
            else:
                self.phaseField[i] = 4  # M

            # Cell Division
            if self.phaseField[i] != 0:
                if cycleTime >= self.CELL_CYCLE_DURATION[None]:
                    current = ti.atomic_add(self.cellsAlive[None], 0)
                    if current + 1 >= self.MAX_CELL_COUNT: # Safe Addition
                        continue
                    new_idx = ti.atomic_add(self.cellsAlive[None], 1)
                    if new_idx < self.MAX_CELL_COUNT:
                        offset = ti.Vector([
                            ti.random() * offset_range - offset_range * 0.5,
                            ti.random() * offset_range - offset_range * 0.5
                        ])
                        new_pos = self.posField[i] + offset
                        self.posField[new_idx] = new_pos
                        self.prevPosField[new_idx] = new_pos  # match to avoid initial velocity
                        self.lastDivField[new_idx] = self.step[None]
                        self.lastDivField[i] = self.step[None]
                        self.neighborsField[new_idx] = 0
                        self.phaseField[new_idx] = 1


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
    def write_buffer_cells(self): # Write to-be-swapped cells to buffer
        self.bufferCount[None] = 0
        for i in range(self.cellsAlive[None]):
            if self.toDelete[i] == 0:
                idx = ti.atomic_add(self.bufferCount[None], 1)
                self.posFieldBuffer[idx] = self.posField[i]
                self.prevPosFieldBuffer[idx] = self.prevPosField[i]
                self.lastDivFieldBuffer[idx] = self.lastDivField[i]
                self.neighborsFieldBuffer[idx] = self.neighborsField[i]
                self.phaseFieldBuffer[idx] = self.phaseField[i]

    @ti.kernel
    def copy_back_buffer(self): # Write the buffer cells back to main fields
        n = self.bufferCount[None]
        for i in range(n):
            self.posField[i] = self.posFieldBuffer[i]
            self.prevPosField[i] = self.prevPosFieldBuffer[i]
            self.lastDivField[i] = self.lastDivFieldBuffer[i]
            self.neighborsField[i] = self.neighborsFieldBuffer[i]
            self.phaseField[i] = self.phaseFieldBuffer[i]
        for i in range(n, self.cellsAlive[None]):
            self.posField[i] = [-1, -1]
            self.prevPosField[i] = [-1, -1]
            self.lastDivField[i] = -1
            self.neighborsField[i] = -1
            self.phaseField[i] = -1
        self.cellsAlive[None] = n
