import taichi as ti

@ti.data_oriented
class ECMHandler:
    def __init__(self, env):
        self.env = env

        self.MAX_COUNT = env.MAX_ECM_COUNT

        # Fields
        self.posField = ti.Vector.field(2, dtype=ti.f32, shape=self.MAX_COUNT) # Current Pos

        # Buffer Fields
        self.posFieldBuffer = ti.Vector.field(2, dtype=ti.f32, shape=self.MAX_COUNT)

        # ECM Grid
        self.grid = ti.field(dtype=ti.i32, shape=(self.env.GRID_RES, self.env.GRID_RES, self.env.MAX_PARTICLES_PER_GRID_CELL))
        self.gridCount = ti.field(dtype=ti.i32, shape=(self.env.GRID_RES, self.env.GRID_RES))

        self.toDelete = ti.field(dtype=ti.i32, shape=self.MAX_COUNT)
        self.bufferCount = ti.field(dtype=ti.i32, shape=())

        self.count = ti.field(dtype=ti.i32, shape=()) # 0

    @ti.func
    def clear_fields(self):
        self.count[None] = 0
        for index in range(self.MAX_COUNT):
            self.posField[index] = [-1, -1]

    @ti.func
    def initialize(self, idx: ti.i32, pos: ti.template()):
        self.posField[idx] = pos

    @ti.func
    def write_buffer_index(self, buffer_i, i):
        self.posFieldBuffer[buffer_i] = self.posField[i]

    @ti.func
    def copy_back_buffer_index(self, i):
        self.posField[i] = self.posFieldBuffer[i]

    @ti.kernel
    def clear_grid_kernel(self): # Clear the grid
        for i in ti.grouped(self.gridCount):
            self.gridCount[i] = 0

    @ti.kernel
    def insert_into_grid_kernel(self): # Place the cell into the correct gridcell
        self.env.particleHandler.insert_into_grid(self)

    @ti.func
    def create(self, posX: ti.f32, posY: ti.f32):
        success = False
        current = ti.atomic_add(self.count[None], 0)
        if current + 1 < self.MAX_COUNT:  # Safe Addition
            new_idx = ti.atomic_add(self.count[None], 1)
            if new_idx < self.MAX_COUNT:
                new_pos = [posX, posY]
                self.initialize(new_idx, new_pos)
                success = True
        return success