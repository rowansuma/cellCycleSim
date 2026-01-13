import taichi as ti

@ti.data_oriented
class ParticleHandler:
    def __init__(self, env, maxCount):
        self.env = env
        self.MAX_COUNT = maxCount

        self.posField = ti.Vector.field(2, dtype=ti.f32, shape=self.MAX_COUNT) # Current Pos

        self.posFieldBuffer = ti.Vector.field(2, dtype=ti.f32, shape=self.MAX_COUNT)

        # ECM Grid
        self.grid = ti.field(dtype=ti.i32, shape=(self.env.GRID_RES, self.env.GRID_RES, self.env.MAX_PARTICLES_PER_GRID_CELL))
        self.gridCount = ti.field(dtype=ti.i32, shape=(self.env.GRID_RES, self.env.GRID_RES))

        self.toDelete = ti.field(dtype=ti.i32, shape=self.MAX_COUNT)
        self.bufferCount = ti.field(dtype=ti.i32, shape=())

        self.tmpWriteCount = ti.field(dtype=ti.i32, shape=())

        self.count = ti.field(dtype=ti.i32, shape=())

    @ti.func
    def rebuild_grid(self):
        # clear grid
        for i, j in self.gridCount:
            self.gridCount[i, j] = 0

        # insert particles
        for i in range(self.count[None]):
            pos = self.posField[i]
            cell_x = ti.min(ti.max(int(pos[0] * self.env.GRID_RES), 0), self.env.GRID_RES - 1)
            cell_y = ti.min(ti.max(int(pos[1] * self.env.GRID_RES), 0), self.env.GRID_RES - 1)

            index = ti.atomic_add(self.gridCount[cell_x, cell_y], 1)
            if index < self.env.MAX_PARTICLES_PER_GRID_CELL:
                self.grid[cell_x, cell_y, index] = i

    @ti.func
    def mark_for_deletion(self, mouse_x: ti.f32, mouse_y: ti.f32, width: ti.f32, shape: ti.i32):   # 0 = circle, 1 = square, 2 = triangle, 3 = triangle
        width = width/self.env.DOMAIN_SIZE
        for i in range(self.count[None]):
            dx = self.posField[i][0] - mouse_x
            dy = self.posField[i][1] - mouse_y
            delete = 0

            if shape == 0:  # Circle
                dist = ti.math.length(ti.Vector([dx, dy]))
                if dist < width/2:
                    delete = 1

            elif shape == 1:  # Square
                if ti.abs(dx) < width/2 and ti.abs(dy) < width/2:
                    delete = 1

            elif shape == 2:  # Upright equilateral triangle
                height = width
                # shift so triangle base is at -height/2, tip is at +height/2
                local_y = dy + height / 2
                if 0 <= local_y <= height:
                    # proportion from base (0) to tip (height)
                    t = local_y / height
                    half_width = (1 - t) * width/2
                    if ti.abs(dx) <= half_width:
                        delete = 1

            elif shape == 3:
                if ti.abs(dx) < width/2:
                    delete = 1


            self.toDelete[i] = delete

    @ti.func
    def create(self, posX: ti.f32, posY: ti.f32):
        success = False
        if 0 < posX < 1 and 0 < posY < 1:
            current = self.count[None]
            if current + 1 < self.MAX_COUNT:  # Safe Addition
                new_idx = ti.atomic_add(self.count[None], 1)
                if new_idx < self.MAX_COUNT:
                    new_pos = [posX, posY]
                    self.initialize(new_idx, new_pos)
                    success = True
        return success

    @ti.func
    def update(self):
        self.clamp_count()

    @ti.func
    def write_buffer(self):
        self.tmpWriteCount[None] = 0
        for i in range(self.count[None]):
            if self.toDelete[i] == 0:
                idx = ti.atomic_add(self.tmpWriteCount[None], 1)
                self.write_buffer_index(idx, i)
        self.bufferCount[None] = self.tmpWriteCount[None]

    @ti.func
    def copy_back_buffer(self):
        n = self.bufferCount[None]
        for i in range(n):
            self.copy_back_buffer_index(i)
        for i in range(n, self.MAX_COUNT):
            self.clear_field_index(i)
        self.count[None] = n

    @ti.func
    def clear_fields(self):
        self.count[None] = 0
        for index in range(self.MAX_COUNT):
            self.clear_field_index(index)

    @ti.func
    def initialize(self, idx: ti.i32, pos: ti.template()):
        self.posField[idx] = pos

    @ti.func
    def write_buffer_index(self, buffer_i, i):
        self.posFieldBuffer[buffer_i] = self.posField[i]

    @ti.func
    def copy_back_buffer_index(self, i):
        self.posField[i] = self.posFieldBuffer[i]

    @ti.func
    def clear_field_index(self, index):
        self.posField[index] = [-1, -1]

    @ti.func
    def clamp_count(self):
        if self.count[None] > self.MAX_COUNT:
            self.count[None] = self.MAX_COUNT

    def export_state(self):
        return {
            "count": self.count.to_numpy(),

            "posField": self.posField.to_numpy(),

            "posFieldBuffer": self.posFieldBuffer.to_numpy(),
        }

    def load_state(self, data):
        self.count.from_numpy(data["count"])

        self.posField.from_numpy(data["posField"])

        self.posFieldBuffer.from_numpy(data["posFieldBuffer"])