import taichi as ti

from particle.particle import ParticleHandler


@ti.data_oriented
class MovingParticleHandler(ParticleHandler):
    parent = ParticleHandler

    def __init__(self, env, maxCount):
        super().__init__(env, maxCount)

        self.prevPosField = ti.Vector.field(2, dtype=ti.f32, shape=self.MAX_COUNT) # Previous Pos

        self.prevPosFieldBuffer = ti.Vector.field(2, dtype=ti.f32, shape=self.MAX_COUNT)

    @ti.func
    def verlet_step(self): # Verlet Calculations for Motion (velocity calc)
        for i in range(self.count[None]):
            pos = self.posField[i]
            prev = self.prevPosField[i]

            new_pos = pos + (pos - prev) * self.env.FRICTION
            self.prevPosField[i] = pos
            self.posField[i] = new_pos

    @ti.func
    def border_constraints(self):
        for i in range(self.count[None]):
            for j in ti.static(range(2)):
                if self.posField[i][j] < 0:
                    self.posField[i][j] = 0
                    v = self.posField[i][j] - self.prevPosField[i][j]
                    self.prevPosField[i][j] = self.posField[i][j] + v * 0.5

                if self.posField[i][j] > 1:
                    self.posField[i][j] = 1
                    v = self.posField[i][j] - self.prevPosField[i][j]
                    self.prevPosField[i][j] = self.posField[i][j] + v * 0.5

    @ti.func
    def handle_collisions(self): # Collisions
        for i in range(self.count[None]):
            pos_i = self.posField[i]
            gridcell_x = ti.min(ti.max(int(pos_i[0] * self.env.GRID_RES), 0), self.env.GRID_RES - 1)
            gridcell_y = ti.min(ti.max(int(pos_i[1] * self.env.GRID_RES), 0), self.env.GRID_RES - 1)

            for offset in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2)))):
                cx = gridcell_x + offset[0]
                cy = gridcell_y + offset[1]
                if 0 <= cx < self.env.GRID_RES and 0 <= cy < self.env.GRID_RES:
                    count = self.gridCount[cx, cy]
                    for j in range(count):
                        other = self.grid[cx, cy, j]
                        if other != i:
                            dx = self.posField[i] - self.posField[other]
                            dist = dx.norm()
                            min_dist = 2 * self.env.CELL_RADIUS
                            if min_dist > dist > self.env.EPSILON:
                                movementOffset = self.env.CELL_RADIUS * self.env.CELL_REPULSION * ((min_dist - dist) / min_dist) * dx.normalized()
                                self.posField[i] += movementOffset
                                self.posField[other] -= movementOffset
                            self.collide(i, other, dist)

    @ti.func
    def collide(self, i, other, dist):
        pass

    @ti.func
    def clear_field_index(self, index):
        MovingParticleHandler.parent.clear_field_index(self, index)
        self.prevPosField[index] = self.posField[index]

    @ti.func
    def initialize(self, idx: ti.i32, pos: ti.template()):
        MovingParticleHandler.parent.initialize(self, idx, pos)
        self.prevPosField[idx] = pos

    @ti.func
    def write_buffer_index(self, buffer_i, i):
        MovingParticleHandler.parent.write_buffer_index(self, buffer_i, i)
        self.prevPosFieldBuffer[buffer_i] = self.prevPosField[i]

    @ti.func
    def copy_back_buffer_index(self, i):
        MovingParticleHandler.parent.copy_back_buffer_index(self, i)
        self.prevPosField[i] = self.prevPosFieldBuffer[i]