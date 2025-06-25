import taichi as ti
@ti.data_oriented
class Env:
    def __init__(self):
        self.SCREEN_SIZE = (1000, 1000)
        self.SUBSTEPS = 3

        self.CELL_COUNT = 50000
        self.CELL_RADIUS = 0.0015
        self.CELL_RADIUS_SCALAR = 600

        self.CELL_COLOR = (255, 0, 0)

        self.posField = ti.Vector.field(2, dtype=ti.f32, shape=self.CELL_COUNT) # Current Pos
        self.prevPosField = ti.Vector.field(2, dtype=ti.f32, shape=self.CELL_COUNT) # Previous Pos



    @ti.kernel
    def initialize_board(self):
        # Populate Cells
        for index in range(self.CELL_COUNT):
            self.posField[index] = [ti.random() * 0.9 + 0.05, ti.random() * 0.9 + 0.05]
            self.prevPosField[index] = self.posField[index]


    @ti.kernel
    def verlet_step(self):
        for index in range(self.CELL_COUNT):
            tempVar = self.posField[index]
            self.posField[index] += (self.posField[index] - self.prevPosField[index])  # No forces for now
            self.prevPosField[index] = tempVar

    @ti.kernel
    def border_constraints(self):
        for i in range(self.CELL_COUNT):
            for j in ti.static(range(2)):
                if self.posField[i][j] < 0: self.posField[i][j] = 0
                if self.posField[i][j] > 1: self.posField[i][j] = 1

    @ti.kernel
    def solve_collisions(self):
        for i in range(self.CELL_COUNT):
            for j in range(i + 1, self.CELL_COUNT):  # Brute force (optimize later with spatial hashing!)
                dx = self.posField[i] - self.posField[j]
                dist = dx.norm()
                min_dist = 2 * self.CELL_RADIUS
                if min_dist > dist > 1e-5:
                    offset = 0.5 * (min_dist - dist) * dx.normalized()
                    self.posField[i] += offset
                    self.posField[j] -= offset