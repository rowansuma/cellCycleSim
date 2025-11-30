import taichi as ti

@ti.data_oriented
class ParticleHandler:
    def __init__(self, env):
        self.env = env

    @ti.func
    def verlet_step(self, handler): # Verlet Calculations for Motion (velocity calc)
        for index in range(handler.count[None]):
            tempVar = handler.posField[index]
            handler.posField[index] += (handler.posField[index] - handler.prevPosField[index])*self.env.FRICTION
            handler.prevPosField[index] = tempVar

    @ti.func
    def border_constraints(self, handler):
        for i in range(handler.count[None]):
            for j in ti.static(range(2)):
                if handler.posField[i][j] < 0:
                    handler.posField[i][j] = 0
                    v = handler.posField[i][j] - handler.prevPosField[i][j]
                    handler.prevPosField[i][j] = handler.posField[i][j] + v * 0.5

                if handler.posField[i][j] > 1:
                    handler.posField[i][j] = 1
                    v = handler.posField[i][j] - handler.prevPosField[i][j]
                    handler.prevPosField[i][j] = handler.posField[i][j] + v * 0.5

    @ti.func
    def handle_collisions(self, handler): # Collisions
        for i in range(handler.count[None]):
            pos_i = handler.posField[i]
            gridcell_x = ti.min(ti.max(int(pos_i[0] * self.env.GRID_RES), 0), self.env.GRID_RES - 1)
            gridcell_y = ti.min(ti.max(int(pos_i[1] * self.env.GRID_RES), 0), self.env.GRID_RES - 1)

            has_neighbors = False
            for offset in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2)))):
                cx = gridcell_x + offset[0]
                cy = gridcell_y + offset[1]
                if 0 <= cx < self.env.GRID_RES and 0 <= cy < self.env.GRID_RES:
                    count = handler.gridCount[cx, cy]
                    for j in range(count):
                        other = handler.grid[cx, cy, j]
                        if other != i:
                            dx = handler.posField[i] - handler.posField[other]
                            dist = dx.norm()
                            min_dist = 2 * self.env.CELL_RADIUS
                            if min_dist > dist > 1e-5:
                                movementOffset = self.env.CELL_RADIUS * self.env.CELL_REPULSION * ((min_dist - dist) / min_dist) * dx.normalized()
                                handler.posField[i] += movementOffset
                                handler.posField[other] -= movementOffset
                            # TODO: Refactor out of general particle code
                            if 1.99*self.env.CELL_RADIUS > dist > 1e-5:
                                self.env.cellHandler.inhibitionField[i] += self.env.INHIBITION_FACTOR
                                if self.env.cellHandler.inhibitionField[i] > self.env.INHIBITION_THRESHOLD:
                                    self.env.cellHandler.inhibitionField[i] = self.env.INHIBITION_THRESHOLD
                                has_neighbors = True

            # Only reduce inhibition once per cell per step if no neighbors found
            if not has_neighbors:
                self.env.cellHandler.inhibitionField[i] -= self.env.INHIBITION_FACTOR

    @ti.func
    def insert_into_grid(self, handler):
        for i in range(handler.count[None]):
            pos = handler.posField[i]
            cell_x = ti.min(ti.max(int(pos[0] * self.env.GRID_RES), 0), self.env.GRID_RES - 1)
            cell_y = ti.min(ti.max(int(pos[1] * self.env.GRID_RES), 0), self.env.GRID_RES - 1)

            index = ti.atomic_add(handler.gridCount[cell_x, cell_y], 1)
            if index < self.env.MAX_PARTICLES_PER_GRID_CELL:
                handler.grid[cell_x, cell_y, index] = i

    @ti.func
    def mark_for_deletion(self, handler, mouse_x: ti.f32, mouse_y: ti.f32, size: ti.f32, shape: ti.i32): # 0 = circle, 1 = square, 2 = triangle
        for i in range(handler.count[None]):
            dx = handler.posField[i][0] - mouse_x
            dy = handler.posField[i][1] - mouse_y
            delete = 0

            if shape == 0:  # Circle
                dist = ti.math.length(ti.Vector([dx, dy]))
                if dist < size:
                    delete = 1

            elif shape == 1:  # Square
                if ti.abs(dx) < size and ti.abs(dy) < size:
                    delete = 1

            elif shape == 2:  # Upright equilateral triangle
                height = size*2
                # shift so triangle base is at -height/2, tip is at +height/2
                local_y = dy + height / 2
                if 0 <= local_y <= height:
                    # proportion from base (0) to tip (height)
                    t = local_y / height
                    half_width = (1 - t) * size
                    if ti.abs(dx) <= half_width:
                        delete = 1

            handler.toDelete[i] = delete

    @ti.func
    def write_buffer(self, handler):
        handler.bufferCount[None] = 0
        for i in range(handler.count[None]):
            if handler.toDelete[i] == 0:
                idx = ti.atomic_add(handler.bufferCount[None], 1)

                handler.write_buffer_index(idx, i)

    @ti.func
    def copy_back_buffer(self, handler):
        n = handler.bufferCount[None]
        for i in range(n):
            handler.copy_back_buffer_index(i)
        handler.count[None] = n
