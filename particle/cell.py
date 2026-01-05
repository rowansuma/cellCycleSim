import taichi as ti

from particle.moving_particle import MovingParticleHandler

@ti.data_oriented
class CellHandler(MovingParticleHandler):
    parent = MovingParticleHandler

    def __init__(self, env, maxCount):
        super().__init__(env, maxCount)

        # Fields
        self.lastDivField = ti.field(dtype=ti.i32, shape=self.MAX_COUNT)
        self.inhibitionField = ti.field(dtype=ti.f32, shape=self.MAX_COUNT)
        self.neighborField = ti.field(dtype=ti.f32, shape=self.MAX_COUNT)
        self.phaseField = ti.field(dtype=ti.i32, shape=self.MAX_COUNT)
        self.mvmtField = ti.Vector.field(3, dtype=ti.f32, shape=self.MAX_COUNT)
        self.cycleDurField = ti.field(dtype=ti.i32, shape=self.MAX_COUNT)  # Cycle duration

        # Buffer Fields
        self.lastDivFieldBuffer = ti.field(dtype=ti.i32, shape=self.MAX_COUNT)
        self.inhibitionFieldBuffer = ti.field(dtype=ti.f32, shape=self.MAX_COUNT)
        self.neighborFieldBuffer = ti.field(dtype=ti.f32, shape=self.MAX_COUNT)
        self.phaseFieldBuffer = ti.field(dtype=ti.i32, shape=self.MAX_COUNT)
        self.mvmtFieldBuffer = ti.Vector.field(3, dtype=ti.f32, shape=self.MAX_COUNT)
        self.cycleDurFieldBuffer = ti.field(dtype=ti.i32, shape=self.MAX_COUNT)

    @ti.func
    def apply_locomotion(self, i: ti.i32):
        for _ in range(self.env.SUBSTEPS):
            repulse_vec = ti.Vector([0.0, 0.0])
            ecm_count = 0
            if self.phaseField[i] != 0:
                gridcell_x = ti.min(ti.max(int(self.posField[i][0] * self.env.GRID_RES), 0), self.env.GRID_RES - 1)
                gridcell_y = ti.min(ti.max(int(self.posField[i][1] * self.env.GRID_RES), 0), self.env.GRID_RES - 1)
                ecm_centroid = ti.Vector([0.0, 0.0])
                for offset in ti.static(ti.grouped(ti.ndrange((-2, 3), (-2, 3)))):
                    cx = gridcell_x + offset[0]
                    cy = gridcell_y + offset[1]
                    if 0 <= cx < self.env.GRID_RES and 0 <= cy < self.env.GRID_RES:
                        count = self.env.ecmHandler.gridCount[cx, cy]
                        for j in range(count):
                            ecm_idx = self.env.ecmHandler.grid[cx, cy, j]
                            dx = self.posField[i] - self.env.ecmHandler.posField[ecm_idx]
                            dist = dx.norm()
                            if dist < self.env.ECM_DETECTION_RADIUS:
                                ecm_centroid += self.env.ecmHandler.posField[ecm_idx]
                                ecm_count += 1
                if ecm_count > 0:
                    ecm_avg_pos = ecm_centroid/ecm_count
                    delta = self.posField[i] - ecm_avg_pos
                    if ti.math.length(delta) > 0.005:
                        repulse_vec = ti.math.normalize(delta)*self.env.ECM_AVOIDANCE_STRENGTH

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

    @ti.func
    def update(self):
        CellHandler.parent.update(self)
        for i in range(self.count[None]):
            self.apply_locomotion(i)
            self.handleCellDependentBehavior(i)
            if self.neighborField[i] == 0:
                self.inhibitionField[i] -= self.env.INHIBITION_FACTOR
            self.neighborField[i] = 0

    @ti.func
    def handleCellDependentBehavior(self, i: ti.i32):
        self.handle_cell_cycle(i)

    @ti.func
    def collide(self, i, other, dist):
        CellHandler.parent.collide(self, i, other, dist)
        if self.env.INHIBITION_RADIUS*self.env.CELL_RADIUS > dist > self.env.EPSILON:
            self.inhibitionField[i] += self.env.INHIBITION_FACTOR
            if self.inhibitionField[i] > self.env.INHIBITION_THRESHOLD:
                self.inhibitionField[i] = self.env.INHIBITION_THRESHOLD
            self.neighborField[i] = 1

    @ti.func
    def handle_cell_cycle(self, i: ti.i32):
        # Use per-cell cycle duration
        cycle_length = self.cycleDurField[i]
        g1_end = max(1, int(0.4 * cycle_length))
        s_end = max(g1_end + 1, g1_end + max(1, int(0.33 * cycle_length)))
        g2_end = max(s_end + 1, s_end + max(1, int(0.17 * cycle_length)))
        if g2_end > cycle_length:
            g2_end = cycle_length
        early_g1_end = max(2, g1_end // 20)

        # Phase Switching
        cycleTime = self.env.step[None] - self.lastDivField[i]
        prev_phase = self.phaseField[i]
        if prev_phase == 0:  # If in G0, stay in G0 until contact inhibition is relieved
            if self.inhibitionField[i] < self.env.INHIBITION_EXIT_THRESHOLD:
                # Leaving G0, reset cycle and enter G1
                self.phaseField[i] = 1
                self.lastDivField[i] = self.env.step[None]
                cycleTime = 0
            else:
                self.phaseField[i] = 0  # Stay in G0
        else:
            # Only allow entry to G0 during early G1
            if cycleTime < early_g1_end and self.inhibitionField[i] >= self.env.INHIBITION_THRESHOLD:
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
            self.mvmtField[i][2] -= self.env.MAX_CELL_SPEED/40
            if self.mvmtField[i][2] < 0:
                self.mvmtField[i][2] = 0

        if self.phaseField[i] == 1:
            self.mvmtField[i][2] += self.env.MAX_CELL_SPEED/10
            if self.mvmtField[i][2] > self.env.MAX_CELL_SPEED:
                self.mvmtField[i][2] = self.env.MAX_CELL_SPEED

        # Cell Division
        if self.phaseField[i] == 4 and cycleTime >= cycle_length:
            offset_range = self.env.REPRODUCTION_OFFSET * self.env.CELL_RADIUS
            offset = ti.Vector([
                ti.random() * offset_range - offset_range * 0.5,
                ti.random() * offset_range - offset_range * 0.5])
            new_pos = self.posField[i] + offset
            self.create(new_pos[0], new_pos[1])
            self.lastDivField[i] = self.env.step[None]

    @ti.func
    def clear_field_index(self, index):
        CellHandler.parent.clear_field_index(self, index)
        self.lastDivField[index] = -1
        self.inhibitionField[index] = -1
        self.neighborField[index] = -1
        self.phaseField[index] = -1
        self.mvmtField[index] = [-1, -1, -1]
        self.cycleDurField[index] = -1

    @ti.func
    def initialize(self, idx: ti.i32, pos: ti.template()):
        CellHandler.parent.initialize(self, idx, pos)
        self.lastDivField[idx] = self.env.step[None]
        self.inhibitionField[idx] = 0
        self.neighborField[idx] = 0
        self.phaseField[idx] = 1
        self.mvmtField[idx] = [ti.random(), 0, self.env.MAX_CELL_SPEED]
        self.cycleDurField[idx] = self.env.CELL_CYCLE_DURATION[None] + int((ti.random() - 0.5) * 10)

    @ti.func
    def write_buffer_index(self, buffer_i, i):
        CellHandler.parent.write_buffer_index(self, buffer_i, i)
        self.lastDivFieldBuffer[buffer_i] = self.lastDivField[i]
        self.inhibitionFieldBuffer[buffer_i] = self.inhibitionField[i]
        self.neighborFieldBuffer[buffer_i] = self.neighborField[i]
        self.phaseFieldBuffer[buffer_i] = self.phaseField[i]
        self.mvmtFieldBuffer[buffer_i] = self.mvmtField[i]
        self.cycleDurFieldBuffer[buffer_i] = self.cycleDurField[i]

    @ti.func
    def copy_back_buffer_index(self, i):
        CellHandler.parent.copy_back_buffer_index(self, i)
        self.lastDivField[i] = self.lastDivFieldBuffer[i]
        self.inhibitionField[i] = self.inhibitionFieldBuffer[i]
        self.neighborField[i] = self.neighborFieldBuffer[i]
        self.phaseField[i] = self.phaseFieldBuffer[i]
        self.mvmtField[i] = self.mvmtFieldBuffer[i]
        self.cycleDurField[i] = self.cycleDurFieldBuffer[i]

    def export_state(self):
        return CellHandler.parent.export_state(self) | {
            "lastDivField": self.lastDivField.to_numpy(),
            "inhibitionField": self.inhibitionField.to_numpy(),
            "neighborField": self.neighborField.to_numpy(),
            "phaseField": self.phaseField.to_numpy(),
            "mvmtField": self.mvmtField.to_numpy(),
            "cycleDurField": self.cycleDurField.to_numpy(),

            "lastDivFieldBuffer": self.lastDivFieldBuffer.to_numpy(),
            "inhibitionFieldBuffer": self.inhibitionFieldBuffer.to_numpy(),
            "neighborFieldBuffer": self.neighborFieldBuffer.to_numpy(),
            "phaseFieldBuffer": self.phaseFieldBuffer.to_numpy(),
            "mvmtFieldBuffer": self.mvmtFieldBuffer.to_numpy(),
            "cycleDurFieldBuffer": self.cycleDurFieldBuffer.to_numpy(),
        }

    def load_state(self, data):
        CellHandler.parent.load_state(self, data)

        self.lastDivField.from_numpy(data["lastDivField"])
        self.inhibitionField.from_numpy(data["inhibitionField"])
        self.neighborField.from_numpy(data["neighborField"])
        self.phaseField.from_numpy(data["phaseField"])
        self.mvmtField.from_numpy(data["mvmtField"])
        self.cycleDurField.from_numpy(data["cycleDurField"])

        self.lastDivFieldBuffer.from_numpy(data["lastDivFieldBuffer"])
        self.inhibitionFieldBuffer.from_numpy(data["inhibitionFieldBuffer"])
        self.neighborFieldBuffer.from_numpy(data["neighborFieldBuffer"])
        self.phaseFieldBuffer.from_numpy(data["phaseFieldBuffer"])
        self.mvmtFieldBuffer.from_numpy(data["mvmtFieldBuffer"])
        self.cycleDurFieldBuffer.from_numpy(data["cycleDurFieldBuffer"])
