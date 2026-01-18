import taichi as ti

from particle.cell import CellHandler

@ti.data_oriented
class FibroblastHandler(CellHandler):
    parent = CellHandler

    def __init__(self, env):
        super().__init__(env, env.MAX_CELL_COUNT)

        self.lastECMPosField = ti.Vector.field(2, dtype=ti.f32, shape=self.MAX_COUNT)
        self.lastECMField = ti.field(dtype=ti.i32, shape=self.MAX_COUNT)
        self.ecmPeriodField = ti.field(dtype=ti.f32, shape=self.MAX_COUNT)

        self.lastECMPosFieldBuffer = ti.Vector.field(2, dtype=ti.f32, shape=self.MAX_COUNT)
        self.lastECMFieldBuffer = ti.field(dtype=ti.i32, shape=self.MAX_COUNT)
        self.ecmPeriodFieldBuffer = ti.field(dtype=ti.f32, shape=self.MAX_COUNT)

    @ti.func
    def handleCellDependentBehavior(self, i: ti.i32):
        FibroblastHandler.parent.handleCellDependentBehavior(self, i)
        self.handle_ecm(i)

    @ti.func
    def handle_ecm(self, i: ti.i32):
        # ECM PERIOD CALCULATIONS
        ecm_nearby_count = 0
        pos_i = self.posField[i]
        gridcell_x = ti.min(ti.max(int(pos_i[0] * self.env.GRID_RES), 0), self.env.GRID_RES - 1)
        gridcell_y = ti.min(ti.max(int(pos_i[1] * self.env.GRID_RES), 0), self.env.GRID_RES - 1)
        for offset in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2)))):
            cx = gridcell_x + offset[0]
            cy = gridcell_y + offset[1]
            if 0 <= cx < self.env.GRID_RES and 0 <= cy < self.env.GRID_RES:
                count = self.env.ecmHandler.gridCount[cx, cy]
                for j in range(count):
                    ecm_idx = self.env.ecmHandler.grid[cx, cy, j]
                    dx = pos_i - self.env.ecmHandler.posField[ecm_idx]
                    dist = dx.norm()
                    if dist < self.env.ECM_DETECTION_RADIUS:
                        ecm_nearby_count += 1
        self.ecmPeriodField[i] = self.env.MIN_ECM_PERIOD+ecm_nearby_count
        if ecm_nearby_count > self.env.ECM_THRESHOLD:
            self.ecmPeriodField[i] = 99999999

        # ECM Deposition
        ecmTime = self.env.step[None] - self.lastECMField[i]
        if ecmTime >= self.ecmPeriodField[i]:
            new_ecm_idx = self.env.ecmHandler.create(self.posField[i][0], self.posField[i][1])
            if self.lastECMPosField[i][0] != -1:
                self.env.ecmHandler.ecmConnectPosField[new_ecm_idx] = self.lastECMPosField[i]
            # self.env.ecmHandler.calculateConnectPos(new_ecm_idx, self.lastECMPosField[i])
            if new_ecm_idx != -1:
                self.lastECMField[i] = self.env.step[None]
                self.lastECMPosField[i] = self.posField[i]

    @ti.func
    def clear_field_index(self, index):
        FibroblastHandler.parent.clear_field_index(self, index)
        self.lastECMPosField[index] = [-1, -1]
        self.lastECMField[index] = -1
        self.ecmPeriodField[index] = -1

    @ti.func
    def initialize(self, idx: ti.i32, pos: ti.template()):
        FibroblastHandler.parent.initialize(self, idx, pos)
        self.lastECMPosField[idx] = [-1, -1]
        self.lastECMField[idx] = self.env.step[None]
        self.ecmPeriodField[idx] = 0

    @ti.func
    def write_buffer_index(self, buffer_i, i):
        FibroblastHandler.parent.write_buffer_index(self, buffer_i, i)
        self.lastECMFieldBuffer[buffer_i] = self.lastECMField[i]
        self.lastECMPosFieldBuffer[buffer_i] = self.lastECMPosField[i]
        self.ecmPeriodFieldBuffer[buffer_i] = self.ecmPeriodField[i]

    @ti.func
    def copy_back_buffer_index(self, i):
        FibroblastHandler.parent.copy_back_buffer_index(self, i)
        self.lastECMPosField[i] = self.lastECMPosFieldBuffer[i]
        self.lastECMField[i] = self.lastECMFieldBuffer[i]
        self.ecmPeriodField[i] = self.ecmPeriodFieldBuffer[i]

    def export_state(self):
        return FibroblastHandler.parent.export_state(self) | {
            "lastECMPosField": self.lastECMPosField.to_numpy(),
            "lastECMField": self.lastECMField.to_numpy(),
            "ecmPeriodField": self.ecmPeriodField.to_numpy(),

            "lastECMPosFieldBuffer": self.lastECMPosFieldBuffer.to_numpy(),
            "lastECMFieldBuffer": self.lastECMFieldBuffer.to_numpy(),
            "ecmPeriodFieldBuffer": self.ecmPeriodFieldBuffer.to_numpy(),
        }

    def load_state(self, data):
        FibroblastHandler.parent.load_state(self, data)

        self.lastECMPosField.from_numpy(data["lastECMPosField"])
        self.lastECMField.from_numpy(data["lastECMField"])
        self.ecmPeriodField.from_numpy(data["ecmPeriodField"])

        self.lastECMPosFieldBuffer.from_numpy(data["lastECMPosFieldBuffer"])
        self.lastECMFieldBuffer.from_numpy(data["lastECMFieldBuffer"])
        self.ecmPeriodFieldBuffer.from_numpy(data["ecmPeriodFieldBuffer"])