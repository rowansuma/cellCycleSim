import taichi as ti

from particle.particle import ParticleHandler


@ti.data_oriented
class ECMHandler(ParticleHandler):
    parent = ParticleHandler

    def __init__(self, env):
        super().__init__(env, env.MAX_ECM_COUNT)

        self.ecmConnectPosField = ti.Vector.field(2, dtype=ti.f32, shape=self.MAX_COUNT)

        self.ecmConnectPosFieldBuffer = ti.Vector.field(2, dtype=ti.f32, shape=self.MAX_COUNT)

    @ti.func
    def update(self):
        ECMHandler.parent.update(self)

    @ti.func
    def clear_field_index(self, index):
        ECMHandler.parent.clear_field_index(self, index)
        self.ecmConnectPosField[index] = [-1, -1]

    @ti.func
    def initialize(self, idx: ti.i32, pos: ti.template()):
        ECMHandler.parent.initialize(self, idx, pos)
        self.ecmConnectPosField[idx] = self.posField[idx]

    @ti.func
    def write_buffer_index(self, buffer_i, i):
        ECMHandler.parent.write_buffer_index(self, buffer_i, i)
        self.ecmConnectPosFieldBuffer[buffer_i] = self.ecmConnectPosField[i]

    @ti.func
    def copy_back_buffer_index(self, i):
        ECMHandler.parent.copy_back_buffer_index(self, i)
        self.ecmConnectPosField[i] = self.ecmConnectPosFieldBuffer[i]

    def export_state(self):
        return ECMHandler.parent.export_state(self) | {
            "ecmConnectPosField": self.ecmConnectPosField.to_numpy(),

            "ecmConnectPosFieldBuffer": self.ecmConnectPosFieldBuffer.to_numpy()
        }

    def load_state(self, data):
        ECMHandler.parent.load_state(self, data)
        self.ecmConnectPosField.from_numpy(data["ecmConnectPosField"])

        self.ecmConnectPosFieldBuffer.from_numpy(data["ecmConnectPosFieldBuffer"])