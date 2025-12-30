import taichi as ti

from particle.particle import ParticleHandler


@ti.data_oriented
class ECMHandler(ParticleHandler):
    parent = ParticleHandler

    def __init__(self, env):
        super().__init__(env, env.MAX_ECM_COUNT)

    @ti.func
    def update(self):
        ECMHandler.parent.update(self)

    @ti.func
    def clear_field_index(self, index):
        ECMHandler.parent.clear_field_index(self, index)

    @ti.func
    def initialize(self, idx: ti.i32, pos: ti.template()):
        ECMHandler.parent.initialize(self, idx, pos)

    @ti.func
    def write_buffer_index(self, buffer_i, i):
        ECMHandler.parent.write_buffer_index(self, buffer_i, i)

    @ti.func
    def copy_back_buffer_index(self, i):
        ECMHandler.parent.copy_back_buffer_index(self, i)

