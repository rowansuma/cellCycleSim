import numpy as np

class StatisticHandler:
    def __init__(self, env):
        self.env = env
        self.fibroHandler = self.env.fibroHandler

        self.GRID_RES = env.GRID_RES
        self.MAX_COUNT_PER_CELL = self.env.MAX_IMAGE_PIXEL_CELLS

        self.fibro_pixel_map = np.zeros(
            (self.GRID_RES, self.GRID_RES), dtype=np.float32
        )

        self.WOUND_THRESHOLD = 0.10

    def get_wound_area(self):
        grid_count_np = self.env.fibroHandler.gridCount.to_numpy()

        self.fibro_pixel_map[:] = grid_count_np / self.MAX_COUNT_PER_CELL
        wound_mask = self.fibro_pixel_map < self.WOUND_THRESHOLD

        pixel_count = np.sum(wound_mask)
        dx = self.env.DOMAIN_SIZE / self.env.GRID_RES
        area_um2 = pixel_count * dx * dx
        area_mm2 = area_um2 / 1000000

        return area_mm2

    def get_wound_width(self, row):
        grid_count_np = self.env.fibroHandler.gridCount.to_numpy()

        self.fibro_pixel_map[:] = grid_count_np / self.MAX_COUNT_PER_CELL
        wound_mask = self.fibro_pixel_map < self.WOUND_THRESHOLD

        if row < 0 or row >= wound_mask.shape[0]:
            raise ValueError("Row index out of bounds")

        wound_pixels = np.where(wound_mask[:, row])[0]

        if wound_pixels.size == 0:
            return 0.0

        width_pixels = wound_pixels[-1] - wound_pixels[0] + 1

        dx = self.env.DOMAIN_SIZE / self.env.GRID_RES
        width_um = width_pixels * dx

        return width_um

    def get_percent_closure(self):
        return 100*(self.env.INITIAL_WOUND_AREA - self.get_wound_area())/self.env.INITIAL_WOUND_AREA