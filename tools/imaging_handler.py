import numpy as np
import imageio
from pathlib import Path

class ImagingHandler:
    def __init__(self, env):
        self.env = env
        self.GRID_RES = env.GRID_RES
        self.MAX_COUNT_PER_CELL = self.env.MAX_IMAGE_PIXEL_CELLS

        self.fibro_pixel_map = np.zeros(
            (self.GRID_RES, self.GRID_RES), dtype=np.float32
        )

    def capture_image(self, path):
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        grid_count_np = self.env.fibroHandler.gridCount.to_numpy()

        self.fibro_pixel_map[:] = grid_count_np / self.MAX_COUNT_PER_CELL

        np.clip(self.fibro_pixel_map, 0.0, 1.0, out=self.fibro_pixel_map)

        imageio.imwrite(
            path+f"/frame_{self.env.step[None]:06d}.png",
            (self.fibro_pixel_map * 255).astype(np.uint8)
        )