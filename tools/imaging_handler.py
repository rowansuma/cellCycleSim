import numpy as np
import imageio
from pathlib import Path
import subprocess

class ImagingHandler:
    def __init__(self, env):
        self.env = env
        self.fibroHandler = self.env.fibroHandler
        self.GRID_RES = env.GRID_RES
        self.MAX_COUNT_PER_CELL = self.env.MAX_IMAGE_PIXEL_CELLS

        self.fibro_pixel_map = np.zeros(
            (self.GRID_RES, self.GRID_RES), dtype=np.float32
        )

    def capture_image(self, path):
        path = path + "/frames"
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        grid_count_np = self.fibroHandler.gridCount.to_numpy()

        self.fibro_pixel_map[:] = grid_count_np / self.MAX_COUNT_PER_CELL
        np.clip(self.fibro_pixel_map, 0.0, 1.0, out=self.fibro_pixel_map)

        imageio.imwrite(
            path + f"/frame_{self.env.step[None]:06d}.png",
            (np.flipud(self.fibro_pixel_map.T) * 255).astype(np.uint8)
        )

    def save_video(self, path):
        run_dir = Path(path)

        cmd = [
            "ffmpeg",
            "-loglevel", "error",
            "-framerate", f"{self.env.VIDEO_FRAME_RATE}",
            "-pattern_type", "glob",
            "-i", f"{path}/frames/frame_*.png",
            "-pix_fmt", "yuv420p",
            "out.mp4"
        ]

        subprocess.run(cmd, check=True, cwd=run_dir)
