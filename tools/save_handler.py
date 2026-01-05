from datetime import datetime
from pathlib import Path
import numpy as np

class SaveHandler:
    def __init__(self, handlers):
        self.handlers = handlers

    def save_state(self):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        save_dir = Path("savestates") / f"save_{timestamp}"
        save_dir.mkdir(parents=True, exist_ok=True)

        for tag, handler in self.handlers.items():
            np.savez_compressed(
                save_dir / f"{tag}_state.npz",
                **handler.export_state())

    def load_state(self, path):
        for tag, handler in self.handlers.items():
            handler.load_state(np.load(path+f"/{tag}_state.npz"))