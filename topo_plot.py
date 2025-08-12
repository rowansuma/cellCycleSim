import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os

# --- Config ---
DATA_DIR = "data"
NMAP_FILE = os.path.join(DATA_DIR, "topo_map.npy")
OUT_IMAGE = os.path.join(DATA_DIR, "topo_map.png")
NUM_TIME_COLORS = 40

# --- Load Topo Map ---
if not os.path.exists(NMAP_FILE):
    raise FileNotFoundError(f"File not found: {NMAP_FILE}")

topo_map = np.load(NMAP_FILE)

# --- Create Colormap ---
# Use plasma colormap, skipping darkest part for better contrast
palette = plt.cm.plasma(np.linspace(0.2, 1.0, NUM_TIME_COLORS))
palette = np.vstack(([0, 0, 0, 1], palette))  # 0 = black = never visited
cmap = ListedColormap(palette)

# --- Plot and Save ---
plt.figure(figsize=(8, 8))
plt.imshow(topo_map.T, cmap=cmap, origin='lower')  # Transposed for correct orientation
plt.axis('off')
plt.title("Cell Expansion Over Time", fontsize=16)
plt.tight_layout()
plt.savefig(OUT_IMAGE, dpi=300)
plt.show()
