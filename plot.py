import os
import shutil

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tomli
from matplotlib.animation import FuncAnimation

# --- Seaborn Theme ---
sns.set_theme(style="darkgrid")

if not os.path.exists("FISS/config.toml"):
    shutil.copyfile("FISS/defaultconfig.toml", "FISS/config.toml")
    print(f"Created config.toml from defaultconfig.toml")

with open('FISS/config.toml', 'rb') as f:
    config = tomli.load(f)

# --- Plot Settings ---
INDICES_SELECTED = [2]

title = 'Graph'
ylabel = 'Count'

FIBROBLAST_CELL_CYCLE_TICKS = config["cells"]["cell_cycle_duration"]

# title = 'Effects of Wound type on Wound Area over Time'
# ylabel = 'Wound Area (mm^2)'

LINE_NAMES = ["fibroblast_count", "ecm_count", "wound_area", "wound_width"]
LINE_LABELS = ["Fibroblast Population", "ECM Particle Count", "Wound Area (mm^2)", "Wound Width (μm)"]
LINE_COLORS = ["#82c6e2", "#5222a1", "#d65f5f", "#d5bb67"]

LINE_NAMES = [LINE_NAMES[i] for i in INDICES_SELECTED]
LINE_LABELS = [LINE_LABELS[i] for i in INDICES_SELECTED]
LINE_COLORS = [LINE_COLORS[i] for i in INDICES_SELECTED]

# LINE_NAMES = ["line", "circle", "triangle", "square"]
# LINE_LABELS = ["Line", "Circle", "Triangle", "Square"]
# LINE_COLORS = ["#ffd166", "#118ab2", "#06d6a0", "#ef476f"]

# GENE_NAMES = [f'gene{i}' for i in range(11)]
# GENE_LABELS = [
#     'ORC1', 'CCNE1', 'CCNE2', 'MCM6', 'WEE1', 'CDK1',
#     'CCNF', 'NUSAP1', 'AURKA', 'CCNA2', 'CCNB2'
# ]
# GENE_COLORS = [
#     '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
#     '#8c564b', '#e377c2', '#7f7f7f', '#333333', '#17becf', '#5716d9'
# ]

PHASE_NAMES = ['g0', 'g1', 's', 'g2/m']
PHASE_LABELS = ['G0', 'G1', 'S', 'G2/M']
PHASE_COLORS = ["#858585", "#66ccff", "#ffcc66", "#66ff66"]


# --- Create Figure and Subplots ---
fig, (ax1) = plt.subplots(1, 1, figsize=(10, 8), sharex=True, constrained_layout=True)


# --- Animation Function ---
def animate(i):
    try:
        data = pd.read_csv('FISS/data/data.csv')

        required_cols = ['step'] + [
            name for name in LINE_NAMES if name in data.columns
        ]

        data = data.dropna(subset=required_cols)

        if data.empty:
            return
    except Exception:
        return

    x = 24*data['step']/FIBROBLAST_CELL_CYCLE_TICKS

    # Clear previous plots
    ax1.cla()
    # ax2.cla()

    # Plot Population & ECM
    for idx, name in enumerate(LINE_NAMES):
        if name in data.columns:
            sns.lineplot(ax=ax1, x=x, y=data[name], label=LINE_LABELS[idx], color=LINE_COLORS[idx])

    ax1.set_ylabel(ylabel)
    ax1.set_title(title)
    ax1.legend(loc='upper left', bbox_to_anchor=(1.05, 1))

    # # Plot Population & ECM
    # for idx, name in enumerate(PHASE_NAMES):
    #     if name in data.columns:
    #         sns.lineplot(ax=ax2, x=x, y=data[name], label=PHASE_LABELS[idx], color=PHASE_COLORS[idx])
    #
    # ax2.set_xlabel('Simulation Step')
    # ax2.set_ylabel('Percentage of Cells')
    # ax2.set_title('Phase Distribution Over Time')
    # ax2.set_ylim(-1, 101)
    # ax2.legend(loc='upper left', bbox_to_anchor=(1.05, 1))

    # # Plot Gene Expressions
    # for idx, gene in enumerate(GENE_NAMES):
    #     if gene in data.columns:
    #         sns.lineplot(ax=ax2, x=x, y=data[gene], label=GENE_LABELS[idx], color=GENE_COLORS[idx % len(GENE_COLORS)])
    #
    ax1.set_xlabel('Hour')
    # ax2.set_ylabel('Gene Expression Level')
    # ax2.set_title('Gene Expression in First Cell Over Time')
    # ax2.legend(loc='upper left', bbox_to_anchor=(1.05, 1))

# --- Start Animation ---
ani = FuncAnimation(fig, animate, interval=10)

plt.show()
