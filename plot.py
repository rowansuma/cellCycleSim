import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
import socket

# --- Seaborn Theme ---
sns.set_theme(style="darkgrid")

# --- Plot Settings ---
LINE_NAMES = ['population', 'ecm']
LINE_LABELS = ['# of Cells', 'ECM Count']
LINE_COLORS = ['#1f77b4', '#ff7f0e']

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

# --- Add Button to Same Figure ---
# The coordinate system is [left, bottom, width, height] in figure-relative units
ax_toggle_phase = fig.add_axes([0.83, 0.12, 0.15, 0.05])
ax_toggle_cells = fig.add_axes([0.83, 0.06, 0.15, 0.05])
ax_toggle_ecm = fig.add_axes([0.83, 0.00, 0.15, 0.05])

toggle_phase_button = Button(ax_toggle_phase, 'Toggle Phase')
toggle_cells_button = Button(ax_toggle_cells, 'Toggle Cells')
toggle_ecm_button = Button(ax_toggle_ecm, 'Toggle ECM')


# --- Socket Command Sender ---
def send_command(cmd):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect(('localhost', 65432))
            s.sendall(cmd.encode('utf-8'))
            response = s.recv(1024)
    except ConnectionRefusedError:
        print("Could not connect to main.py socket server.")

def toggle_phase(event):
    send_command("toggle_display_phase")

def toggle_cells(event):
    send_command("toggle_display_cells")

def toggle_ecm(event):
    send_command("toggle_display_ecm")

toggle_phase_button.on_clicked(toggle_phase)
toggle_cells_button.on_clicked(toggle_cells)
toggle_ecm_button.on_clicked(toggle_ecm)


# --- Animation Function ---
def animate(i):
    try:
        data = pd.read_csv('data/data.csv')
    except Exception:
        return

    x = data['step']

    # Clear previous plots
    ax1.cla()
    # ax2.cla()

    # Plot Population & ECM
    for idx, name in enumerate(LINE_NAMES):
        if name in data.columns:
            sns.lineplot(ax=ax1, x=x, y=data[name], label=LINE_LABELS[idx], color=LINE_COLORS[idx])

    ax1.set_ylabel('Count')
    ax1.set_title('Cell Population and ECM Count Over Time')
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
    ax1.set_xlabel('Simulation Step')
    # ax2.set_ylabel('Gene Expression Level')
    # ax2.set_title('Gene Expression in First Cell Over Time')
    # ax2.legend(loc='upper left', bbox_to_anchor=(1.05, 1))

# --- Start Animation ---
ani = FuncAnimation(fig, animate, interval=1000)

plt.show()
