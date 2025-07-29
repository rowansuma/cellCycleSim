import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Use seaborn style
sns.set_theme(style="darkgrid")

# Line settings for population + ECM
LINE_NAMES = ['population', 'ecm']
LINE_LABELS = ['# of Cells', 'ECM Count']
LINE_COLORS = ['#1f77b4', '#ff7f0e']

# Line settings for genes
GENE_NAMES = [f'gene{i}' for i in range(11)]
GENE_LABELS = [
    'ORC1', 'CCNE1', 'CCNE2', 'MCM6', 'WEE1', 'CDK1',
    'CCNF', 'NUSAP1', 'AURKA', 'CCNA2', 'CCNB2'
]
GENE_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#333333', '#17becf', '#5716d9'
]

# Create two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

def animate(i):
    data = pd.read_csv('data.csv')
    x = data['step']

    # Clear both subplots
    ax1.cla()
    ax2.cla()

    # --- Plot Cell Population & ECM ---
    for idx, name in enumerate(LINE_NAMES):
        if name in data.columns:
            sns.lineplot(ax=ax1, x=x, y=data[name], label=LINE_LABELS[idx], color=LINE_COLORS[idx])
    ax2.set_xlabel('Simulation Step')
    ax1.set_ylabel('Count')
    ax1.set_title('Cell Population and ECM Count Over Time')
    ax1.legend(loc='upper left', bbox_to_anchor=(1.05, 1))

    # --- Plot Gene Expressions ---
    for idx, gene in enumerate(GENE_NAMES):
        if gene in data.columns:
            sns.lineplot(ax=ax2, x=x, y=data[gene], label=GENE_LABELS[idx], color=GENE_COLORS[idx % len(GENE_COLORS)])
    ax2.set_xlabel('Simulation Step')
    ax2.set_ylabel('Gene Expression Level')
    ax2.set_title('Gene Expression in First Cell Over Time')
    ax2.legend(loc='upper left', bbox_to_anchor=(1.05, 1))

    plt.tight_layout()


# Animate
ani = FuncAnimation(fig, animate, interval=1000)

plt.tight_layout()
plt.show()
