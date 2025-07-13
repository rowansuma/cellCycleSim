import random
from itertools import count
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.style.use('fivethirtyeight')

GENE_NAMES = [f'gene{i}' for i in range(11)]
GENE_LABELS = [
    'ORC1', 'CCNE1', 'CCNE2', 'MCM6', 'WEE1', 'CDK1', 'CCNF', 'NUSAP1', 'AURKA', 'CCNA2', 'CCNB2'
]

GENE_COLORS = [
    '#1f77b4',  # blue
    '#ff7f0e',  # orange
    '#2ca02c',  # green
    '#d62728',  # red
    '#9467bd',  # purple
    '#8c564b',  # brown
    '#e377c2',  # pink
    '#7f7f7f',  # gray
    '#333333',  # black
    '#17becf',  # cyan
    '#5716d9',  # dark purple
]

def animate2(i):
    data = pd.read_csv('data.csv')
    x = data['step']
    plt.cla()
    for idx, gene in enumerate(GENE_NAMES):
        if gene in data.columns:
            plt.plot(x, data[gene], label=GENE_LABELS[idx] if idx < len(GENE_LABELS) else gene, color=GENE_COLORS[idx % len(GENE_COLORS)])
    plt.xlabel('Simulation Step')
    plt.ylabel('Gene Expression Level')
    plt.title('Gene Expression in First Cell Over Time')
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()

ani = FuncAnimation(plt.gcf(), animate2, interval=1000)

plt.tight_layout()
plt.show()