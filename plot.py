import random
from itertools import count
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.style.use('fivethirtyeight')

LINE_NAMES = ['population', 'ecm']
LINE_LABELS = ['# of Cells', 'ECM Count']
LINE_COLORS = ['#1f77b4', '#ff7f0e']

def animate2(i):
    data = pd.read_csv('data.csv')
    x = data['step']
    plt.cla()
    for idx, name in enumerate(LINE_NAMES):
        if name in data.columns:
            plt.plot(x, data[name], label=LINE_LABELS[idx], color=LINE_COLORS[idx])
    plt.xlabel('Simulation Step')
    plt.ylabel('Count')
    plt.title('Cell Population and ECM Count Over Time')
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()

ani = FuncAnimation(plt.gcf(), animate2, interval=1000)

plt.tight_layout()
plt.show()