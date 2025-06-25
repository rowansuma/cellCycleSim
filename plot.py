import random
from itertools import count
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.style.use('fivethirtyeight')


def animate2(i):
    data = pd.read_csv('data.csv')
    x = data['step']
    y5 = data['population']

    plt.cla()

    plt.plot(x, y5, label='# of Cells')

    plt.legend(loc='upper left')
    plt.tight_layout()


ani = FuncAnimation(plt.gcf(), animate2, interval=1000)

plt.tight_layout()
plt.show()