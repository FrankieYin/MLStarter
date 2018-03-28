"""

This program is meant to be a visual aid to our main training program.

It's goal is to produce meaningful plots that reveal the relationship
between identification accuracy and hyperparameters.

"""

import matplotlib.pyplot as plt
import numpy as np

def plot_random_points(n):
    x = np.random.rand(9, )
    y = np.array([v + 0.07* np.random.uniform(-1, 1) for v in x ])
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    plt.scatter(x, y, c='k')
    plt.ylabel("y", rotation=0, fontweight='bold')
    plt.xlabel("x", fontweight="bold")
    ax.yaxis.set_label_coords(-0.1, 0.5)

    # fit the curve with 9 degree polynomial
    p = np.polyfit(x, y, 9)
    f = np.poly1d(p)

    x_new = np.linspace(np.min(x), np.max(x), 200)
    y_new = f(x_new)

    fig = plt.figure(2)
    ax_fit = fig.add_subplot(111)
    plt.plot(x, y, 'ko', x_new, y_new, 'k')
    ax_fit.set_ylim(0, 1)
    plt.ylabel("y", rotation=0, fontweight='bold')
    plt.xlabel("x", fontweight="bold")
    ax_fit.yaxis.set_label_coords(-0.1, 0.5)

    # fit with linear equation
    p = np.polyfit(x, y, 1)
    f = np.poly1d(p)

    x_new = np.linspace(np.min(x), np.max(x), 200)
    y_new = f(x_new)

    fig = plt.figure(3)
    ax_fit = fig.add_subplot(111)
    plt.plot(x, y, 'ko', x_new, y_new, 'k')
    ax_fit.set_ylim(0, 1)
    plt.ylabel("y", rotation=0, fontweight='bold')
    plt.xlabel("x", fontweight="bold")
    ax_fit.yaxis.set_label_coords(-0.1, 0.5)

def main():
    plot_random_points(9)
    plt.show()

if __name__ == '__main__':
    main()