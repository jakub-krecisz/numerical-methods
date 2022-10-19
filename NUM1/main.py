"""
----------------------------------------------------------------
    The program analyzes the difference between derivative value
    computed between the discrete derivative and the exact
    derivative in given point
----------------------------------------------------------------
    Autor: Jakub Kręcisz                      Kraków, 16.10.2022
----------------------------------------------------------------
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import *


def rightApproximation(h):
    return (np.sin(POINT + h) - np.sin(POINT)) / h


def centralApproximation(h):
    return (np.sin(POINT + h) - np.sin(POINT - h)) / (2 * h)


def getDifference(function, hPoints):
    return np.abs(function(hPoints) - np.cos(POINT))


def drawPlot(plot, dataType):
    plot.grid(True)
    plot.set_title(f'Mismatch in {dataType} type')

    hPoints = np.logspace(-PRECISION[dataType], 0, num=POINT_AMOUNT + 1, dtype=dataType)
    diffCentral = np.array(getDifference(centralApproximation, hPoints), dtype=dataType)
    diffRight = np.array(getDifference(rightApproximation, hPoints), dtype=dataType)

    plot.loglog(hPoints, diffCentral, 'tab:green')
    plot.loglog(hPoints, diffRight, 'tab:blue')
    plot.legend(['Central Derivative', 'Right Derivative'])


def generatePlots():
    fig, axs = plt.subplots(2)
    fig.suptitle('Mismatch between discrete derivative and exact derivative\n'
                 f'Function: sin(x) Mismatch in point x={POINT}')
    fig.set_size_inches(7, 10)
    for ax in axs.flat:
        ax.set(xlabel="h", ylabel="$|D_hf(x) - f'(X)|$")

    drawPlot(axs[0], 'float32')
    drawPlot(axs[1], 'double')
    plt.savefig(FILE_NAME_PLOT, dpi=300)


def printTable(dataType):
    hPoints = np.logspace(-PRECISION[dataType], 0, num=POINT_AMOUNT + 1)
    diffCentral = np.array(getDifference(centralApproximation, hPoints))
    diffRight = np.array(getDifference(rightApproximation, hPoints))

    df = pd.DataFrame(data={'H': hPoints,
                            'Central Derivative difference': diffCentral,
                            'Right Derivative difference': diffRight})

    with pd.option_context('display.float_format', lambda x: f'{x:,.{PRECISION[dataType]}f}'):
        print(df[::int(NUM_OF_ROWS)])


if __name__ == '__main__':
    if sys.argv[1] == 'plot':
        generatePlots()
    elif sys.argv[1] == 'table':
        printTable(sys.argv[2])
    else:
        print(f'Bad argument! - [plot/table] instead of: {sys.argv[1]}')
        sys.exit()
