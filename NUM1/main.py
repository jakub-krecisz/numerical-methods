"""
---------------------------------------------------------------
    The program analyzes the mismatch between derivative value
    computed between the discrete derivative and the exact
    derivative in given point
---------------------------------------------------------------
    Autor: Jakub Krecisz                     Krakow, 27.04.2022
---------------------------------------------------------------
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import POINT, POINT_AMOUNT, PRECISION, \
    FILE_NAME_PLOT, FILE_NAME_TABLE


def rightApproximation(h):
    return (np.sin(POINT + h) - np.sin(POINT)) / h


def centralApproximation(h):
    return (np.sin(POINT + h) - np.sin(POINT - h)) / (2 * h)


def getDifference(function, hPoints):
    return np.abs(function(hPoints) - np.cos(POINT))


def drawPlot(plot, dataType):
    plot.grid(True)
    plot.set_title(f'Mismatch in {dataType} type')
    hPoints = np.logspace(PRECISION[dataType], 0, num=POINT_AMOUNT, dtype=dataType)
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
    plt.savefig(FILE_NAME_PLOT, dpi=200)


def generateTable(dataType):
    fig, ax = plt.subplots()
    ax.axis('off')
    ax.axis('tight')

    hPoints = np.logspace(PRECISION[dataType], 0, num=POINT_AMOUNT+1, dtype=dataType)
    diffCentral = np.array(getDifference(centralApproximation, hPoints), dtype=dataType)
    diffRight = np.array(getDifference(rightApproximation, hPoints), dtype=dataType)

    table = pd.DataFrame(data={'H': hPoints,
                               'Central Derivative difference': diffCentral,
                               'Right Derivative difference': diffRight})

    ax.table(cellText=table.values[::int((POINT_AMOUNT / 20))], colLabels=table.columns, loc='center')
    fig.tight_layout()
    plt.savefig(FILE_NAME_TABLE.format(daType=dataType))


if __name__ == '__main__':
    if sys.argv[1] == 'plot':
        generatePlots()
    elif sys.argv[1] == 'table':
        generateTable('float32' if sys.argv[2] == 'float32' else 'double')
    else:
        print(f'Bad argument! - [plot/table] instead of: {sys.argv[1]}')
        sys.exit()
