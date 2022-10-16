"""
---------------------------------------------------------------
    The program analyzes the mismatch between derivative value
    computed between the discrete derivative and the derivative
---------------------------------------------------------------
    Autor: Jakub Krecisz                     Krakow, 27.04.2022
---------------------------------------------------------------
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

from config import POINT, POINT_AMOUNT, PRECISION, FILE_PLOT


def leftApproximation(h):
    return (np.sin(POINT) - np.sin(POINT - h)) / h

def rightApproximation(h):
    return (np.sin(POINT + h) - np.sin(POINT)) / h

def centralApproximation(h):
    return (np.sin(POINT + h) - np.sin(POINT - h)) / (2 * h)

def getDifference(function, hPoint):
    return np.absolute(function(hPoint) - np.cos(POINT))

def drawPlot(plot, dataType):
    plot.grid(True)
    plot.set_title(f'Mismatch in {dataType} type')
    hPoints = np.logspace(PRECISION[dataType], 0, num=POINT_AMOUNT, dtype=dataType)
    diffLeft = np.array(getDifference(leftApproximation, hPoints), dtype=dataType)
    diffCentral = np.array(getDifference(centralApproximation, hPoints), dtype=dataType)
    diffRight = np.array(getDifference(rightApproximation, hPoints), dtype=dataType)
    # plot.loglog(hPoints, diffLeft, 'tab:red')
    plot.loglog(hPoints, diffCentral, 'tab:green')
    plot.loglog(hPoints, diffRight, 'tab:blue')
    plot.legend(['Left Derivative', 'Central Derivative'])

def generatePlots():
    fig, axs = plt.subplots(2)
    fig.suptitle('Mismatch between discrete derivative and derivative\n'
                 f'Function: sin(x) Mismatch in point x={POINT}')
    fig.set_size_inches(7, 10)
    for ax in axs.flat:
        ax.set(xlabel="h", ylabel="$|D_hf(x) - f'(X)|$")

    drawPlot(axs[0], 'float32')
    drawPlot(axs[1], 'double')
    plt.savefig(FILE_PLOT, dpi=200)

def generateTables():
    dataType = 'float32'
    hPoints = np.logspace(PRECISION[dataType], 0, num=POINT_AMOUNT, dtype=dataType)
    diffLeft = np.array(getDifference(leftApproximation, hPoints), dtype=dataType)
    diffCentral = np.array(getDifference(centralApproximation, hPoints), dtype=dataType)
    diffRight = np.array(getDifference(rightApproximation, hPoints), dtype=dataType)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f'Wrong number of arguments!')
        sys.exit()

    if sys.argv[1] == 'plot':
        generatePlots()
    elif sys.argv[1] == 'table':
        generateTables()
    else:
        print(f'Bad argument! - [plot/table] instead of: {sys.argv[1]}')
        sys.exit()
