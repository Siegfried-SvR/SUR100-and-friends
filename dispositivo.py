import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

#probando
class Dispositivo:
    def __init__(self, name):
        self.name = name

        # if type(parametro1) == tuple:
        self.x = None
        self.y = None
        self.slope = None
        self.offset = None
        self.R2 = None
        self.rutaCalibracion = None
        self.columnsNames = None

    def calibrar(self, filename):
        self.rutaCalibracion = filename
        df = pd.read_table(filename)

        self.columnsNames = list(df.columns)
        '''Dos detalles:
        1.utilizo el iloc para poder llamarlo por orden de columna. # También se podría llamar
        por nombre de columna pero creo que sería más engorroso (porque va a variar según dispositivo)
        2. Aplico "reshape(-1, 1) porque sino no funciona la función de LinearRegression"'''
        self.x = np.array(df.iloc[:, 0]).reshape(-1, 1)
        self.y = np.array(df.iloc[:, 1]).reshape(-1, 1)

        reg = linear_model.LinearRegression().fit(self.x, self.y)
        self.slope = float(reg.coef_)
        self.offset = float(reg.intercept_)
        self.R2 = reg.score(self.x, self.y)

        self.printReporteCalibracion()

    # def leerArchivoCalibracion(self, filename):

    def printReporteCalibracion(self):
        # Salida de terminal
        if self.slope and self.offset:
            print("Calibracion " + str(self.name))
            print("\tslope: " + str(self.slope))
            print("\toffset: " + str(self.offset))
            print("\tCoeCor: " + str(self.R2))
        else:
            print('Realizar el ajuste antes')

    def tension2Displacement(self, val):
        return (val - self.offset) / self.slope

    def displacement2Tension(self, val):
        return val * self.slope + self.offset

    def plotearCalibracion(self):
        # x, y, keyl = self.leerArchivoCalibracion()
        fig, ax = plt.subplots()
        line1, = ax.plot(self.x, self.y, "kx")
        line2, = ax.plot([min(self.x), max(self.x)],
                         [min(self.x) * self.slope + self.offset,
                          max(self.x) * self.slope + self.offset]
                         , "r--")
        ax.grid()
        ax.set_xlabel(self.columnsNames[0])
        ax.set_ylabel(self.columnsNames[1])
        ax.set_title("Curva de calibración - " + self.name)
        ax.legend((line1, line2), ("Datos", "Ajuste"))
        ax.text(
            0,
            (max(self.y) + min(self.y)) * .5, "y = " + str(np.round(self.slope, 4))
            + "x+" + str(np.round(self.offset
                                  , 4))
        )
        ax.text(0, (max(self.y) + min(self.y)) * .4, "R2: " + str(round(self.R2,4)))
        # plt.savefig(self.name + "_cal.png")
        plt.show()
        plt.clf()


if __name__ == '__main__':
    lvdt = Dispositivo('LVDT1')
    filename = "../datos/calibraciones/lvdt1.txt"
    lvdt.calibrar(filename)
    lvdt.plotearCalibracion()