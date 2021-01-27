from functools import reduce
from datetime import datetime
import pandas as pd
import numpy as np


# Canal -> Clase para manejo de canales de adquisición de datos. Guarda
#         información de adquisición (x,y); contiene metodos para
#         lectura de distintas procedencias (Instron, DAQ...); metodos
#         básicos para corrimientos en X e Y (en general, modificaciones
#         permisibles sobre los datos que no alteren los resultados)
class Canal:
    def __init__(self, name, dispositivo, archivo):
        self.name = name
        self.dispositivo = dispositivo
        self.archivo = archivo
        self._x = []
        self._y = []

    # Lectura de datos de DAQ, col_x y col_y son los indices de las
    # columnas con los datos

    def mm_ss2seconds(self, mm_ss):
        x = datetime.strptime(mm_ss, '%M:%S.%f')
        ss = x.minute * 60 + x.second + x.microsecond / 1000000
        return ss

    def LoadUSB1208FS(self, col_x, col_y):
        df = pd.read_table(self.archivo, skiprows=(6), header=None)

        # paso la columna de tiempo para que sea solo segundos
        self._x = np.array(list(map(self.mm_ss2seconds, df.iloc[:, col_x].values)))
        self._y = df.iloc[:, col_y].values


    # Lectura de datos de la INSTRON8802, col_x y col_y son los indices
    # de las columnas con los datos
    def LoadINSTRON8802(self, col_x, col_y):
        df = pd.read_table(self.archivo, sep=";")
        self._x = df.iloc[:, col_x].values
        self._y = df.iloc[:, col_y].values

    def tension2HumanReadable(self):
        for k1 in range(0, len(self._y)):
            self._y[k1] = self.dispositivo.tension2HumanReadable(self._y[k1])

    def humanReadable2Tension(self):
        for k1 in range(0, len(self._y)):
            self._y[k1] = self.dispositivo.humanReadable2Tension(self._y[k1])

    # setea referencia de y restando a todo el valor del índice k
    # correspondiente
    def setRefy(self, k, L):
        y0 = 0
        k2 = 0
        for k1 in range(k - L, k + L + 1):
            y0 += self._y[k1]
            k2 += 1
        y0 /= k2
        # fig = plt.figure()
        # plt.plot(self._y[int(k-L):int(k+L+1)],"kx")
        # plt.plot([0,2*L+1],[y0,y0],"b")
        # plt.show()
        for k1 in range(0, len(self._y)):
            self._y[k1] -= y0

    # setea referencia de x restando a todo el valor del índice k
    # correspondiente
    #   resta _x[k] a todo _x
    #   filtra valores menores a _x[k]
    def setRefx(self, k):
        x0 = self._x[k]
        x_novo = []
        y_novo = []
        for k1 in range(0, len(self._x)):
            if self._x[k1] >= x0:
                x_novo.append(self._x[k1] - x0)
                y_novo.append(self._y[k1])
        self._x = x_novo
        self._y = y_novo

    # devuelve indice de valor de x
    def indicex(self, x):
        k0 = 0
        for k1 in self._x:
            if k1 >= x:
                return k0
            else:
                k0 += 1
        return k0

    # para un intervalo [k-Li;k+Li] de samples devuelve media y desviación
    def err1(self, k, La):
        m = 0
        for k1 in range(k - Li, k + Li + 1):
            m += self._y[k1]
        m /= 2 * Li + 1
        d = 0
        for k1 in range(k - Li, k + Li + 1):
            d += (self._y[k1] - m) ** 2
        d /= 2 * Li + 1
        return (m, d)

    # para un intervalo [k-Li;k+Li] de samples devuelve minima diferencia
    # entre dos puntos consecutivos
    def err2(self, k, Li):
        d = 0
        for k1 in range(k - Li, k + Li + 1):
            diff = np.abs(self._y[k1 + 1] - self._y[k1])
            if diff > 0 and (diff < d or d == 0):
                d = diff
        return d


if __name__ == '__main__':
    filename = "../datos/Apriete1/MCC1.csv"
    lvdt = None
    ch1 = Canal("LVDT2", None, filename)  # El canal 1 y 2 tienen los LVDTs 1 y 2 invertidos
    ch1.LoadUSB1208FS(1, 4)
