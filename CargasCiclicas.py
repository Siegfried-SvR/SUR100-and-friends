import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from scipy.signal import find_peaks


#Dispositivo -> clase para asignación de dispositivos de medición: LVDT,
#               celda de carga, actuador. Contiene datos referentes a
#               la calibración y conversión de datos.
class Dispositivo:
    
    #inicialización: -> si parametro 1 es una tupla, se asigna el primer y
    #                segundo valor como pendiente y ordenada
    #                -> si parametro 1 es un string, se lo considera como
    #                archivo con datos de calibración
    #                -> si no se ingresa parametro1 queda (1,0) por defecto
    def __init__(self,name,parametro1=(1,0)):
        self.name = name
        
        if type(parametro1) == tuple:
            self.slope = parametro1[0]
            self.offset = parametro1[1]
        
        elif type(parametro1) == str:
            self.rutaCalibracion = parametro1
            self.slope, self.offset, self.R2 = self.calibrar()
            self.reporteCalibracion()
            self.plotearCalibracion()
        
    def calibrar(self):
        x,y,keyl = self.leerArchivoCalibracion()
        v = ajusteLinealOrden1(x,y)
        R2 = rCuadrado(x,y,v)
        
        return (v[0],v[1],R2)
    
    def leerArchivoCalibracion(self):
        df = pd.read_table(self.rutaCalibracion)
        dt = df.to_dict()
        
        #recupero claves
        keyl = []
        for k1 in dt:
            keyl.append(k1)
        n = len(dt[keyl[0]])
        
        #recupero datos
        x = np.zeros(n)
        y = np.zeros(n)
        for k1 in range(0,n):
            x[k1] = float(dt[keyl[0]][k1])
            y[k1] = float(dt[keyl[1]][k1])
        return (x,y,keyl)
        
    def plotearCalibracion(self):
        x,y,keyl = self.leerArchivoCalibracion()
        fig = plt.figure()
        line1, = plt.plot(x,y,"kx")
        line2, = plt.plot([min(x),max(x)],
                          [min(x)*self.slope + self.offset,
                           max(x)*self.slope + self.offset]
                          ,"b--")
        plt.grid()
        plt.xlabel(keyl[0])
        plt.ylabel(keyl[1])
        plt.title("Curva de calibración - " + self.name)
        plt.legend((line1,line2),("Datos","Ajuste"))
        plt.text(
                    0,
                    (max(y) + min(y))*.5, "y = "+str(np.round(self.slope, 4))
                    + "x+" + str(np.round(self.offset
                    , 4))
                )
        plt.text(0, (max(y)+min(y))*.4, "R2: "+str(self.R2))
        plt.savefig(self.name+"_cal.png")
        plt.clf()
        
    def reporteCalibracion(self):
        #Salida de terminal
        print("Calibracion "+str(self.name))
        print("\tslope: "+str(self.slope))
        print("\toffset: "+str(self.offset))
        print("\tCoeCor: "+str(self.R2))
    
    def tension2HumanReadable(self,val):
        return (val - self.offset)/self.slope
    
    def humanReadable2Tension(self,val):
        return val*self.slope + self.offset
    

#Canal -> Clase para manejo de canales de adquisición de datos. Guarda
#         información de adquisición (x,y); contiene metodos para
#         lectura de distintas procedencias (Instron, DAQ...); metodos
#         básicos para corrimientos en X e Y (en general, modificaciones
#         permisibles sobre los datos que no alteren los resultados)
class Canal:
    
    def __init__(self,name,Dispositivo,archivo):
        self.name = name
        self.Dispositivo = Dispositivo
        self.archivo = archivo
        self._x = []
        self._y = []
    
    #Lectura de datos de DAQ, col_x y col_y son los indices de las
    #columnas con los datos
    def LoadUSB1208FS(self,col_x,col_y):
        df = pd.read_table(self.archivo,skiprows=(6))
        df = df.to_dict()
        #recupero las claves de los diccionarios
        keyl=[]
        for k1 in df:
            keyl.append(k1)
        n = len(df[keyl[0]])
        
        self._x = np.zeros(n)
        self._y = np.zeros(n)
        
        for k1 in range(0,n):
            #Pelea con el formato de fechas (tiempo) del DAQ
            #Formatos posibles: dd.hh:mm:ss
            #                   hh:mm:ss
            #                   mm:ss.ms
            if col_x == 1:
                cadenaDeString = df[keyl[col_x]][k1]
                cadenaDeString = cadenaDeString.split(":")
                
                if  (len(cadenaDeString)==3 and
                    len(cadenaDeString[0].spit("."))>1):
                        var = (24*3600*float(cadenaDeString[0].split(".")[0])
                            + 3600*float(cadenaDeString[0].split(".")[1])
                            + 60*float(cadenaDeString[1])
                            + float(cadenaDeString[2]))
                    
                elif  (len(cadenaDeString)==3 and 
                       len(cadenaDeString[0].spit("."))==1):
                        var = (3600*float(cadenaDeString[0])
                            + 60*float(cadenaDeString[1])
                            + float(cadenaDeString[2]))
                                    
                elif  len(cadenaDeString)==2:
                        var = (60*float(cadenaDeString[0])
                            + float(cadenaDeString[1]))
                self._x[k1] = float(var)
                
            else:
                self._x[k1] = float(df[keyl[col_x]][k1])
            self._y[k1] = float(df[keyl[col_y]][k1])
    
    #Lectura de datos de la INSTRON8802, col_x y col_y son los indices
    #de las columnas con los datos
    def LoadINSTRON8802(self,col_x,col_y):
        df = pd.read_table(self.archivo,sep=";")
        df = df.to_dict()
        #recupero las claves de los diccionarios
        keyl=[]
        for k1 in df:
            keyl.append(k1)
        n = len(df[keyl[0]])
        
        self._x = np.zeros(n)
        self._y = np.zeros(n)
        for k1 in range(0,n):
            self._x[k1] = float(df[keyl[col_x]][k1])
            self._y[k1] = float(df[keyl[col_y]][k1])
    
    def tension2HumanReadable(self):
        for k1 in range(0,len(self._y)):
            self._y[k1] = self.Dispositivo.tension2HumanReadable(self._y[k1])
    
    def humanReadable2Tension(self):
        for k1 in range(0,len(self._y)):
            self._y[k1] = self.Dispositivo.humanReadable2Tension(self._y[k1])
    
    #setea referencia de y restando a todo el valor del índice k
    #correspondiente
    def setRefy(self,k,L):
        y0 = 0
        k2 = 0
        for k1 in range(k-L,k+L+1):
            y0+=self._y[k1]
            k2+=1
        y0/=k2
        #fig = plt.figure()
        #plt.plot(self._y[int(k-L):int(k+L+1)],"kx")
        #plt.plot([0,2*L+1],[y0,y0],"b")
        #plt.show()
        for k1 in range(0,len(self._y)):
            self._y[k1] -= y0
    
    #setea referencia de x restando a todo el valor del índice k
    #correspondiente
    #   resta _x[k] a todo _x
    #   filtra valores menores a _x[k]
    def setRefx(self,k):
        x0 = self._x[k]
        x_novo = []
        y_novo = []
        for k1 in range(0,len(self._x)):
            if self._x[k1] >= x0:
                x_novo.append(self._x[k1]-x0)
                y_novo.append(self._y[k1])
        self._x = x_novo
        self._y = y_novo
    
    #devuelve indice de valor de x
    def indicex(self,x):
        k0=0
        for k1 in self._x:
            if k1 >= x:
                return k0
            else:
                k0+=1
        return k0
    
    #para un intervalo [k-Li;k+Li] de samples devuelve media y desviación
    def err1(self,k,La):
        m = 0
        for k1 in range(k-Li,k+Li+1):
            m += self._y[k1]
        m/=2*Li+1
        d=0
        for k1 in range(k-Li,k+Li+1):
            d += (self._y[k1]-m)**2
        d/=2*Li+1
        return (m,d)
    
    #para un intervalo [k-Li;k+Li] de samples devuelve minima diferencia
    #entre dos puntos consecutivos
    def err2(self,k,Li):
        d=0
        for k1 in range(k-Li,k+Li+1):
            diff = np.abs(self._y[k1+1]-self._y[k1])
            if diff > 0 and (diff <d or d == 0):
                d = diff
        return d


#OperacionCanales -> Clase para realizar operaciones sobre canales
#                    más cercanas a la manipulación de datos:
#                    promedios, media movil
class OperacionCanales:
    
    def __init__(self,name,ListaCanales):
        self.name = name
        self.ListaCanales = list(ListaCanales)
        if len(ListaCanales)==1:
            self._x = self.ListaCanales[0]._x
            self._y = self.ListaCanales[0]._y
    
    def promedio(self):
        n1 = len(self.ListaCanales[0]._y)
        n2 = len(self.ListaCanales)
        self._x = self.ListaCanales[0]._x
        self._y = []
        for k1 in range(0,n1):
            s = 0
            for k2 in range(0,n2):
                s+=self.ListaCanales[k2]._y[k1]
            self._y.append(s/n2)
    
    #busco el indice correspondiente a un valor de x arbitrario.
    def indicex(self,x):
        k0=0
        for k1 in self._x:
            if k1 >= x:
                return k0
            else:
                k0+=1
        return k0
    
    #calcula media movil considerando una ventana [-L1, L2]
    #a intervalos de p
    def mediamovil(self,L1,L2,p):
        if p == 0:
            p=1
        mmx = []
        mmy = []
        n = len(self._x)
        k1 = L1
        while k1 <= n-L2-1:
            s = 0
            for k2 in range(k1-L1,k1+L2+1):
                s+=self._y[k2]
            s/=(L2+L1+1)
            mmx.append(self._x[k1])
            mmy.append(s)
            k1+=p
        self._x = mmx
        self._y = mmy
    
    #reduce valores de acuerdo a los parámetros indicados generando
    #una ventana [-L1, L2] a intervalos de p
    def screening(self,L1,L2,p):
        if p == 0:
            p = 1
        sx = []
        sy = []
        n = len(self._x)
        k1 = L1
        while k1 <= n-L2-1:
            sx.append(self._x[k1])
            sy.append(self._y[k1])
            k1+=p
        self._x = sx
        self._y = sy

##### FUNCIONES #####

#Rutina sencilla para identificación de zeros
#busca primer valor de x que haga que y cambie
#de signo. Por interpolación entre dos puntos
#devuelve un zero.
def buscar_zero(x,y):
    n = len(x)
    for k in range(0,n-1):
        if x[k]*x[k+1] <= 0:
            a = (y[k+1] - y[k])/(x[k+1] - x[k])
            b = y[k] - a*x[k]
            return b

#Ajuste lineal de primer orden
def ajusteLinealOrden1(x,y):
    n = len(x)
    X = np.reshape(x,(n,1))
    Y = np.reshape(y,(n,1))
    M = np.concatenate(         #Esta matriz gobierna el orden de ajuste
                        (
                           X,
                           np.ones((n,1))
                        ),
                        axis=1
                      )
    Mt = M.T
    Y = np.matmul(Mt,Y)
    M = np.matmul(Mt,M)
    M = np.linalg.inv(M)
    v = np.matmul(M,Y)
    return v

#Coeficiente de correlación
def rCuadrado(x,y,v):
    #x e y son listas de ajuste, v contiene los parámetros
    #de ajuste en orden descendiente.
    SStot = 0
    SSres = 0
    Ym = 0
    for k1 in y:
        Ym += k1
    Ym/=len(y)
    for k1 in range(0,len(y)):
        SStot += (y[k1] - Ym)**2
        SSres += (y[k1] - (v[0]*x[k1] + v[1]))**2
    return 1 - SSres/SStot

#Calcula media y desviación estándar
def med_dev(x):
    n = len(x)
    m = 0
    for k1 in range(0,n):
        m += x[k1]
    m/=n
    d=0
    for k1 in range(0,n):
        d += (x[k1]-m)**2
    d/=n
    return (m,d)


##### ANALISIS #####


#Cargo y calibro LVDTs y celda de carga
lvdt1 = Dispositivo("LVDT1","./calibraciones/lvdt1.txt")
lvdt2 = Dispositivo("LVDT2","./calibraciones/lvdt2.txt")
lvdt3 = Dispositivo("LVDT3","./calibraciones/lvdt3.txt")
lvdt4 = Dispositivo("LVDT4","./calibraciones/lvdt4.txt")
carga = Dispositivo("Celda de carga",(2/5,0))


#archivos a leer
F1 = "MCC1.csv"
F2 = "MCC2.csv"


#Cargo canales, leo los archivos y convierto tensión a desplazamiento
ch1 = Canal("LVDT2",lvdt2,F1) #El canal 1 y 2 tienen los LVDTs 1 y 2 invertidos
ch1.LoadUSB1208FS(1,2)
ch1.tension2HumanReadable()

ch2 = Canal("LVDT1",lvdt1,F1)
ch2.LoadUSB1208FS(1,3)
ch2.tension2HumanReadable()

ch3 = Canal("LVDT3",lvdt3,F1)
ch3.LoadUSB1208FS(1,4)
ch3.tension2HumanReadable()

ch4 = Canal("LVDT4",lvdt4,F2)
ch4.LoadUSB1208FS(1,2)
ch4.tension2HumanReadable()

ch5 = Canal("LVDT6",carga,F2)
ch5.LoadUSB1208FS(1,5)
ch5.tension2HumanReadable()

#Seteo las referencias de los LVDTs a un promedio de los primeros
#600 samples
ch1.setRefy(300,300)
ch2.setRefy(300,300)
ch3.setRefy(300,300)
ch4.setRefy(300,300)

#Promedio los 4 canales de los LVDTs y aplico media movil
desp = OperacionCanales("promedio",[ch1,ch2,ch3,ch4])
desp.promedio()
desp.mediamovil(50,50,10)
carg= OperacionCanales("Carga",[ch5])
carg.mediamovil(50,50,10)

#Resumen de operaciones:
#                       -No hay operaciones, solo grafico de datos.
#                       -toma un acanal de desplazamiento cualquiera
#                       -recupera los picos de la señal
#                       -grafica los resultados y pide hacer un clic
#                        para tomar una referencia
#                       -A partir de la referencia, considera 10 ciclos
#                        para graficar la señal de los 6 LVDTs en
#                        función de la carga registrada

#grafico carga y pido 2 referencias para limitar la triangular
fig = plt.figure(1)
plt.plot(ch5._x,ch5._y,"k")
plt.grid()
plt.xlabel("Tiempo /s")
plt.ylabel("Carga /kN")
pts=plt.ginput(2)

#recupero el indice correspondiente a la primer referencia
ka = carg.indicex(pts[0][0])
kb = carg.indicex(pts[1][0])

#grafico desplazamiento carga
fig = plt.figure(2)
lin1, = plt.plot(carg._y[:ka],desp._y[:ka],"darkgray")
lin2, = plt.plot(carg._y[ka:kb],desp._y[ka:kb],"k")
plt.legend((lin1,lin2), ("Remoción del pad","Triangular"))
plt.grid()
plt.axis((0,22.5,-1.5,1.5))
plt.title("Curva desplazamiento-carga")
plt.ylabel("Desplazamiento /mm")
plt.xlabel("Carga /kN")
plt.savefig("Desplazamiento-Carga.png")
plt.show()
plt.clf()

#grafico programa de cargas
fig = plt.figure(3)
plt.plot(carg._x,carg._y,"k")
plt.grid()
plt.xlabel("Tiempo /s")
plt.ylabel("Carga /kN")
plt.axis((0,max(carg._x),0,25))
plt.savefig("Carga (L).png")

#recupero indices con las dos referencias
ka = carg.indicex(pts[0][0])
kb = carg.indicex(pts[1][0])

#La siguiente rutina es para identificar curvas de carga
intervalos = [] #lista con pares de indices indicando intervalos
                #de carga creciente
control1 = 2 #Dentro de un intervalo relevante.
             #0: minimo, 1: dentro, 2: maximo
control2 = 0 #samples a considerar antes de la
             #siguiente medición
tolerancia = (max(carg._y[ka:kb]) 
            - min(carg._y[ka:kb]))*.48 #desviacion del target aceptado
target = (max(carg._y[ka:kb]) 
         + min(carg._y[ka:kb]))/2 #carga esperada

lista_p0 = [] #valores de fuerza de apriete
fig = plt.figure(4) #grafico de control
for k3 in range(ka,kb):
    if carg._y[k3] < target - tolerancia and control1 == 2:
        control1 = 0
    elif (carg._y[k3]<target+tolerancia
          and control1 == 0
          and carg._y[k3] > target - tolerancia):
        control1 = 1
        intervalos.append(k3) #Entrada a curva de carga
    elif carg._y[k3]>target+tolerancia and control1 == 1:
        control1 = 2
        intervalos.append(k3) #salida de curva de carga
        plt.plot(
                  carg._y[intervalos[-2]:intervalos[-1]],
                  desp._y[intervalos[-2]:intervalos[-1]]
                )
        p0 = buscar_zero(
                         desp._y[intervalos[-2]:intervalos[-1]],
                         carg._y[intervalos[-2]:intervalos[-1]]
                        )
        plt.plot([p0],[0],"kx")
        lista_p0.append(p0)
plt.grid()
plt.show()

print(lista_p0)
F,dev = med_dev(lista_p0)
print("Fuerza de apriete: "+str(F)+"kN")
print("Desviación: "+str(dev)+"kN")
