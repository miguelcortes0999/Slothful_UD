import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt

class PromedioMovil:
    def __init__(self, datos: (List, Tuple, pd.DataFrame), ventana: int = 3, pronostico: int = 1):
        '''
        PromedioMovil(datos(df,list,tuple), ventana:int, periodos:int(1))\n
        Calcula el promedio móvil simple de una serie de datos.\n
        Argumentos:
        datos: una lista, tupla o DataFrame con los datos de la serie.
        ventana: un entero que indica el tamaño de la ventana de promedio móvil.
        periodos: un entero opcional que indica el número de periodos a pronosticar.
                    Por defecto es 1.
        Ejemplo:\n
        datos = [1, 2, 3, 4, 5, 6]\n
        df= pd.DataFrame(datos)\n
        pm = PromedioMovil(df, ventana=4, pronostico=3)\n
        pm.calcular_promedio_movil()\n
        print(pm.resultado)\n
        pm.graficar()\
        '''
        self.datos = datos
        self.ventana = ventana
        self.pronostico = pronostico
        
    def calcular_promedio_movil(self):
        # Comprobar el tipo de dato de usuario
        if isinstance(self.datos, pd.DataFrame):
            self.historico = self.datos.copy()
        elif isinstance(self.datos, (list, tuple)):
            self.historico = pd.DataFrame(self.datos)
        else:
            raise TypeError("El tipo de datos ingresado no es válido.")
        # Calcular el promedio movil el numero suministrado
        pronostico = self.historico.copy()
        for _ in range(1,self.pronostico+1):
            pronostico.loc[ len(pronostico)+1] = [list(pronostico.iloc[:,0].rolling(window=self.ventana).mean())[-1]]
        # Devolver el resultado
        self.resultado = pronostico.shift(1)
    
    # Grafica el pronostico
    def graficar(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.historico.index, self.historico.iloc[:, 0], label='Histórico')
        plt.plot(self.resultado.index, self.resultado.iloc[:, 0], label='Pronóstico')
        plt.xlabel('Fecha')
        plt.ylabel('Valor')
        plt.title(f'Promedio móvil ({self.ventana} periodos, {self.pronostico} periodos pronosticados)')
        plt.legend()
        plt.show()

#datos = [1, 2, 3, 4, 5, 6]
#df= pd.DataFrame(datos)
#pm = PromedioMovil(df, ventana=4, pronostico=3)
#pm.calcular_promedio_movil()
#print(pm.resultado)
#pm.graficar()

class PromedioMovilPonderado:
    def __init__(self, datos: (List, Tuple, pd.DataFrame), ventana: int = None, pronostico: int = 1, pesos: List[float] = None):
        '''
        PromedioMovilPonderado(datos(df,list,tuple), ventana:int, periodos:int(1), pesos:List[float] = None)\n
        Calcula el promedio móvil ponderado de una serie de datos.\n
        Argumentos:
        datos: una lista, tupla o DataFrame con los datos de la serie.
        ventana: un entero que indica el tamaño de la ventana de promedio móvil.
        periodos: un entero opcional que indica el número de periodos a pronosticar.
                    Por defecto es 1.
        pesos: una lista opcional de flotantes que indica los pesos correspondientes a cada valor en la ventana.
                    Si no se especifica, se utilizará un promedio móvil simple.
        Ejemplo:\n
        datos = [1, 2, 3, 4, 5, 6]\n
        df= pd.DataFrame(datos)\n
        pm = PromedioMovil(df, ventana=4, pronostico=3, pesos=[0.1, 0.2, 0.3, 0.4])\n
        pm.calcular_promedio_movil()\n
        print(pm.resultado)\n
        pm.graficar()\
        '''
        self.datos = datos
        # Comprobar el tipo de dato de usuario
        if isinstance(self.datos, pd.DataFrame):
            self.historico = self.datos.copy()
        elif isinstance(self.datos, (list, tuple)):
            self.historico = pd.DataFrame(self.datos)
        else:
            raise TypeError("El tipo de datos ingresado no es válido.")
        # Comprobación de longitudes y tamaños similares
        if ventana is None and pesos is None:
            raise TypeError("No ingreso pesos ni tamaño de la ventana válido.")
        
        if ventana is not None and pesos is not None:
            if ventana != len(pesos):
                raise TypeError("Tamaño de pesos y ventana distintos.")
            else:
                self.ventana = ventana  
                self.pesos = pesos 
                self.pronostico = pronostico
        else:
            if ventana is None:
                if sum(pesos)!=1:
                    raise TypeError("Suma de los pesos es diferente de 1.")
                self.ventana = len(pesos)
                self.pesos = pesos 
                self.pronostico = pronostico
            if pesos is None:
                if ventana<1:
                    raise TypeError("La ventana no es posible, debe ser >= 1.")
                if ventana>len(datos):
                    raise TypeError("La ventana es mas grande que los datos historicos.")
                self.ventana = ventana
                self.pesos = [1/ventana]*ventana
                self.pronostico = pronostico
        
    def calcular_promedio_movil_ponderado(self):            
        # Calcular el promedio móvil ponderado para el número de periodos suministrados
        pronostico_pmp = self.historico.copy()
        for _ in range(self.pronostico):
            vector_historico = np.array(pronostico_pmp.iloc[-self.ventana:, 0])
            vector_pesos = np.array(self.pesos)
            producto_punto = np.dot(vector_historico, vector_pesos)
            pronostico_pmp.loc[len(pronostico_pmp)] = producto_punto
        self.resultado = pronostico_pmp
    
    # Grafica el pronóstico
    def graficar(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.historico.index, self.historico.iloc[:, 0], label='Histórico')
        plt.plot(self.resultado.index, self.resultado.iloc[:, 0], label='Pronóstico')
        plt.xlabel('Fecha')
        plt.ylabel('Valor')
        plt.title(f'Promedio móvil ponderado ({self.ventana} periodos, {self.pronostico} periodos pronosticados)')
        plt.legend
        plt.show()

#datos = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#df= pd.DataFrame(datos)
#pmp = PromedioMovilPonderado(df, ventana=4, pronostico=5, pesos=[0.1,0.2,0.3,0.4])
#pmp.calcular_promedio_movil_ponderado()
#print(pmp.resultado)
#pmp.graficar()

class RegresionLinealSimple():
    def __init__(self, datos: (List, Tuple, pd.DataFrame)):
        '''
        RegresionLienalSimple(datos(df,list,tuple))\n
        Calcula la regresion lineal para una serie de datos.\n
        Argumentos:
        datos: una lista, tupla o DataFrame con los datos de la serie (el inidce del data frame sera tomado como eje x).
        Ejemplo:\n
        datos = [5,7,9,11,13,15,17,19,21,23,25]\n
        reg = RegresionLinealSimple(datos)\n
        reg.calcular_regresion()\n
        print(reg.ecuacion())\n
        print(reg.predecir([45,60,120,34]))\n
        pm.graficar()\n
        '''
        self.datos = datos
        # Comprobar el tipo de dato de usuario
        if isinstance(self.datos, pd.DataFrame):
            self.historico = self.datos.copy()
        elif isinstance(self.datos, (list, tuple)):
            self.historico = pd.DataFrame(self.datos)
        else:
            raise TypeError("El tipo de datos ingresado no es válido.")
        self.x = np.array(list(self.historico.index))
        self.y = np.array(list(self.historico.iloc[:,0]))
    
    def calcular_regresion(self):
        self.b = None
        self.m = None
        x_mean = np.mean(self.x)
        y_mean = np.mean(self.y)
        s_xy = np.sum((self.x - x_mean) * (self.y - y_mean))
        s_xx = np.sum((self.x - x_mean) ** 2)
        self.m = s_xy / s_xx
        self.b = y_mean - self.m * x_mean
    
    def predecir(self, x_nuevos: np.array):
        if self.b is None or self.m is None:
            raise ValueError("La regresión aún no se ha calculado.")
        resultado = pd.DataFrame([None for x in x_nuevos],index=x_nuevos)
        for xs in x_nuevos:
            resultado.loc[xs,0] = self.b + self.m * xs
        return resultado

    def ecuacion(self):
        return {'m':self.m, 'b':self.b}

    def graficar(self):
        plt.figure(figsize=(12, 6))
        plt.scatter(self.historico.iloc[:, :-1], self.historico.iloc[:, -1], label='Histórico')
        plt.plot(self.historico.iloc[:, :-1], self.predecir(np.array(self.historico.iloc[:, :-1])), label='Regresión lineal simple')
        plt.xlabel('Variable independiente')
        plt.ylabel('Variable dependiente')
        plt.title('Regresión lineal simple')
        plt.legend()

#datos = [5,7,9,11,13,15,17,19,21,23,25]
#reg = RegresionLinealSimple(datos)
#reg.calcular_regresion()
#print(reg.ecuacion())
#print(reg.predecir([45,60,120,34]))
    
class SuavizacionExponencialTriple():
    def __init__(self, datos: (List, Tuple, pd.DataFrame), alpha: float, beta: float, gamma: float, nivel_inicial: Optional[float] = None, tendencia_inicial: Optional[float] = None, estacionalidad_inicial: Optional[List[float]] = None):
        self.datos = datos
        # Comprobar el tipo de dato de usuario
        if isinstance(self.datos, pd.DataFrame):
            self.historico = self.datos.copy()
        elif isinstance(self.datos, (list, tuple)):
            self.historico = pd.DataFrame(self.datos)
        else:
            raise TypeError("El tipo de datos ingresado no es válido.")
        self.y = np.array(self.historico.iloc[:,0])
        self.x = np.array(self.historico.index)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.nivel_inicial = nivel_inicial
        self.tendencia_inicial = tendencia_inicial
        self.estacionalidad_inicial = estacionalidad_inicial
        self.nivel = None
        self.tendencia = None
        self.estacionalidad = None
        self.m = 0

    def calcular_regresion(self):
        n = len(self.y)
        # Inicializar los valores de nivel, tendencia y estacionalidad
        nivel = [self.y[0]] if self.nivel_inicial is None else [self.nivel_inicial]
        tendencia = [self.y[1] - self.y[0]] if self.tendencia_inicial is None else [self.tendencia_inicial]
        estacionalidad = [self.y[i] - nivel[0] for i in range(n)] if self.estacionalidad_inicial is None else self.estacionalidad_inicial.copy()

        # Calcular los valores de nivel, tendencia y estacionalidad para cada período
        for i in range(1, n):
            # Calcular los valores suavizados
            nivel_actual = self.alpha * (self.y[i] - estacionalidad[i - 1]) + (1 - self.alpha) * (nivel[i - 1] + tendencia[i - 1]) # cambiamos - por / (self.y[i] - nivel_actual)
            tendencia_actual = self.beta * (nivel_actual - nivel[i - 1]) + (1 - self.beta) * tendencia[i - 1]
            estacionalidad_actual = self.gamma * (self.y[i] / nivel_actual) + (1 - self.gamma) * estacionalidad[i - 1] # cambiamos - por / (self.y[i] - nivel_actual)
            # Agregar los valores suavizados a las listas correspondientes
            nivel.append(nivel_actual)
            tendencia.append(tendencia_actual)
            estacionalidad.append(estacionalidad_actual)

        # Guardar los valores de nivel, tendencia y estacionalidad como atributos de la clase
        self.nivel = np.array(nivel)
        self.tendencia = np.array(tendencia)
        self.estacionalidad = np.array(estacionalidad)
        print(self.nivel)
        print(self.tendencia)
        print(self.estacionalidad)


    def predecir(self, periodos: int):
        if self.nivel is None or self.tendencia is None or self.estacionalidad is None:
            raise ValueError("La regresión aún no se ha calculado.")
        # Calcular los valores de la serie suavizada para los períodos futuros
        y_suavizado = self.nivel[-1] + np.arange(1, periodos + 1) * self.tendencia[-1] + self.estacionalidad[-self.m:][:periodos].tolist()
        return pd.DataFrame(y_suavizado, index=np.arange(self.x[-1] + 1, self.x[-1] + periodos + 1))
    
datos = [3494, 3379, 3453, 3220, 3380, 3382, 3266, 3179, 3464, 3261, 3119, 3135]
modelo = SuavizacionExponencialTriple(datos, 0.86, 0.1, 0.09)
modelo.calcular_regresion()
predicciones = modelo.predecir(12)
print(predicciones)


