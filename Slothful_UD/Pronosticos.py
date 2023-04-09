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
        Argumentos:\n
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
    def __init__(self, datos: (List, Tuple, pd.DataFrame), alpha: float, beta: float, gamma: float,
                 tipo_nivel: str = 'adi', tipo_estacionalidad: str = 'adi', ciclo: int = None,
                 nivel_inicial: Optional[float] = None, tendencia_inicial: Optional[float] = None,
                 estacionalidad_inicial: Optional[float] = None):
        '''
        SuavizacionExponencialTriple(datos(df,list,tuple), alfa:float, beta:float, gamma float, ciclo: int, tipo_nivel:'add' or 'mul', tipo_estacionalidad:'add' or 'mul',
        nivel_inicial=float, tendencia_inicial:float, estacionalidad_inicial: [float])\n
        Clase para implementar un modelo de suavización exponencial triple para la realización de pronósticos.\n
        Argumentos:\n
                datos : list, tuple, pd.DataFrame
            Datos para realizar el pronóstico. Puede ser una lista, tupla o DataFrame de pandas.
        alpha : float
            Parámetro de suavización para el nivel. Debe estar entre 0 y 1.
        beta : float
            Parámetro de suavización para la tendencia. Debe estar entre 0 y 1.
        gamma : float
            Parámetro de suavización para la estacionalidad. Debe estar entre 0 y 1.
        tipo_nivel : str, optional
            Tipo de suavización para el nivel, puede ser 'adi' para aditiva o 'mul' para multiplicativa.
            El valor por defecto es 'adi'.
        tipo_estacionalidad : str, optional
            Tipo de suavización para la estacionalidad, puede ser 'adi' para aditiva o 'mul' para multiplicativa.
            El valor por defecto es 'adi'.
        ciclo : int, optional
            Cantidad de períodos que conforman un ciclo estacional. Si no se especifica, se asume que el ciclo
            es la longitud de los datos. El valor por defecto es None.
        nivel_inicial : float, optional
            Valor inicial del nivel. Si no se especifica, se calcula automáticamente como el primer valor de los datos.
            El valor por defecto es None.
        tendencia_inicial : float, optional
            Valor inicial de la tendencia. Si no se especifica, se calcula automáticamente como la diferencia entre
            el segundo y el primer valor de los datos.
            El valor por defecto es None.
        estacionalidad_inicial : float, optional
            Valor inicial de la estacionalidad. Si no se especifica, se calcula automáticamente como la media de
            los valores de los datos en el primer ciclo.
            El valor por defecto es None.
        Ejemplo:\n
        modelo = SuavizacionExponencialTriple(datos, 0.2, 0.35, 0.4, ciclo=4,
                                                tipo_nivel='mul', tipo_estacionalidad='mul', nivel_inicial=86.31,
                                                tendencia_inicial=5.47, estacionalidad_inicial=[0.76,1.03,0.88,1.33])
        modelo.calcular_regresion()
        predicciones = modelo.predecir(4)
        print(modelo.pronostico_pasado)
        print(modelo.pronostico)
        modelo.graficar()
        '''
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
        self.tipo_nivel = tipo_nivel
        self.tipo_estacionalidad = tipo_estacionalidad
        if ciclo == None:
            self.ciclo = len(datos)
        else:
            if len(datos) % ciclo == 0:
                self.ciclo = ciclo
            else:
                raise ('No es posible agrupar los datos en la cantidad de ciclos suministrado.')

    def calcular_regresion(self):
        n = len(self.y)
        # Inicializar los valores de nivel, tendencia y estacionalidad
        nivel = []
        tendencia = []
        estacionalidad = [np.mean(self.y[i::self.ciclo]) / np.mean(self.y) for i in range(self.ciclo)] if self.estacionalidad_inicial is None else self.estacionalidad_inicial.copy()
        # Calcular los valores de nivel, tendencia y estacionalidad para cada período
        for i in range(n):
            # Calcular los valores suavizados
            if i != 0:
                if self.tipo_nivel == 'adi':
                    nivel_actual = self.alpha * (self.y[i] - estacionalidad[-self.ciclo]) + (1 - self.alpha) * (nivel[-1] + tendencia[-1]) 
                elif self.tipo_nivel == 'mul':
                    nivel_actual = self.alpha * (self.y[i] / estacionalidad[-self.ciclo]) + (1 - self.alpha) * (nivel[-1] + tendencia[-1])
                else:
                    raise TypeError('El tipo de nivel no es permitido, elija entre "adi" o "mul".')    
                tendencia_actual = self.beta * (nivel_actual - nivel[-1]) + (1 - self.beta) * tendencia[-1]
            else:
                nivel_actual = self.y[0] if self.nivel_inicial is None else self.nivel_inicial
                tendencia_actual = (self.y[1] - self.y[0]) if self.tendencia_inicial is None else self.tendencia_inicial
            if self.tipo_estacionalidad == 'adi':
                estacionalidad_actual = self.gamma * (self.y[i] - nivel_actual) + (1 - self.gamma) * estacionalidad[-self.ciclo]
            elif self.tipo_estacionalidad == 'mul':
                estacionalidad_actual = self.gamma * (self.y[i] / nivel_actual) + (1 - self.gamma) * estacionalidad[-self.ciclo] 
            else:
                raise TypeError('El tipo de estacionalidad no es permitido, elija entre "adi" o "mul".')      
            # Agregar los valores suavizados a las listas correspondientes
            nivel.append(nivel_actual)
            tendencia.append(tendencia_actual)
            estacionalidad.append(estacionalidad_actual)
        # Guardar los valores de nivel, tendencia y estacionalidad como atributos de la clase
        self.nivel = np.array(nivel)
        self.tendencia = np.array(tendencia)
        self.estacionalidad = np.array(estacionalidad)
        # Calcular valor entrenado
        multiplicacion = np.multiply(np.ones(len(self.y)), self.tendencia )
        suma = np.add(multiplicacion, self.nivel)
        y_suavizado_real = np.multiply(self.estacionalidad[0:-self.ciclo],suma)
        self.pronostico_pasado = pd.DataFrame(y_suavizado_real)
        return self.pronostico_pasado

    def predecir(self, periodos: int):
        if self.nivel is None or self.tendencia is None or self.estacionalidad is None:
            raise ValueError("La regresión aún no se ha calculado.")
        # Calcular los valores de la serie suavizada para los períodos futuros
        multiplicacion = np.multiply(np.arange(1, periodos + 1), self.tendencia[-1] )
        suma = np.add(multiplicacion, self.nivel[-1])
        y_suavizado_pronostico = np.multiply(suma, self.estacionalidad[-periodos::])
        self.pronostico = pd.DataFrame(y_suavizado_pronostico, index=np.arange(self.x[-1] + 1, self.x[-1] + periodos + 1))
        self.pronostico = pd.concat([self.pronostico_pasado,self.pronostico], axis=0)
        return self.pronostico
    
    def graficar(self):
        plt.plot(self.historico.index, self.historico.iloc[:,0], label='Histórico')
        plt.plot(self.pronostico.index, self.pronostico.iloc[:,0], color='orange',
                 label='Pronostico triple con alpha,beta,gamma = '+str((self.alpha, self.beta, self.gamma)))
        plt.xlabel('Variable independiente')
        plt.ylabel('Variable dependiente')
        plt.title('Suaviazación exponencial triple alpha,beta,gamma = '+str((self.alpha, self.beta, self.gamma)))
        plt.legend()
        plt.show()
    
#datos = [77,105,89,135,100,125,115,155,120,145,135,170]
#modelo = SuavizacionExponencialTriple(datos, 0.2, 0.35, 0.4, ciclo=4, tipo_nivel='mul', tipo_estacionalidad='mul',
#                                      nivel_inicial=86.31, tendencia_inicial=5.47, estacionalidad_inicial=[0.76,1.03,0.88,1.33])
#modelo.calcular_regresion()
#predicciones = modelo.predecir(4)
#print(modelo.pronostico_pasado)
#print(modelo.pronostico)
#modelo.graficar()

class SuavizacionExponencialDoble():
    def __init__(self, datos: (List, Tuple, pd.DataFrame), alpha: float = None, beta: float = None,
                 nivel_inicial: float = None, tendencia_inicial: float = None):
        '''
        SuavizacionExponencialDoble(datos(df,list,tuple), alfa:float, beta:float, tendencia_inicial:float, nivel_inicial:float)\n
        Clase que implementa el modelo de suavización exponencial doble para predecir valores futuros de una serie temporal.
        Argumentos:\n
        datos (List, Tuple, pd.DataFrame): Serie temporal de datos.
        alpha (float): Parámetro de suavización para el nivel de la serie.
        beta (float): Parámetro de suavización para la tendencia de la serie.
        nivel_inicial (float): Valor inicial para el nivel de la serie.
        tendencia_inicial (float): Valor inicial para la tendencia de la serie.
        Ejemplo:\n
        datos = [77,105,89,135,100,125,115,155,120,145,135,170]
        modelo = SuavizacionExponencialDoble(datos, alpha=0.8, beta=0.5, nivel_inicial = 77, tendencia_inicial = 10)
        print(modelo.calcular_suavizacion_exponencial_doble())
        print(modelo.predecir(3))
        modelo.graficar()
        '''
        self.datos = datos
        # Comprobar el tipo de dato de usuario
        if isinstance(self.datos, pd.DataFrame):
            self.historico = self.datos.copy()
        elif isinstance(self.datos, (list, tuple)):
            self.historico = pd.DataFrame(self.datos)
        else:
            raise TypeError("El tipo de datos ingresado no es válido.")
        self.x = self.historico.index
        self.alpha = alpha
        self.beta = beta
        self.nivel_inicial = nivel_inicial
        self.tendencia_inicial = tendencia_inicial
        self.resultado = None
        if self.alpha is None or self.beta is None:
            raise TypeError("No se ha especificado los valores de alpha y beta.")
        elif self.alpha < 0 or self.alpha > 1 or self.beta < 0 or self.beta > 1:
            raise TypeError("Los valores de alpha y beta deben estar entre 0 y 1.")
        
    def calcular_suavizacion_exponencial_doble(self):
        # Definir periodo de inicio de pronostioc de suavización exponencial doble
        if self.nivel_inicial is None or self.tendencia_inicial is None: 
            inicio_periodo =2
        else:
            inicio_periodo =1
        # Inicializar la suavización exponencial doble
        if self.nivel_inicial is not None :
            nivel_t = self.nivel_inicial
        else:
            nivel_t = self.historico.iloc[1,0]
        if self.tendencia_inicial is not None:
            tendencia_t = self.tendencia_inicial
        else:
            tendencia_t = self.historico.iloc[1,0] - self.historico.iloc[0,0]
        # Calcular la suavización exponencial doble para el número de periodos suministrados
        if inicio_periodo == 2:
            self.nivel = [None, nivel_t]
            self.tendencia = [None, tendencia_t]
            self.valores_pronostico = [None, None]
        else:
            self.nivel = [nivel_t]
            self.tendencia = [tendencia_t]
            self.valores_pronostico = [None]
        for i in range(inicio_periodo, len(self.historico)):
            nivel_t = self.alpha * self.historico.iloc[i,0] + (1 - self.alpha) * (self.nivel[-1] + self.tendencia[-1])
            tendencia_t = self.beta * (nivel_t - self.nivel[-1]) + (1 - self.beta) * self.tendencia[-1]
            pronostico_sed = self.nivel[-1] + self.tendencia[-1]
            self.nivel.append(nivel_t)
            self.tendencia.append(tendencia_t)
            self.valores_pronostico.append(pronostico_sed)
        self.pronostico_pasado = pd.DataFrame(self.valores_pronostico, index=self.x)
        return self.pronostico_pasado
    
    def predecir(self, periodos: int = 1):
        self.periodos = periodos
        if self.nivel is None or self.tendencia is None:
            raise ValueError("La regresión aún no se ha calculado.")
        self.pronostico = self.pronostico_pasado.copy()
        print(self.pronostico)
        # Calcular los valores de la serie suavizada para los períodos futuros
        for i in range (1, periodos + 1):
            print(len(self.pronostico), self.nivel[-1] + i * self.tendencia[-1])
            self.pronostico.loc[len(self.pronostico), 0] = self.nivel[-1] + i * self.tendencia[-1]
        return self.pronostico
    
    # Grafica el pronóstico
    def graficar(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.historico.index, self.historico.iloc[:, 0], label='Histórico')
        plt.plot(self.pronostico.index, self.pronostico.iloc[:, 0], label='Pronóstico')
        plt.xlabel('Fecha')
        plt.ylabel('Valor')
        plt.title(f'Suavización exponencial doble (alpha={self.alpha}, beta={self.beta}, {self.periodos} periodos pronosticados)')
        plt.legend()
        plt.show()

#datos = [77,105,89,135,100,125,115,155,120,145,135,170]
#modelo = SuavizacionExponencialDoble(datos, alpha=0.8, beta=0.5, nivel_inicial = 77, tendencia_inicial = 10)
#print(modelo.calcular_suavizacion_exponencial_doble())
#print(modelo.predecir(3))
#modelo.graficar()

