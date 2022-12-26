# INICIO-MODELO DE SECUENCIACION

#Improtar librerias necesarias
from os import system
from itertools import combinations
from random import choice
from pulp import *
from gurobipy import *
import matplotlib.pyplot as plt

# Creacion de clase
class Secuenciacion():
    # Inicializar variables
    def __init__(self, tareas:dict):
        '''
        tareas: diccionario de key:tareas value:diccionarios de maquinas con duraciones 
        de cada tarea respectiva en la maquina, en caso de estar repetida la maquina en
        el proceso se debe agregar el nombre de la maquinaseguido de raya al piso y el 
        numero de la priorida, se debe ingresar en el diccionario de cada tarea las 
        maquinas en el orden que van a ser procesados

        Ejemplo:\n
        {'T1' : { 'M1':5  , 'M2':7   , 'M3':7   },\n
        'T2' : { 'M1_1':3 , 'M2_1':7 , 'M1_2':2  , 'M2_2':7 },\n
        'T3' : { 'M3_1':5 , 'M2':8   , 'M3_2':9 },\n
        'T4' : { 'M1':4   , 'M2':7   , 'M3':6   },\n
        'T5' : { 'M2_1':5 , 'M1':6   , 'M2_2':7  , 'M3':2 },\n
        'T6' : { 'M1':8   , 'M3':5   , 'M2':4   },\n
        'T7' : { 'M3_1':9 , 'M2':2   , 'M1':5    , 'M3_2':5 },\n
        'T8' : { 'M2_1':3 , 'M3':4   , 'M2_2':1 }}\n
        '''
        self.modelo = LpProblem("modelo_secuanciacion", sense=LpMinimize)
        self.M = 0
        for tarea in tareas.keys():
            for maquina in tareas[tarea]:
                self.M += tareas[tarea][maquina] * 2
        self.M = int(self.M)
        self.tareas = tareas
        self.Crear_Variables()
        self.Restricciones_Solape_Tiempos()
        self.Restriccion_Secuecnia()
        self.Restricciones_Tiempo_Maximo()
        self.Funcion_Objetivo()
        self.Solucionar()

    # Crear vairbales modelo
    def Crear_Variables(self):
        self.nombresTareas = list(self.tareas.keys())
        self.nombresMaquinas = list(set([maquina.split('_')[0] for tarea in self.tareas.keys() for maquina in self.tareas[tarea]]))
        self.nombresMaquinasTareas = dict([(tarea,list(self.tareas[tarea].keys())) for tarea in self.nombresTareas])
        # Crear variables Tiempos de inicio
        self.nombresTiemposInicio = [tarea+'_'+maquina for tarea in self.nombresTareas for maquina in self.nombresMaquinasTareas[tarea]]
        self.TiemposInicio = LpVariable.dicts("TiemposInicio", self.nombresTiemposInicio , lowBound=0, cat='Continuos')
        # Crear agrupacion por Tareas
        self.diccionarioNombresSecuencia = dict((tarea,[]) for tarea in self.nombresTareas)
        for nombresTiemposInicio in self.nombresTiemposInicio:
            nombresTiemposInicioSplit = nombresTiemposInicio.split('_')
            self.diccionarioNombresSecuencia[nombresTiemposInicioSplit[0]].append(nombresTiemposInicio)
        # Crear agrupacion por Maquinas
        self.diccionarioNombresMaquinas = dict((maquina,[]) for maquina in self.nombresMaquinas)
        for maquina in self.nombresMaquinas:
            for nombresTiemposInicio in self.nombresTiemposInicio:
                if maquina in nombresTiemposInicio:
                    self.diccionarioNombresMaquinas[maquina].append(nombresTiemposInicio)
        # Crear variables Binarias de activacion
        self.nombresBinarias = []
        for maquina in self.nombresMaquinas:
            self.nombresBinarias += list(combinations(self.diccionarioNombresMaquinas[maquina],2))
        self.BinariaActivacion = LpVariable.dicts("BinariaActivacion", self.nombresBinarias , cat='Binary')
    
    # Restriccion de secuenciacion
    def Restricciones_Solape_Tiempos(self):
        for nombresTiemposInicio in self.nombresBinarias:
            nti1, nti2 = nombresTiemposInicio[0], nombresTiemposInicio[1] 
            tnti1, tnti2 = nti1.split('_',1)[0], nti2.split('_',1)[0]
            mnti1, mnti2 = nti1.split('_',1)[1], nti2.split('_',1)[1]
            dur1, dur2 = self.tareas[tnti1][mnti1], self.tareas[tnti2][mnti2]
            self.modelo += self.TiemposInicio[nti1] + dur1 <= self.TiemposInicio[nti2] + self.BinariaActivacion[nti1, nti2]*self.M
            self.modelo += self.TiemposInicio[nti2] + dur2 <= self.TiemposInicio[nti1] + (1-self.BinariaActivacion[nti1, nti2])*self.M

    # Restriccion de Tiempo Maximo
    def Restricciones_Tiempo_Maximo(self):
        self.tiempoMinimo = LpVariable("TiemposMinimo", lowBound=0, cat='Continuos')
        for tarea in self.diccionarioNombresSecuencia.keys():
            ultimaMaquina = self.diccionarioNombresSecuencia[tarea][-1]
            tnti, mnti = ultimaMaquina.split('_',1)[0], ultimaMaquina.split('_',1)[1] 
            self.modelo += self.TiemposInicio[ultimaMaquina] + self.tareas[tnti][mnti] <= self.tiempoMinimo
    
    # Restriccion de Secuencia
    def Restriccion_Secuecnia(self):
        for tarea in self.diccionarioNombresSecuencia.keys():
            for n in range(len(self.diccionarioNombresSecuencia[tarea])-1):
                nti1, nti2 = self.diccionarioNombresSecuencia[tarea][n], self.diccionarioNombresSecuencia[tarea][n+1]
                tnti1 = nti1.split('_',1)[0]
                mnti1 = nti1.split('_',1)[1]
                self.modelo += self.TiemposInicio[nti1] + self.tareas[tnti1][mnti1] <= self.TiemposInicio[nti2]
    
    # Declaracion de Funcion objetivo
    def Funcion_Objetivo(self):
        self.modelo += self.tiempoMinimo

    # Solucionar modelo
    def Solucionar(self):
        self.status = self.modelo.solve(solver=GUROBI(msg = False))
        if 'Optimal'== LpStatus[self.modelo.status]:
            print('-'*5+' Modelo solucionado correctamente '+'-'*5)
            self.horizonteTemporal = round(value(self.modelo.objective),0)
        else:
            raise 'Porblema en factibilidad del modelo'

    # Diccionario de tiempos de inicio
    def diccionarioTiemposInicio(self):
        self.tiempos = {}
        for v in self.modelo.variables():
            if 'TiemposInicio' in str(v):
                nombre = str(v)
                nombre = nombre.replace('TiemposInicio_','')
                self.tiempos[nombre] = round(v.varValue,0)
        return self.tiempos

    # Generar Diagrama de Gantt
    def diagramaGantt(self):
        fig, ax = plt.subplots(1)
        plt.title('Diagrama de Gantt')
        plt.xlabel('Tiempos de inicio')
        plt.ylabel('Maquinas')
        for tareas in self.nombresTareas:
            inicios = []
            maquinas = []
            duraciones = []
            for nombreInicio in self.tiempos.keys():
                if tareas in nombreInicio:
                    inicios.append(self.tiempos[nombreInicio])
                    tar, maq = nombreInicio.split('_',1)[0], nombreInicio.split('_',1)[1] 
                    duraciones.append(self.tareas[tar][maq])
                    maquinas.append(maq.split('_')[0])
            ax.barh(maquinas, duraciones, left=inicios, label=tareas)
        plt.legend()
        plt.show()

# FIN-MODELO DE SECUENCIACION

