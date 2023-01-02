#Improtar librerias necesarias
import matplotlib.pyplot as plt
from itertools import combinations
from itertools import permutations
from pulp import *
from gurobipy import *

# Creacion de clase Modelo Secuenciacion Programacion Lineal
class SecuenciacionProgramacionLineal():
    # Inicializar variables
    def __init__(self, tareas:dict):
        '''
        SecuenciacionProgramacionLineal(tareas: dict[str ,dict[float, ..., float]])\n
        tareas: diccionario de key:tareas value:diccionarios de maquinas con duraciones 
        de cada tarea respectiva en la maquina, en caso de estar repetida la maquina en
        el proceso se debe agregar el nombre de la maquinaseguido de raya al piso y el 
        numero de la priorida, se debe ingresar en el diccionario de cada tarea las 
        maquinas en el orden que van a ser procesados
        Ejemplo:\n
        tareas={'T1' : { 'M1':5  , 'M2':7   , 'M3':7   },\n
                'T2' : { 'M1_1':3 , 'M2_1':7 , 'M1_2':2  , 'M2_2':7 },\n
                'T3' : { 'M3_1':5 , 'M2':8   , 'M3_2':9 },\n
                'T4' : { 'M1':4   , 'M2':7   , 'M3':6   },\n
                'T5' : { 'M2_1':5 , 'M1':6   , 'M2_2':7  , 'M3':2 },\n
                'T6' : { 'M1':8   , 'M3':5   , 'M2':4   },\n
                'T7' : { 'M3_1':9 , 'M2':2   , 'M1':5    , 'M3_2':5 },\n
                'T8' : { 'M2_1':3 , 'M3':4   , 'M2_2':1 }}\n
        '''
        self.modelo = LpProblem("modelo_secuanciacion_maquinas", sense=LpMinimize)
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
        self.TiempoMinimo = LpVariable("TiemposMinimo", lowBound=0, cat='Continuos')
        for tarea in self.diccionarioNombresSecuencia.keys():
            ultimaMaquina = self.diccionarioNombresSecuencia[tarea][-1]
            tnti, mnti = ultimaMaquina.split('_',1)[0], ultimaMaquina.split('_',1)[1] 
            self.modelo += self.TiemposInicio[ultimaMaquina] + self.tareas[tnti][mnti] <= self.TiempoMinimo
    
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
        self.modelo += self.TiempoMinimo

    # Solucionar modelo
    def Solucionar(self):
        self.status = self.modelo.solve(solver=GUROBI(msg = False))
        if 'Optimal'== LpStatus[self.modelo.status]:
            print('-'*5+' Modelo solucionado correctamente '+'-'*5)
            self.horizonteTemporal = round(value(self.modelo.objective),0)
        else:
            raise 'Porblema en factibilidad del modelo'

    # Diccionario de tiempos de inicio
    def Diccionario_TiemposInicio(self):
        self.tiempos = {}
        for v in self.modelo.variables():
            if 'TiemposInicio' in str(v):
                nombre = str(v)
                nombre = nombre.replace('TiemposInicio_','')
                self.tiempos[nombre] = round(v.varValue,0)
        return self.tiempos

    # Generar Diagrama de Gantt
    def Diagrama_Gantt(self):
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
        plt.legend(bbox_to_anchor=(1.02, 1.0), loc='upper left')
        plt.show()

# Creacion de clase Balnceo Linea Progrmacion Lineal
class BalanceoLienaProgramacionLineal():
    def __init__(self, tareas, produccionDiaraDeseada, produccionDiaraActual):
        '''
        BalanceoLienaProgramacionLineal(tareas: dict[str, list[float, str]], produccionDiaraDeseada: int, produccionDiaraActual: int)\n
        Enviar diccionario con nombre de cada tarea como llave, y cada valor debe ser una lista en la cual el primer valor debe ser
        un valor tipo int o float que representara el tiempo en SEGUNDOS de la tarea, y en la siguiente posicion de la lista un str
        con el nombre del predecesor, en caso de ser mas de un predecesor separarlos con el simbolo '-', es decir es un simbolo reservado 
        para su uso, y evitar el uso usar espacios
        Para el segundo argumento se debe enviar la capacidad de unidades que desea realizar en un dia
        Para el tercer argumento se debe enviar la capacidad de unidades que puede realizar en un dia
        Ejemplo:\n
        tareas={'A' : [12   , '-'],\n
                'B' : [24   , '-'],\n
                'C' : [42   , 'A'] ,\n
                'D' : [6    , 'A-B'],\n
                'E' : [18   , 'B'],\n
                'F' : [6.6  , 'C'],\n
                'G' : [19.2 , 'C'],\n
                'H' : [36   , 'C-D'],\n
                'I' : [16.2 , 'F-G-H'],\n
                'J' : [22.8 , 'E-H'],\n
                'K' : [30   , 'I-J'] }\n
        produccionDiaraDeseada =  500\n
        tiempoFuncionamientoDiario = 500\n
        '''
        self.modelo = LpProblem("modelo_balanceo_linea", sense=LpMinimize)
        self.tareas = tareas
        self.unidadesDeseadas = produccionDiaraDeseada
        self.unidadesActual = produccionDiaraActual
        self.Calcular_TakeTime()
        self.Estaciones_Minimas()
        self.Crear_Variables()
        self.Restriccion_Predecesores()
        self.Restriccion_Activaciones()
        self.Funcion_Objetivo()
        self.Solucionar()
    
    # Calcular Take time
    def Calcular_TakeTime(self):
        self.takeTime = (self.unidadesActual*60)/self.unidadesDeseadas
        for tarea in self.tareas.keys():
            if self.tareas[tarea][0] >= self.takeTime:
                self.takeTime = self.tareas[tarea][0]
    
    # Calcular minimo de estaciones sugeridas, mas un margen
    def Estaciones_Minimas(self):
        self.estaciones = sum([self.tareas[tarea][0] for tarea in self.tareas.keys()])/self.takeTime
        self.estaciones = int((self.estaciones//1)+1)+1 # + 1 Extra para descartar infactibilidad del modelo
        print(self.estaciones)
    
    # Crear vairbales modelo
    def Crear_Variables(self):
        self.BinariaEstacion = LpVariable.dicts("BinariaEstacion", (estacion for estacion in range(self.estaciones)) , cat='Binary')
        self.BinariaTareaEstacion = LpVariable.dicts("BinariaTareaEstacion", ((tarea,estacion) for estacion in range(self.estaciones) for tarea in self.tareas.keys()) , cat='Binary')

    # Restriccion de predecesores
    def Restriccion_Predecesores(self):
        for tarea in self.tareas.keys():
            if self.tareas[tarea][1] != '-':
                predecesores = self.tareas[tarea][1].split('-')
                for estacion in range(self.estaciones):
                    self.modelo += lpSum(self.BinariaTareaEstacion[predecesor,estacionacu] for predecesor in predecesores for estacionacu in range(0,estacion+1)) >= self.BinariaTareaEstacion[tarea,estacion]*len(predecesores) 

    # Restriccion Activaciones de estaciones y tareas por estacion
    def Restriccion_Activaciones(self):
        for estacion in range(self.estaciones):
            self.modelo += lpSum(self.BinariaTareaEstacion[tarea,estacion]*self.tareas[tarea][0] for tarea in self.tareas.keys()) <= self.takeTime*self.BinariaEstacion[estacion]
        for tarea in self.tareas.keys():
            self.modelo += lpSum(self.BinariaTareaEstacion[tarea,estacion] for estacion in range(self.estaciones)) == 1

    # Declaracion de Funcion objetivo
    def Funcion_Objetivo(self):
        self.modelo += lpSum(self.BinariaEstacion[estacion] for estacion in range(self.estaciones))

    # Solucionar modelo multiobjetivo
    def Solucionar(self):
        # Modelo Uso minimo de estaciones
        self.status = self.modelo.solve(solver=GUROBI(msg = False))
        if 'Optimal'== LpStatus[self.modelo.status]:
            self.horizonteTemporal = round(value(self.modelo.objective),0)
        else:
            raise 'Porblema en factibilidad del modelo'
        estaciones = 0
        # Asignacion de Restriccion Minima de Estaciones
        for v in self.modelo.variables():
            if 'BinariaEstacion' in str(v):
                estaciones += v.varValue
        self.MaximoTiempoEstacion = LpVariable('MaximoTiempoEstacion', lowBound=0, cat='Continuous')
        for estacion in range(self.estaciones):
            self.modelo += lpSum(self.BinariaTareaEstacion[tarea,estacion]*self.tareas[tarea][0] for tarea in self.tareas.keys()) <= self.MaximoTiempoEstacion
        self.modelo += lpSum(self.BinariaEstacion[estacion] for estacion in range(self.estaciones)) == estaciones 
        self.modelo += (1/sum([self.tareas[tarea][0] for tarea in self.tareas.keys()]))*(estaciones*self.MaximoTiempoEstacion)
        # Asignacion Maximizar Eficiencia de Linea en modelo
        self.status = self.modelo.solve(solver=GUROBI(msg = False))
        if 'Optimal'== LpStatus[self.modelo.status]:
            print('-'*5+' Modelo solucionado correctamente '+'-'*5)
            self.horizonteTemporal = round(value(self.modelo.objective),0)
        else:
            raise 'Porblema en factibilidad del modelo'
    
    # Diccionario tareas por estacion
    def Diccionario_Estaciones(self):
        self.ActivacionEstacion = {}
        for v in self.modelo.variables():
            if 'BinariaTareaEstacion' in str(v) and v.varValue>0:
                nombre = str(v)
                nombre = nombre.replace('BinariaTareaEstacion_','')
                nombre = nombre.replace("(",'')
                nombre = nombre.replace(")",'')
                nombre = nombre.replace("_",'')
                nombre = nombre.replace("'",'')
                nombre = nombre.split(',')
                self.ActivacionEstacion[nombre[0]] = 'Estacion '+ str(int(nombre[1])+1)
        return self.ActivacionEstacion

# Creacion de clase Secuenciacion Regla de Jhonson
class SecuenciacionReglaJhonson():
    '''
    SecuenciacionReglaJhonson(tareas: dict[str ,dict[str:float, str:float, str:float]])\n
    Aplicacion de la regla de Jonson, para esto se debe enviar un diccioanrio de diccioanrios
    de la siguiente forma, tareas: diccionario de key:tareas value:diccionarios de maquinas con duraciones 
    de cada tarea respectiva en la maquina, en este caso no todas las tareas deben ser procesadas en el mismo
    orden en a traves de cada una de las maquinas, esta heuristica acepta menor o igual a 3 maquinas
    Ejemplo:\n
    tareas={'T1' :{'M1':3,'M2':7,'M3':3},\n
            'T2' :{'M1':1,'M2':4,'M3':9},\n
            'T3' :{'M1':7,'M2':6,'M3':3},\n
            'T4' :{'M1':2,'M2':3,'M3':1},\n
            'T5' :{'M1':3,'M2':2,'M3':4},\n
            'T6' :{'M1':1,'M2':8,'M3':7},\n
            'T7' :{'M1':9,'M2':1,'M3':8},\n
            'T8' :{'M1':1,'M2':5,'M3':8},\n
            'T9' :{'M1':8,'M2':2,'M3':9},\n
            'T10':{'M1':6,'M2':1,'M3':7}}
    '''
    def __init__(self, tareas):
        self.tareasBase = tareas
        if len(tareas[list(tareas.keys())[0]])==2:
            self.tareas = tareas
        elif len(tareas[list(tareas.keys())[0]])==3:
            self.tareas={}
            maquinas = list(tareas[list(tareas.keys())[0]].keys())
            for tarea in tareas.keys():
                self.tareas[tarea] = {maquinas[0]+'-'+maquinas[1]:tareas[tarea][maquinas[0]]+tareas[tarea][maquinas[1]],
                maquinas[1]+'-'+maquinas[2]:tareas[tarea][maquinas[1]]+tareas[tarea][maquinas[2]]}
        else:
            raise 'El numero de tareas excede las 3 posibles que soluciona regla de Jhonson'
        self.nombresTareas = list(self.tareas.keys())
        self.nombresMaquinas = list(list(self.tareas.values())[0].keys())
        print(self.tareas)
        self.EncontrarCombinaciones()
        self.CalcularPosibilidades()
        self.CalcularSecuencias()
        self.CalcularTiemposSecuencias()

    # Encontrar combinaciones posibles segun regla de Jhonson
    def EncontrarCombinaciones(self):
        self.Combinaciones = []
        while self.tareas != {} :
            self.maximo = list(list(self.tareas.values())[0].values())[0]
            for tarea in self.tareas.keys():
                for maquina in self.nombresMaquinas:
                    if self.tareas[tarea][maquina] < self.maximo:
                        self.maximo = self.tareas[tarea][maquina]
            asignacion = []
            for tarea in self.tareas.keys():
                if self.tareas[tarea][self.nombresMaquinas[0]] == self.maximo:
                    asignacion.append([tarea,'I'])
                elif self.tareas[tarea][self.nombresMaquinas[1]] == self.maximo:
                    asignacion.append([tarea,'F'])
            tareas = list(set([tarea[0] for tarea in asignacion]))
            for tarea in tareas:
                self.tareas.pop(tarea)
            self.Combinaciones.append(asignacion)
        return self.Combinaciones

    # Calcular posibles combinaciones segun orden calculado
    def CalcularPosibilidades(self):
        self.SecuenciasPosibles = [[]]
        for combinacion in self.Combinaciones:
            permutaciones = list(permutations(combinacion,len(combinacion)))
            for i in range(len(permutaciones)):
                permutaciones[i] = list(permutaciones[i])
            aux=[]
            for secuancia in self.SecuenciasPosibles:
                for posibilidad in permutaciones:
                    aux.append(secuancia+posibilidad)
            self.SecuenciasPosibles = aux
        return self.SecuenciasPosibles 

    # Calcular cada una de las seceuncias a partir de combinaciones de posibilidades
    def CalcularSecuencias(self):
        self.Secuencias = []
        for secuencia in self.SecuenciasPosibles:
            inicio = []
            fin = []
            for tarea in secuencia:
                if tarea[1]=='F':
                    fin.insert(0, tarea[0])
                else:
                    inicio.append(tarea[0])
            self.Secuencias.append(inicio+fin)
        return self.Secuencias

    # Calcular tiempo de cada combinacion de posibilidad
    def CalcularTiemposSecuencias(self):
        self.TiemposProcesosSecuencias = []
        for secuencia in self.Secuencias:
            self.TiemposProcesosSecuencias.append(self.CalcularTiempoProceso(secuencia))
        return self.TiemposProcesosSecuencias   

    # Calcular tiempo de proceso para cada secuencia
    def CalcularTiempoProceso(self, secuencia):
        duraciones = []
        for tarea in secuencia:
            duraciones.append([j for j in self.tareasBase[tarea].values()])
        matriz = [ [0 for j in i] for i in self.tareasBase.values()]
        for i in range(len(matriz)):
            for j in range(len(matriz[i])):
                if i==0 and j==0:
                    matriz[i][j] = duraciones[i][j]
                elif i==0:
                    matriz[i][j] = matriz[i][j-1] + duraciones[i][j]
                elif j==0:
                    matriz[i][j] = matriz[i-1][j] + duraciones[i][j]
                else:
                    matriz[i][j] = max([matriz[i][j-1],matriz[i-1][j]]) + duraciones[i][j]
        return matriz[i][j]

SecuenciacionReglaJhonson()

# Creacion de clase Secuenciacion Regla CDS
class SecuenciacionReglaCDS():
    def __init__(self, tareas):
        '''
        SecuenciacionReglaCDS(tareas: dict[str ,dict[str:float, ..., str:float]])\n
        Aplicacion de la regla de CDS, para esto se debe enviar un diccioanrio de diccioanrios
        de la siguiente forma, tareas: diccionario de key:tareas value:diccionarios de maquinas con duraciones 
        de cada tarea respectiva en la maquina, en este caso no todas las tareas deben ser procesadas en el mismo
        orden en a traves de cada una de las maquinas, esta heuristica acepta igual o mayor a 3 maquinas,
        solo que aplicara la misma regla de Jhonson
        Ejemplo:\n
        tareas={'T1' :{'M1':3,'M2':7,'M3':3,'M4':3},\n
                'T2' :{'M1':1,'M2':4,'M3':9,'M4':9},\n
                'T3' :{'M1':7,'M2':6,'M3':3,'M4':1},\n
                'T4' :{'M1':2,'M2':3,'M3':1,'M4':7},\n
                'T5' :{'M1':3,'M2':2,'M3':4,'M4':2},\n
                'T6' :{'M1':1,'M2':8,'M3':7,'M4':9},\n
                'T7' :{'M1':9,'M2':1,'M3':8,'M4':4},\n
                'T8' :{'M1':1,'M2':5,'M3':8,'M4':1},\n
                'T9' :{'M1':8,'M2':2,'M3':9,'M4':4},\n
                'T10':{'M1':6,'M2':1,'M3':7,'M4':4}}
        '''
        self.tareasBase = tareas
        self.tareas = tareas
        self.nombreTareas = list(tareas.keys())
        self.nombreMaquinas = list(list(tareas.values())[0].keys())
        self.diccioanrioTareas = []
        self.Combinaciones = []
        self.SecuenciasPosibles = []
        self.Secuencias = []
        self.TiemposProcesosSecuencias = []
        for i in range(1,len(self.nombreMaquinas)):
            maquinaFicticia1 = self.nombreMaquinas[0:i]
            maquinaFicticia2 = self.nombreMaquinas[-i::]
            tareasAuxiliar={}
            for tarea in self.nombreTareas:
                tareasAuxiliar[tarea] = {
                    "-".join(maquinaFicticia1):sum([self.tareas[tarea][maquina] for maquina in maquinaFicticia1]),
                    "-".join(maquinaFicticia2):sum([self.tareas[tarea][maquina] for maquina in maquinaFicticia2]),}
            self.diccioanrioTareas.append(tareasAuxiliar)
            combinaciones = self.EncontrarCombinaciones(tareasAuxiliar)
            self.Combinaciones.append(combinaciones)
            posibilidades = self.CalcularPosibilidades(combinaciones)
            self.SecuenciasPosibles.append(posibilidades)
            secuencias = self.CalcularSecuencias(posibilidades) 
            self.Secuencias.append(secuencias)
            tiempos = self.CalcularTiemposSecuencias(secuencias)
            self.TiemposProcesosSecuencias.append(tiempos)

    # Encontrar combinaciones posibles segun regla de Jhonson
    def EncontrarCombinaciones(self, tareasAuxiliar):
        Combinaciones = []
        while tareasAuxiliar!= {} :
            nombresMaquinas = list(list(tareasAuxiliar.values())[0].keys())
            maximo = list(list(tareasAuxiliar.values())[0].values())[0]
            for tarea in tareasAuxiliar.keys():
                for maquina in nombresMaquinas:
                    if tareasAuxiliar[tarea][maquina] < maximo:
                        maximo = tareasAuxiliar[tarea][maquina]
            asignacion = []
            for tarea in tareasAuxiliar.keys():
                if tareasAuxiliar[tarea][nombresMaquinas[0]] == maximo:
                    asignacion.append([tarea,'I'])
                elif tareasAuxiliar[tarea][nombresMaquinas[1]] == maximo:
                    asignacion.append([tarea,'F'])
            tareas = list(set([tarea[0] for tarea in asignacion]))
            for tarea in tareas:
                tareasAuxiliar.pop(tarea)
            Combinaciones.append(asignacion)
        return Combinaciones

    # Calcular posibles combinaciones segun orden calculado
    def CalcularPosibilidades(self, combinaciones):
        SecuenciasPosibles = [[]]
        for combinacion in combinaciones:
            permutaciones = list(permutations(combinacion,len(combinacion)))
            for i in range(len(permutaciones)):
                permutaciones[i] = list(permutaciones[i])
            auxiliar=[]
            for secuancia in SecuenciasPosibles:
                for posibilidad in permutaciones:
                    auxiliar.append(secuancia+posibilidad)
            SecuenciasPosibles = auxiliar
        return SecuenciasPosibles

    # Calcular cada una de las seceuncias a partir de combinaciones de posibilidades
    def CalcularSecuencias(self, SecuenciasPosibles):
        Secuencias = []
        for secuencia in SecuenciasPosibles:
            inicio = []
            fin = []
            for tarea in secuencia:
                if tarea[1]=='F':
                    fin.insert(0, tarea[0])
                else:
                    inicio.append(tarea[0])
            Secuencias.append(inicio+fin)
        return Secuencias
    
    # Calcular tiempo de cada combinacion de posibilidad
    def CalcularTiemposSecuencias(self, Secuencias):
        TiemposProcesosSecuencias = []
        for secuencia in Secuencias:
            TiemposProcesosSecuencias.append(self.CalcularTiempoProceso(secuencia))
        return TiemposProcesosSecuencias

    # Calcular tiempo de proceso para cada secuencia
    def CalcularTiempoProceso(self, secuencia):
        duraciones = []
        for tarea in secuencia:
            duraciones.append([j for j in self.tareasBase[tarea].values()])
        matriz = [ [0 for j in i] for i in self.tareasBase.values()]
        for i in range(len(matriz)):
            for j in range(len(matriz[i])):
                if i==0 and j==0:
                    matriz[i][j] = duraciones[i][j]
                elif i==0:
                    matriz[i][j] = matriz[i][j-1] + duraciones[i][j]
                elif j==0:
                    matriz[i][j] = matriz[i-1][j] + duraciones[i][j]
                else:
                    matriz[i][j] = max([matriz[i][j-1],matriz[i-1][j]]) + duraciones[i][j]
        return matriz[i][j]   