from math import round

class InventarioPerfiles:
    def __init__(self, cantidades, medidas, tipos=False):
        '''
        parametros: cantidades:List medidas:list opcional( tipos:lista )
        accion: Creación de inventario por tipo perfil
        returns: None
        '''
        #verificación de argumento de tipos
        if tipos==False:
            self.tipos_i=['Ref' for n in cantidades]
        else:
            #verificación de tamaño de listas
            if len(cantidades)==len(tipos):
                self.tipos_i=tipos
            else:
                print('\tERROR longitud de listas de distintos tamaños')
        #verificación de tamaño de listas
        if len(cantidades)==len(medidas):
            self.cantidades_i=cantidades
            self.medidas_i=medidas
            #creación de inventario
            self.inventario=[]
            for n,cantidad in enumerate(self.cantidades_i):
                for m in range(cantidad):
                    self.inventario.append([self.medidas_i[n],self.tipos_i[n]])
        else:
            print('\tERROR longitud de listas de distintos tamaños')

    def Requerimientos(self, cantidades, medidas, tipos=False):
        '''
        parametros: cantidades:List medidas:list opcional( tipos:lista )
        accion: Creacion de requerimeintos por tipo perfil
        returns: None
        '''
        #verificación de opcion varaibel tipos
        if tipos==False:
            self.tipos_r=['Ref' for n in cantidades]
        else:
            #verificación de tamañao de listas
            if len(cantidades)==len(tipos):
                self.tipos_r=tipos
        #verificación de tamaño de listas
        if len(cantidades)==len(medidas):        
            self.cantidades_r=cantidades
            self.medidas_r=medidas
            #creación de requerimientos
            self.requerimiento=[]
            for n,cantidad in enumerate(self.cantidades_r):
                for m in range(cantidad):
                    self.requerimiento.append([self.medidas_r[n],self.tipos_r[n]])
        else:
            print('\tERROR longitud de listas de distintos tamaños')
    
    def Optimizar_2D(self, ancho_corte=0):
        '''
        parametros: opcional( ancho_corte:float )
        acccion: Organizar de amyor a menor para aplicacion de algortimo de optimizacion
        returns: Asginacion de perfiles requeridos a cada perfil en inventario segun tipo
        '''
        self.inventario=sorted(self.inventario, key=lambda x: x[0],reverse=True)
        self.requerimiento=sorted(self.requerimiento, key=lambda x: x[0],reverse=True)
        self.faltantes=[]
        self.asignacion=[]
        #Aplicar algortimo por cada tipo_r
        for tipo_r in set(self.tipos_r):
            #Separar listas correspondientes al tipo
            inventarios=list(filter(lambda x : x[1]==tipo_r , self.inventario))
            requerimientos=list(filter(lambda x : x[1]==tipo_r , self.requerimiento))
            #crear valor de residuo por perfil de inventario
            for n,perfil_i in enumerate(inventarios):
                inventarios[n].insert(0, perfil_i[0])
            perfil_f=[]
            #asignar todos los perfiles de requerimeinto del tipo
            for n,perfil_r in enumerate(requerimientos):
                for m,perfil_i in enumerate(inventarios):
                    #si es poisble asigne el requerimeinto al perfil de inventario
                    if perfil_i[1]>=perfil_r[0]:
                        #se usa round para solucionar erro de binarios de lenguaje
                        inventarios[m][1]=round((inventarios[m][1]-perfil_r[0]-ancho_corte)*10)/10
                        inventarios[m].append(perfil_r[0])
                        break
                    #en caso de no poderse agregar y sea el ultimo perfil de inventario asignarlo a faltantes
                    if m==int(len(inventarios))-1:
                        print('Nunca se puedo agregar ',perfil_r[0],' de ',tipo_r)
                        perfil_f.append(perfil_r[0])
            #ceracion de faltantes
            self.faltantes.append([tipo_r]+perfil_f)
            #ceracion de asignacion
            self.asignacion+=inventarios
        return self.asignacion

def print_straight(array):
    '''
    params: array:list()
    acccion: Imprimir de forma ordenada objetos del array
    returns: None
    '''
    for row in array:
        print(row)  
    print('\n')

'''
#Base 1
#Objeto.InventarioPerfiles
cantidades_i=[5,2,1]
medidas_i=[250.0,600.0,150.0]
#variable opcional en caso de multiples tipos
#3er argumento opcional
tipos_i=['Ref 744','Ref 8025','Ref 744']

#Objeto.Requerimeintos
cantidades_r=[7,12,2]
medidas_r=[78.5,105.0,20.0]
#variable obligatoria si incluyo tipos en la creaccio de Objeto
tipos_r=['Ref 744','Ref 8025','Ref 744']

#Objeto.Optimizar_2D
#variable opcional en caso de ancho de corte>0, default=0
ancho_corte=0.4

#creacion de modelo de optimizacion corte lineal con grosor
Objeto=InventarioPerfiles(cantidades_i,medidas_i,tipos_i)
Objeto.Requerimientos(cantidades_r,medidas_r,tipos_r)
Objeto.Optimizar_2D(ancho_corte)
print_straight(Objeto.inventario)
print_straight(Objeto.asignacion)
print_straight(Objeto.faltantes)
'''

'''
CONSULTA DE VARIABLES
Objeto.tipos_i: tipos de perfiles de invntario
Objeto.invantario: listas de inventario por tipo
Objeto.tipos_r: tipos de perfiles de requerimiento
Objeto.requerimiento: listas de requerimiento por tipo
Objeto.asignacion: listas de asignacion por perfil (en posicion 0) con sobrante en 1ra posicion
Objeto.faltantes: listas de perfiles faltantes por tipo
'''
