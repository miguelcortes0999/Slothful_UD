import pathlib
from setuptools import find_packages, setup

HERE = pathlib.Path(__file__).parent

VERSION = '0.0.1' #Muy importante, deberéis ir cambiando la versión de vuestra librería según incluyáis nuevas funcionalidades
PACKAGE_NAME = 'Slothful_UD' #Debe coincidir con el nombre de la carpeta 
AUTHOR = 'Miguel Cortes - Miguel Miranda - Eduyn Lopez'
AUTHOR_EMAIL = 'miguelcortes0999@gmail.com - '
URL = 'https://github.com/miguelcortes0999/Industrial-Engineering' #Modificar con vuestros datos

LICENSE = 'GNU General Public License v3.0'
DESCRIPTION = 'Modelos heuristicos o porgrmación lineal de Ingeneria Industrial para Pronosticos, Planeacion Agregada y Plan Maestro de Producción' #Descripción corta
LONG_DESCRIPTION = (HERE / "README.md").read_text(encoding='utf-8') #Referencia al documento README con una descripción más elaborada
LONG_DESC_TYPE = "text/markdown"


#Paquetes necesarios para que funcione la libreía. Se instalarán a la vez si no lo tuvieras ya instalado
INSTALL_REQUIRES = ['pandas','pulp','gurobipy','matplotlib']

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type=LONG_DESC_TYPE,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    install_requires=INSTALL_REQUIRES,
    license=LICENSE,
    packages=find_packages(),
    include_package_data=True
)