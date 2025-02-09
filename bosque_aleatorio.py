""" 
    Funciones y clases para entrenamiento y prediccion con bosques aleatorios (mediante arboles de decision numericos).
    
    Se asume que los datos vienen en forma de una lista de diccionarios, donde cada diccionario representa una instancia (la cual puede tener nombres de atributos diferentes).
    
"""

__author__ = "Jesus Flores Lacarra"
__date__ = "febrero 2025"

import arboles_numericos as an
from collections import Counter
import random

def entrena_bosque(datos, target, clase_default, max_profundidad=1, num_arboles=3, num_atributos=None):
    """
    Funcion que entrena un bosque aleatorio a partir de un conjunto de datos.
    
    Parametros:
        datos: list(dict)
            Lista de diccionarios, donde cada diccionario representa una instancia con los valores de sus atributos.
        target: str
            El nombre del atributo que se quiere predecir
        clase_default: str
            El valor de la clase por default
        max_profundidad: int
            Profundidad maxima de los arboles de decision.
        num_arboles: int
            Numero de arboles en el bosque.
        num_atributos: int 
            Numero de atributos a considerar en cada arbol.
        
    Regresa:
        bosque: lista de arboles de decision, donde cada arbol es un diccionario con la estructura de un arbol de decision.
        
    """
    bosque = []
    random.seed(42)
    random.shuffle(datos)
    tamano_conjunto = len(datos) // num_arboles
    for i in range(num_arboles):
        datos_entrenamiento = datos[i*tamano_conjunto : (i+1)*tamano_conjunto]
        bosque.append(an.entrena_arbol(datos_entrenamiento, target, clase_default, max_profundidad=max_profundidad, variables_seleccionadas=num_atributos))
    
    return bosque

def predice_bosque(bosque, datos):
    """
    Funcion que predice la clase de una instancia a partir de un bosque aleatorio.
    
    Parametros:
        bosque: list(dict)
            Lista de diccionarios, donde cada diccionario representa un arbol de decision.
        datos: dict
            Diccionario con los valores de los atributos de una instancia.
        
    Regresa:
        clase: str
            La clase predicha para la instancia.
        
    """
    predicciones = []
    for arbol in bosque:
        predicciones.append(an.predice_arbol(arbol, datos))
        
    return [Counter(prediccion).most_common(1)[0][0] for prediccion in zip(*predicciones)]

def evalua_bosque(bosque, datos, target):
    """
    Funcion que evalua la precision de un bosque aleatorio.

    Parametros:
        bosque: list(Nodo)
            Lista de arboles de decision.
        datos: list(dict)
            Lista de diccionarios, donde cada diccionario representa una instancia con los valores de sus atributos.
        target: str
            El nombre del atributo que se quiere predecir

    Regresa:
        accuracy: float
            La precision del bosque
    """
    predicciones = predice_bosque(bosque, datos)
    return sum(1 for p, d in zip(predicciones, datos) if p == d[target]) / len(datos)