""" 
    Funciones y clases para entrenamiento y prediccion con bosques aleatorios (mediante arboles de decision numericos).
    
    Se asume que los datos vienen en forma de una lista de diccionarios, donde cada diccionario representa una instancia (la cual puede tener nombres de atributos diferentes).
    
"""

__author__ = "Jesus Flores Lacarra"
__date__ = "febrero 2025"

import arboles_numericos as an


def entrena_bosque(datos, target, clase_default, max_profundidad, num_arboles, num_atributos):
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
    # TODO: Implementar
    for _ in range(num_arboles):
        bosque.append(an.entrena_arbol(datos, target, clase_default, max_profundidad, num_atributos))
    
    
    return bosque