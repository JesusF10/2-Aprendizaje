import utileria as ut
import bosque_aleatorio as ba
import os
import random

# Descarga y descomprime los datos

url = "https://archive.ics.uci.edu/static/public/59/letter+recognition.zip"
archivo = "datos/letters.zip"
archivo_datos = "datos/letter-recognition.data"
atributos = ['Letter'] + [f'feature_{i}' for i in range(1, 17)]

# Descarga datos
if not os.path.exists("datos"):
    os.makedirs("datos")
if not os.path.exists(archivo):
    ut.descarga_datos(url, archivo)
    ut.descomprime_zip(archivo)

#Extrae datos y convierte a numericos
datos = ut.lee_csv(
    archivo_datos,
    atributos=atributos,
    separador=","
)

alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

for d in datos:
    d['Letter'] = alphabet.index(d['Letter'])
    for i in range(1, 17):
        d[f'feature_{i}'] = float(d[f'feature_{i}'])

# Selecciona los artributos
target = 'Letter'
atributos = [target] + [f'feature_{i}' for i in range(1, 17)]

# Selecciona un conjunto de entrenamiento y de validación
random.seed(42)
datos = datos[:500]
random.shuffle(datos)
N = int(0.8*len(datos))
datos_entrenamiento = datos[:N]
datos_validacion = datos[N:]

# Valores de los parametros
# i: numero de arboles
# j: profundidad
# k: numero de variables
params = [(i, j, k) for i in [1, 3, 5, 10, 20, 30] for j in [1, 3, 5, 10, 20, 30] for k in [None, 5, 10, 15]]

errores = []

for num_arboles, profundidad, num_variables in params:
    bosque = ba.entrena_bosque(
        datos_entrenamiento,
        target,
        atributos,
        max_profundidad=profundidad,
        num_arboles=num_arboles,
        num_atributos=num_variables
    )
    error_en_muestra = ba.evalua_bosque(bosque, datos_entrenamiento, target)
    error_en_validacion = ba.evalua_bosque(bosque, datos_validacion, target)
    errores.append( (num_arboles, profundidad, num_variables, error_en_muestra, error_en_validacion) )
    
# Muestra los errores
print('n_arboles'.center(15) + 'profundidad'.center(15) + 'n_variables'.center(15) + 'Ein'.center(15) + 'E_out'.center(15))
print('-' * 75)

for num_arboles, profundidad, num_variables, error_entrenamiento, error_validacion in errores:
    print(
        f'{num_arboles}'.center(15),
        f'{profundidad}'.center(15),
        f'{num_variables}'.center(15) 
        + f'{error_entrenamiento:.2f}'.center(15) 
        + f'{error_validacion:.2f}'.center(15)
    )
print('-' * 75 + '\n')

#print("El minimo error fue con los parametros: \t", min_error)

print(min(errores, key=lambda x: x[-1]))

