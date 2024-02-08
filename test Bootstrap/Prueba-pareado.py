# Bloque de código para instalar las bibliotecas necesarias para correr IngeoML
#pip install --upgrade "jax[cpu]"
#pip install optax
#pip install IngeoML

# Cuántos núcleos disponibles hay en el CPU
import os
from IngeoML import StatisticSamples
from sklearn.metrics import accuracy_score,recall_score
import numpy as np
import sys

print(sys.version)

n_cores = os.cpu_count()
print(f"Número de núcleos de CPU disponibles: {n_cores}")

# se obtiene la distribución muestral de la la media y la desvición estándar
# de 10 muestras bootstrap de una población de cinco datos
statistic = StatisticSamples(num_samples=10, statistic=np.mean)
empirical_distribution = np.r_[[3, 4, 5, 2, 4]]
print(statistic(empirical_distribution))
print("Muestras de statistic (mean) \n",statistic._samples)
print(statistic(empirical_distribution))
print("Muestras de statistic (mean) \n",statistic._samples)
statistic2 = StatisticSamples(num_samples=10, statistic=np.std)
print(statistic2(empirical_distribution))
print("Muestras de statistic2 (std) \n",statistic2._samples)
# en este caso se usand dos objetos-funciones "statistic" y "statistic2"


# se repite el experimento
statistic = StatisticSamples(num_samples=10, statistic=np.mean)
empirical_distribution = np.r_[[3, 4, 5, 2, 4]]
print(statistic(empirical_distribution))
print("Muestras de statistic (mean) \n",statistic._samples)
statistic2 = StatisticSamples(num_samples=10, statistic=np.mean)
print(statistic2(empirical_distribution))
print("Muestras de statistic2 (mean2) \n",statistic2._samples)
# en este caso se utilizó la misma distribución "empirical_distribution"


# se repite el experimento
statistic = StatisticSamples(num_samples=10, statistic=np.mean)
empirical_distribution = np.r_[[3, 4, 5, 2, 4]]
print(statistic(empirical_distribution))
print("Muestras de statistic (mean) \n",statistic._samples)
statistic = StatisticSamples(num_samples=10, statistic=np.mean)
print(statistic(empirical_distribution))
print("Muestras de statistic2 (mean2) \n",statistic._samples)
# en este caso se utilizó la misma distribución "empirical_distribution" 
# y el mismo nombre de funcion-objeto "statistic" y sin embargo las 
# muestras son distintas

# se repite el ejemplo con métricas de desempeño y nuevamente 
# las muestras son distintas
labels = np.r_[[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]]
pred   = np.r_[[0, 0, 1, 0, 0, 1, 1, 1, 0, 1]]
print(accuracy_score(labels, pred))
acc = StatisticSamples(num_samples=10, statistic=accuracy_score)
acc(labels, pred)
print("Muestras de acc \n",acc._samples)
print(recall_score(labels, pred,zero_division =0.0))
rec = StatisticSamples(num_samples=10, statistic=recall_score)
rec(labels, pred)
print("Muestras de recall \n",rec._samples)
