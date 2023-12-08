# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 08:37:17 2023

@author: sst
"""

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta

# Générer des données simulées pour 4 ans
np.random.seed(42)

# 4 ans, 12 mois, 52 semaines
total_periods = 4 * 12 * 52
time_series = np.random.randint(50, 150, total_periods)

# Fonction pour visualiser la série temporelle
def plot_time_series(data, title='Série Temporelle'):
    plt.plot(data)
    plt.title(title)
    plt.show()

# Visualiser la série temporelle simulée
plot_time_series(time_series, title='Série Temporelle Simulée')

# Appliquer la transformation de Fourier (à des fins d'illustration)
fft_values, frequencies = np.fft.fft(time_series), np.fft.fftfreq(len(time_series))
plt.plot(frequencies, np.abs(fft_values))
plt.title('Spectre de Fréquence (simulé)')
plt.show()

# Modèle ARIMA
order = (1, 1, 1)  # Remplacez p, d, q par les ordres appropriés pour votre cas
model = ARIMA(time_series, order=order)
fit_model = model.fit()

# Nombre de périodes pour la prédiction
n_periods = 12 * 4  # Prédire les 12 mois de la 5e année

# Prédiction avec ARIMA
arima_forecast = fit_model.get_forecast(steps=n_periods)
arima_mean = arima_forecast.predicted_mean

# Visualiser la prédiction ARIMA
plt.plot(time_series, label='Observations')
plt.plot(np.arange(len(time_series), len(time_series) + n_periods), arima_mean, color='red', label='Prédiction ARIMA')
plt.title('Prédiction ARIMA pour la 5e année')
plt.legend()
plt.show()
