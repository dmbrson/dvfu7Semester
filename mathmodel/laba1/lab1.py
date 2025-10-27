import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

df = pd.read_csv('china_population.csv')
df = df.sort_values('Year').reset_index(drop=True)

population = df['Population'].values
t_min = df['Year'].min()
t_max = df['Year'].max()
t_scaled = (df['Year'] - t_min) / (t_max - t_min)

def malthus_model_scaled(t_scaled, r, x0):
    t_relative = t_scaled * (t_max - t_min)
    return x0 * np.exp(r * t_relative)

initial_guess_malthus = [0.02, population[0]]
params_malthus, _ = curve_fit(
    malthus_model_scaled,
    t_scaled,
    population,
    p0=initial_guess_malthus,
    maxfev=10000
)
r_malthus, x0_malthus = params_malthus
print(f"Модель Мальтуса: r={r_malthus:.5f}, x0={x0_malthus:.0f}")

def logistic_model_scaled(t_scaled, r, k, x0):
    t_relative = t_scaled * (t_max - t_min)
    return (k * x0 * np.exp(r * t_relative)) / (k - x0 + x0 * np.exp(r * t_relative))

initial_guess_logistic = [0.02, population.max() * 1.5, population[0]]
params_logistic, _ = curve_fit(
    logistic_model_scaled,
    t_scaled,
    population,
    p0=initial_guess_logistic,
    maxfev=10000
)
r_logistic, k_logistic, x0_logistic = params_logistic
print(f"Модель Ферхюльста: r={r_logistic:.5f}, k={k_logistic:.0f}, x0={x0_logistic:.0f}")

t_future = np.arange(df['Year'].min(), 2051)
t_scaled_future = (t_future - t_min) / (t_max - t_min)

y_malthus = malthus_model_scaled(t_scaled_future, r_malthus, x0_malthus)
y_logistic = logistic_model_scaled(t_scaled_future, r_logistic, k_logistic, x0_logistic)

plt.figure(figsize=(10, 6))
plt.plot(df['Year'], population, 'b-o', label='Реальные данные', linewidth=2)
plt.xlabel('Год')
plt.ylabel('Население')
plt.title('Население Китая (реальные данные)')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(df['Year'], population, 'ko', label='Реальные данные', markersize=4)
plt.plot(t_future, y_malthus, 'r--', label='Модель Мальтуса', linewidth=2)
plt.xlabel('Год')
plt.ylabel('Население')
plt.title('Реальные данные + Модель Мальтуса (прогноз до 2050)')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(df['Year'], population, 'ko', label='Реальные данные', markersize=4)
plt.plot(t_future, y_logistic, 'g-', label='Модель Ферхюльста', linewidth=2)
plt.xlabel('Год')
plt.ylabel('Население')
plt.title('Реальные данные + Модель Ферхюльста (прогноз до 2050)')
plt.legend()
plt.grid(True)
plt.show()



years_pred = np.arange(2025, 2036)
t_scaled_pred = (years_pred - t_min) / (t_max - t_min)
pred_malthus = malthus_model_scaled(t_scaled_pred, r_malthus, x0_malthus)
pred_logistic = logistic_model_scaled(t_scaled_pred, r_logistic, k_logistic, x0_logistic)

df_pred = pd.DataFrame({
    'Год': years_pred,
    'Прогноз Мальтуса': np.round(pred_malthus).astype(int),
    'Прогноз Ферхюльста': np.round(pred_logistic).astype(int)
})

print("\nПрогноз численности населения Китая (2025–2035):")
print(df_pred.to_string(index=False))