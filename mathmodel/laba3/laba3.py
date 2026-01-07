import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
from scipy.integrate import solve_ivp

df = pd.read_csv('crocodiles.csv')
x_name, y_name, t_name = 'X', 'Y', 'Time'
t_data = df[t_name].values
H_data = df[x_name].values
L_data = df[y_name].values

y0 = [H_data[0], L_data[0]]

# Функция решения модели
def solve_model(params, t, y0):
    r1, r2, a1, a2, b1, b2 = params

    def lotka_volterra(t, y):
        H, L = y
        dHdt = r1 * H - a2 * H * L - b1 * H * H
        dLdt = r2 * L - a1 * H * L - b2 * L * L
        return [dHdt, dLdt]

    solution = solve_ivp(lotka_volterra, [t[0], t[-1]], y0, t_eval=t, method='RK45')
    return solution.y

# Функция ошибки для подбора параметров
def objective(params):
    y_pred = solve_model(params, t_data, y0)
    error_H = np.mean((y_pred[0] - H_data)**2)
    error_L = np.mean((y_pred[1] - L_data)**2)
    return error_H + error_L

# Оптимизация параметров модели
initial_guess = np.random.random(size=6)
print("Начальное приближение:", initial_guess)

bounds = [(0.001, 5), (0.001, 5), (0.001, 2), (0.001, 2), (0.0001, 2), (0.0001, 2)]

result = minimize(objective, initial_guess, method='L-BFGS-B', bounds=bounds)
print("Найденные параметры:", result.x)
r1, r2, a1, a2, b1, b2 = result.x

# Функция модели для прогноза
def lotka_volterra_model(t, y):
    H, L = y
    dHdt = r1 * H - a1 * H * L - b1 * H * H
    dLdt = r2 * L - a2 * H * L - b2 * L * L
    return [dHdt, dLdt]

# Реализация метода Рунге-Кутта 4-го порядка
def runge_kutta_4th_order(f, y0, t_span, n_steps):
    t_start, t_end = t_span
    t = np.linspace(t_start, t_end, n_steps)
    h = (t_end - t_start) / (n_steps - 1)
    y = np.zeros((len(y0), n_steps))
    y[:, 0] = y0

    for i in range(n_steps - 1):
        k1 = h * np.array(f(t[i], y[:, i]))
        k2 = h * np.array(f(t[i] + h/2, y[:, i] + k1/2))
        k3 = h * np.array(f(t[i] + h/2, y[:, i] + k2/2))
        k4 = h * np.array(f(t[i] + h, y[:, i] + k3))
        y[:, i+1] = y[:, i] + (k1 + 2*k2 + 2*k3 + k4) / 6

    return t, y

# Настройки для прогноза
H0 = H_data[0]
L0 = L_data[0]
n_steps = 1000
y0 = [H0, L0]

forecast_period = 1.5
t_max_original = t_data[-1]
t_min_original = t_data[0]
t_range_original = t_max_original - t_min_original

t_span_forecast = (t_min_original, t_min_original + t_range_original * forecast_period)
n_steps_forecast = int(n_steps * forecast_period)

# Вычисляем прогноз
t_forecast, y_forecast = runge_kutta_4th_order(lotka_volterra_model, y0, t_span_forecast, n_steps_forecast)

historical_mask = t_forecast <= t_max_original
forecast_mask = t_forecast > t_max_original

t_historical = t_forecast[historical_mask]
t_forecast_only = t_forecast[forecast_mask]

H_historical = y_forecast[0][historical_mask]
H_forecast = y_forecast[0][forecast_mask]
L_historical = y_forecast[1][historical_mask]
L_forecast = y_forecast[1][forecast_mask]

# Визуализация: история + прогноз
plt.figure(figsize=(16, 10))

plt.plot(t_data, H_data, 'dodgerblue', label='Реальные данные: X', linewidth=2)
plt.plot(t_data, L_data, 'limegreen', label='Реальные данные: Y', linewidth=2)

plt.plot(t_historical, H_historical, 'royalblue', linestyle='--', label='Модель: X', linewidth=2)
plt.plot(t_historical, L_historical, 'forestgreen', linestyle='--', label='Модель: Y', linewidth=2)

plt.plot(t_forecast_only, H_forecast, 'royalblue', linestyle=':', alpha=0.7, label='Прогноз: X', linewidth=2)
plt.plot(t_forecast_only, L_forecast, 'forestgreen', linestyle=':', alpha=0.7, label='Прогноз: Y', linewidth=2)

plt.axvline(x=t_max_original, color='red', linestyle='--', alpha=0.7, label='Начало прогноза')

plt.xlabel('Время')
plt.ylabel('Численность популяций')
plt.legend()
plt.show()

# Фазовый портрет
plt.figure(figsize=(12, 8))

plt.plot(H_data, L_data, 'darkred', label='Реальные данные', linewidth=2)

plt.plot(H_historical, L_historical, 'red', linestyle='--', label='Модельные данные', linewidth=2)

plt.plot(H_forecast, L_forecast, 'orange', linestyle='-.', label='Прогноз', linewidth=2)

plt.plot(H_historical[-1], L_historical[-1], 'ro', markersize=8, label='Начало прогноза')

plt.xlabel('Популяция X')
plt.ylabel('Популяция Y')
plt.legend()
plt.show()
