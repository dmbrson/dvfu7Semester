import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
from scipy.integrate import solve_ivp

# Загрузка данных
df = pd.read_csv('data/hudson-bay-lynx-hare.csv')

t_data = df['Year'].values
H_data = df[' Hare'].values
L_data = df[' Lynx'].values

y0 = [H_data[0], L_data[0]]

# Модель Лотки – Вольтерры
def model(params, t, y0):
    alpha, beta, delta, gamma = params

    def lotka_volterra(t, y):
        H, L = y
        dHdt = alpha * H - beta * H * L
        dLdt = delta * H * L - gamma * L
        return [dHdt, dLdt]

    sol = solve_ivp(
        lotka_volterra,
        (t[0], t[-1]),
        y0,
        t_eval=t,
        method='RK45'
    )

    return sol.y

# Функция ошибки
def objective(params):
    y_pred = model(params, t_data, y0)
    error_H = np.mean((y_pred[0] - H_data) ** 2)
    error_L = np.mean((y_pred[1] - L_data) ** 2)
    return error_H + error_L

# Оценка параметров
initial_guess = [1.0, 0.1, 0.01, 0.375]
bounds = [(0.001, 10), (0.001, 1), (0.001, 1), (0.001, 1)]

result = minimize(
    objective,
    initial_guess,
    bounds=bounds,
    method='L-BFGS-B'
)

alpha, beta, delta, gamma = result.x

print("Найденные параметры:")
print(f"alpha = {alpha:.4f}, beta = {beta:.4f}, delta = {delta:.4f}, gamma = {gamma:.4f}")

# Явная система для Рунге–Кутты
def lotka_volterra(t, y):
    H, L = y
    dHdt = alpha * H - beta * H * L
    dLdt = delta * H * L - gamma * L
    return [dHdt, dLdt]

# Метод Рунге–Кутты 4-го порядка
def runge_kutta_4th_order(f, y0, t_span, n_steps):
    t_start, t_end = t_span
    t = np.linspace(t_start, t_end, n_steps)
    h = (t_end - t_start) / (n_steps - 1)

    y = np.zeros((len(y0), n_steps))
    y[:, 0] = y0

    for i in range(n_steps - 1):
        k1 = h * np.array(f(t[i], y[:, i]))
        k2 = h * np.array(f(t[i] + h / 2, y[:, i] + k1 / 2))
        k3 = h * np.array(f(t[i] + h / 2, y[:, i] + k2 / 2))
        k4 = h * np.array(f(t[i] + h, y[:, i] + k3))

        y[:, i + 1] = y[:, i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return t, y

# Решение на историческом периоде
H0 = H_data[0]
L0 = L_data[0]

t_span = (t_data[0], t_data[-1])
n_steps = 1000

t_rk, y_rk = runge_kutta_4th_order(
    lotka_volterra,
    [H0, L0],
    t_span,
    n_steps
)

# Визуализация (временные ряды)
plt.figure(figsize=(15, 10))

plt.plot(df['Year'], H_data, 'dodgerblue', label='Реальные данные: зайцы')
plt.plot(df['Year'], L_data, 'limegreen', label='Реальные данные: рыси')

plt.plot(t_rk, y_rk[0], 'royalblue', linestyle='--', label='Модель: зайцы')
plt.plot(t_rk, y_rk[1], 'forestgreen', linestyle='--', label='Модель: рыси')

plt.legend()
plt.xlabel('Год')
plt.ylabel('Численность')
plt.grid(True)
plt.show()

# Фазовый портрет
plt.figure(figsize=(10, 8))

plt.plot(df[' Hare'], df[' Lynx'], 'darkred', label='Реальные данные')
plt.plot(y_rk[0], y_rk[1], 'red', linestyle='--', label='Модель')

plt.xlabel('Численность зайцев')
plt.ylabel('Численность рысей')
plt.legend()
plt.grid(True)
plt.show()

# Прогноз
H0_forecast = y_rk[0][-1]
L0_forecast = y_rk[1][-1]

last_year = t_data[-1]
forecast_years = 10

t_span_forecast = (last_year, last_year + forecast_years)
n_steps_forecast = 1000

t_forecast, y_forecast = runge_kutta_4th_order(
    lotka_volterra,
    [H0_forecast, L0_forecast],
    t_span_forecast,
    n_steps_forecast
)

# Визуализация прогноза
plt.figure(figsize=(15, 10))

plt.plot(df['Year'], H_data, 'dodgerblue', label='История: зайцы')
plt.plot(df['Year'], L_data, 'limegreen', label='История: рыси')

plt.plot(t_rk, y_rk[0], 'blue', linestyle='--', alpha=0.7)
plt.plot(t_rk, y_rk[1], 'green', linestyle='--', alpha=0.7)

plt.plot(t_forecast, y_forecast[0], 'red', label='Прогноз: зайцы')
plt.plot(t_forecast, y_forecast[1], 'orange', label='Прогноз: рыси')

plt.axvline(x=last_year, color='black', linestyle=':', label='Начало прогноза')

plt.xlabel('Год')
plt.ylabel('Численность')
plt.legend()
plt.grid(True)
plt.show()
