import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Функция для стационарных точек
def stabile(r, b):
    sq = np.sqrt(b * (r - 1))
    return (-sq, -sq, r - 1), (sq, sq, r - 1)

# Параметры модели Лоренца
sigma = 10.0
r = 17.0
b = 8.0 / 3.0

# Система Лоренца
def lorenz_system(t, state, sigma, r, b):
    x, y, z = state
    dx_dt = sigma * (y - x)
    dy_dt = x * (r - z) - y
    dz_dt = x * y - b * z
    return [dx_dt, dy_dt, dz_dt]

# Начальные условия и временной интервал
initial_state = [1.0, 1.0, 1.0]
t_span = (0, 100)
t_eval = np.linspace(0, 100, 10000)

# Решение системы
solution = solve_ivp(
    lorenz_system, t_span, initial_state,
    args=(sigma, r, b), t_eval=t_eval, method='RK45'
)
x, y, z = solution.y

# 3D график фазовой траектории с градиентом времени
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Создаём градиент цвета по времени
colors = plt.cm.viridis(np.linspace(0, 1, len(x)))
for i in range(1, len(x)):
    ax.plot(x[i-1:i+1], y[i-1:i+1], z[i-1:i+1], color=colors[i], lw=1.2)

# Начальные условия
ax.scatter(*initial_state, color='red', s=60, label='Начальные условия')

# Стационарные точки
u1, u2 = stabile(r, b)
ax.scatter(*u1, color='orange', s=50, label='Отрицательная стационарная точка')
ax.scatter(*u2, color='purple', s=50, label='Положительная стационарная точка')

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title(f"Фазовая траектория системы Лоренца, r = {r}")
ax.grid(True)
ax.legend()
plt.show()

# Эффект бабочки: две близкие траектории 3D
initial_state1 = [1.0, 1.0, 1.0]
initial_state2 = [1.001, 1.0, 1.0]  # небольшое смещение
t_eval_short = np.linspace(0, 50, 10000)

sol1 = solve_ivp(lorenz_system, t_span, initial_state1,
                 args=(sigma, r, b), t_eval=t_eval_short, method='RK45')
sol2 = solve_ivp(lorenz_system, t_span, initial_state2,
                 args=(sigma, r, b), t_eval=t_eval_short, method='RK45')

x1, y1, z1 = sol1.y
x2, y2, z2 = sol2.y

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Градиент для первой траектории
colors1 = plt.cm.plasma(np.linspace(0, 1, len(x1)))
colors2 = plt.cm.inferno(np.linspace(0, 1, len(x2)))

for i in range(1, len(x1)):
    ax.plot(x1[i-1:i+1], y1[i-1:i+1], z1[i-1:i+1], color=colors1[i], lw=1, alpha=0.8)
    ax.plot(x2[i-1:i+1], y2[i-1:i+1], z2[i-1:i+1], color=colors2[i], lw=1, alpha=0.8)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Эффект бабочки: близкие начальные условия")
ax.grid(True)
plt.show()

# 2D график X(t) для двух близких траекторий
plt.figure(figsize=(12, 5))
plt.plot(sol1.t, x1, color='blue', lw=1.5, label='X₁(t), начальные: (1,1,1)')
plt.plot(sol2.t, x2, color='orange', lw=1.5, label='X₂(t), начальные: (1.001,1,1)')
plt.xlabel('Время t')
plt.ylabel('X(t)')
plt.title('Сравнение двух близких траекторий (эффект бабочки)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
