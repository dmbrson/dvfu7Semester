import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ОДУ Брюсселятора (без диффузии)
def brusselator_ode(t, u, a, b):
    x, y = u
    dxdt = a - (b + 1) * x + x**2 * y
    dydt = b * x - x**2 * y
    return [dxdt, dydt]

# Наборы параметров
params = [
    {'a': 1.0, 'b': 0.5},
    {'a': 1.0, 'b': 2.0},
    {'a': 1.0, 'b': 3.0}
]

# Фазовые портреты и временные ряды
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

for i, param in enumerate(params):
    a, b = param['a'], param['b']

    t_span = (0, 50)
    t_eval = np.linspace(0, 50, 2000)

    sol = solve_ivp(
        brusselator_ode, t_span, [0.5, 0.5],
        args=(a, b), t_eval=t_eval, method='RK45'
    )

    x, y = sol.y

    # ----- Фазовый портрет с градиентом -----
    colors = plt.cm.viridis(np.linspace(0, 1, len(x)))
    for k in range(1, len(x)):
        axes[0, i].plot(x[k-1:k+1], y[k-1:k+1], color=colors[k], lw=1.2)

    # Равновесие
    axes[0, i].scatter(a, b/a, color='red', s=50, zorder=5)
    axes[0, i].set_title(f'Фазовый портрет (a={a}, b={b})')
    axes[0, i].set_xlabel('x')
    axes[0, i].set_ylabel('y')
    axes[0, i].grid(True, alpha=0.3)

    # ----- Временные ряды -----
    axes[1, i].plot(sol.t, x, label='x(t)', lw=1.5)
    axes[1, i].plot(sol.t, y, '--', label='y(t)', lw=1.5)
    axes[1, i].set_title('Временные ряды')
    axes[1, i].set_xlabel('t')
    axes[1, i].set_ylabel('Концентрация')
    axes[1, i].legend()
    axes[1, i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Брюсселятор с диффузией=
Dx, Dy = 0.1, 0.05
L = 20.0
N = 100
dx = L / (N - 1)

# используем последний набор параметров
a, b = params[-1]['a'], params[-1]['b']

# Начальные условия с шумом
x0 = a + 0.05 * np.random.randn(N)
y0 = b/a + 0.05 * np.random.randn(N)

def brusselator_diffusion(t, state):
    x = state[:N]
    y = state[N:]

    dxdt = np.zeros_like(x)
    dydt = np.zeros_like(y)

    for i in range(1, N - 1):
        dxdt[i] = a - (b + 1)*x[i] + x[i]**2*y[i] + Dx*(x[i+1] - 2*x[i] + x[i-1]) / dx**2
        dydt[i] = b*x[i] - x[i]**2*y[i] + Dy*(y[i+1] - 2*y[i] + y[i-1]) / dx**2

    # граничные условия
    dxdt[0] = dxdt[1]
    dxdt[-1] = dxdt[-2]
    dydt[0] = dydt[1]
    dydt[-1] = dydt[-2]

    return np.concatenate([dxdt, dydt])

# Решение
initial_state = np.concatenate([x0, y0])
t_eval = np.linspace(0, 50, 250)

solution = solve_ivp(
    brusselator_diffusion, (0, 50), initial_state,
    t_eval=t_eval, method='BDF'
)

# Визуализация
x_st = solution.y[:N, :]
y_st = solution.y[N:, :]
space = np.linspace(0, L, N)
time = solution.t

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# x(r,t)
im1 = axes[0, 0].imshow(
    x_st, aspect='auto', cmap='plasma',
    extent=[time[0], time[-1], L, 0]
)
axes[0, 0].set_title('Пространственно-временная динамика x(r,t)')
axes[0, 0].set_xlabel('t')
axes[0, 0].set_ylabel('r')
plt.colorbar(im1, ax=axes[0, 0])

# y(r,t)
im2 = axes[0, 1].imshow(
    y_st, aspect='auto', cmap='inferno',
    extent=[time[0], time[-1], L, 0]
)
axes[0, 1].set_title('Пространственно-временная динамика y(r,t)')
axes[0, 1].set_xlabel('t')
axes[0, 1].set_ylabel('r')
plt.colorbar(im2, ax=axes[0, 1])

# Профили x(r)
for idx in [0, 50, 100, 200]:
    axes[1, 0].plot(space, x_st[:, idx], label=f't={time[idx]:.1f}')
axes[1, 0].set_title('Пространственные профили x')
axes[1, 0].set_xlabel('r')
axes[1, 0].set_ylabel('x')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Осцилляции в точке
point = N // 2
axes[1, 1].plot(time, x_st[point], label='x(t)')
axes[1, 1].plot(time, y_st[point], '--', label='y(t)')
axes[1, 1].set_title(f'Осцилляции при r = {space[point]:.1f}')
axes[1, 1].set_xlabel('t')
axes[1, 1].set_ylabel('Концентрация')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
