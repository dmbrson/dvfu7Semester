import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import math


class LogarithmicSpiral:
    def __init__(self, a=0.1, b=0.15):
        self.a = a
        self.b = b

    def parametric_equation(self, t):
        t = np.asarray(t)
        x = self.a * np.exp(self.b * t) * np.cos(t)
        y = self.a * np.exp(self.b * t) * np.sin(t)
        return x, y



class RationalSpline:
    def __init__(self, points, weights=None, degree=3):
        self.points = np.array(points)
        self.n_points = len(points)
        self.degree = degree

        if weights is None:
            self.weights = np.ones(self.n_points)
        else:
            self.weights = np.array(weights)

    def chord_length_parameterization(self):
        chord_lengths = np.zeros(self.n_points)
        for i in range(1, self.n_points):
            dx = self.points[i, 0] - self.points[i - 1, 0]
            dy = self.points[i, 1] - self.points[i - 1, 1]
            chord_lengths[i] = chord_lengths[i - 1] + math.sqrt(dx ** 2 + dy ** 2)

        if chord_lengths[-1] > 0:
            chord_lengths /= chord_lengths[-1]

        return chord_lengths

    def fit_spline(self, s=0):

        t = self.chord_length_parameterization()

        sort_idx = np.argsort(t)
        t_sorted = t[sort_idx]
        points_sorted = self.points[sort_idx]
        weights_sorted = self.weights[sort_idx]

        self.spline_x = interpolate.UnivariateSpline(t_sorted, points_sorted[:, 0],
                                                     w=weights_sorted, s=s, k=3)
        self.spline_y = interpolate.UnivariateSpline(t_sorted, points_sorted[:, 1],
                                                     w=weights_sorted, s=s, k=3)

        return self.spline_x, self.spline_y

    def evaluate(self, t_eval):
        if not hasattr(self, 'spline_x'):
            self.fit_spline()

        x_eval = self.spline_x(t_eval)
        y_eval = self.spline_y(t_eval)

        return x_eval, y_eval


def generate_spiral_points(a=0.1, b=0.15, n_points=400):
    spiral = LogarithmicSpiral(a, b)

    t = np.linspace(0, 6 * np.pi, n_points)
    x, y = spiral.parametric_equation(t)

    points = np.column_stack([x, y])
    return points, spiral


def calculate_spiral_error(spiral, points, spline, n_eval=2000):
    t_dense = np.linspace(0, 6 * np.pi, n_eval)
    x_exact, y_exact = spiral.parametric_equation(t_dense)

    # mask = ~np.isnan(x_exact_dense) & ~np.isnan(y_exact_dense) & \
    #        (np.abs(x_exact_dense) < 10) & (np.abs(y_exact_dense) < 10)
    # x_exact = x_exact_dense[mask]
    # y_exact = y_exact_dense[mask]

    t_eval = np.linspace(0, 1, n_eval)
    x_spline, y_spline = spline.evaluate(t_eval)

    spline_mask = ~np.isnan(x_spline) & ~np.isnan(y_spline) & \
                  (np.abs(x_spline) < 10) & (np.abs(y_spline) < 10)
    x_spline_clean = x_spline[spline_mask]
    y_spline_clean = y_spline[spline_mask]

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(x_exact, y_exact, 'k-', alpha=0.7, linewidth=2, label='Точная кривая')
    plt.plot(points[:, 0], points[:, 1], 'ro', markersize=3, alpha=0.6, label='Точки аппроксимации')
    plt.plot(x_spline_clean, y_spline_clean, 'b-', alpha=0.7, linewidth=1, label='Сплайн')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Сравнение кривых')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')

    from scipy.spatial import cKDTree

    exact_points = np.column_stack([x_exact, y_exact])
    spline_points = np.column_stack([x_spline, y_spline])

    tree = cKDTree(exact_points)
    distances, _ = tree.query(spline_points)

    plt.subplot(1, 3, 2)
    plt.hist(distances, bins=50, alpha=0.7, color='red')
    plt.xlabel('Расстояние до точной кривой')
    plt.ylabel('Частота')
    plt.title('Распределение ошибок')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.scatter(x_spline_clean, y_spline_clean, c=distances, cmap='hot', s=10)
    plt.colorbar(label='Ошибка')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Пространственное распределение ошибок')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')

    plt.tight_layout()
    plt.show()

    rmse = np.sqrt(np.mean(distances ** 2))
    max_error = np.max(distances)
    mean_error = np.mean(distances)

    print(f"Корректная среднеквадратичная ошибка: {rmse:.6f}")
    print(f"Корректная максимальная ошибка: {max_error:.6f}")
    print(f"Корректная средняя ошибка: {mean_error:.6f}")

    return rmse, max_error, mean_error


def main_spiral():
    print("Эксперименты с параметрическим рациональным сплайном (ЛОГАРИФМИЧЕСКАЯ СПИРАЛЬ)")

    points, spiral = generate_spiral_points(a=0.1, b=0.15, n_points=300)
    print(f"Параметры спирали: a={spiral.a}, b={spiral.b}")
    print(f"Количество точек: {len(points)}")

    smoothing_params = [0, 0.0001, 0.001, 0.01]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, s in enumerate(smoothing_params):
        ax = axes[i]

        spline = RationalSpline(points)
        spline.fit_spline(s=s)

        t_eval = np.linspace(0, 1, 1000)
        x_spline, y_spline = spline.evaluate(t_eval)

        t_exact = np.linspace(0, 6 * np.pi, 1500)
        x_exact, y_exact = spiral.parametric_equation(t_exact)

        ax.plot(x_exact, y_exact, 'k-', alpha=0.5, label='Точная спираль')
        ax.plot(points[:, 0], points[:, 1], 'ro', markersize=2, alpha=0.5, label='Точки')
        ax.plot(x_spline, y_spline, 'b-', label=f'Сплайн (s={s})')

        ax.set_title(f'Параметр сглаживания s={s}')
        ax.set_aspect('equal')
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.show()

    print("\n")
    print("Анализ ошибки:")

    spline_optimal = RationalSpline(points)
    spline_optimal.fit_spline(s=0.001)

    calculate_spiral_error(spiral, points, spline_optimal)
    rmse, max_err, mean_err = calculate_spiral_error(spiral, points, spline_optimal)

    print(f"\nИтоговые результаты:")
    print(f"КMSE: {rmse:.6f}")
    print(f"Максимальная ошибка: {max_err:.6f}")
    print(f"Средняя ошибка: {mean_err:.6f}")




if __name__ == "__main__":
    main_spiral()