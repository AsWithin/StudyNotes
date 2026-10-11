"""可导入 Notebook 的交互式过拟合案例。

首次在 Notebook 安装：%pip install ipympl
重启内核后，在绘图前运行：%matplotlib widget

    from Orthers.Overfitting import overfit_example
    ofit = overfit_example(False)  # False=分类，True=回归

选 Degree 后点击 fit data；点击绘图区添加样本，Shift+点击添加类别 0。
保留 ofit 引用以保持控件响应。直接运行本文件也可打开桌面交互窗口。
仅 import 不会生成数据、训练模型或弹出窗口。
"""

import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.widgets import Button, RadioButtons

__all__ = ["OverfitExample", "overfit_example"]


def _sigmoid(z):
    """数值稳定的 sigmoid，避免指数溢出。"""
    return np.exp(-np.logaddexp(0, -np.asarray(z)))


def _features(X, degree):
    """含常数列的多项式：1, x0, x1, x0², x0*x1, x1², ...。"""
    X = np.asarray(X, dtype=float)
    if X.shape[1] == 1:
        return np.polynomial.polynomial.polyvander(X[:, 0], degree)
    return np.column_stack([
        X[:, 0] ** (total - j) * X[:, 1] ** j
        for total in range(degree + 1) for j in range(total + 1)
    ])


def _fit_logistic(A, y):
    """牛顿法 + 回溯线搜索；不依赖 sklearn/scipy。

    使用极弱 L2（1e-6，不惩罚截距），避免可分数据使系数趋向无穷。
    它用于数值稳定，仍允许高阶模型明显过拟合。展示的交叉熵不含此项。
    """
    weights = np.zeros(A.shape[1])
    penalty = np.full(A.shape[1], 1e-6)
    penalty[0] = 0

    def objective(w):
        z = A @ w
        return np.mean(np.logaddexp(0, z) - y * z) + 0.5 * np.dot(penalty, w * w)

    for _ in range(100):
        p = _sigmoid(A @ weights)
        gradient = A.T @ (p - y) / len(y) + penalty * weights
        if np.max(np.abs(gradient)) < 1e-8:
            break
        hessian = (A.T * (p * (1 - p))) @ A / len(y)
        hessian += np.diag(penalty + 1e-10)
        step = np.linalg.solve(hessian, gradient)
        current = objective(weights)
        rate = 1.0
        for _ in range(30):
            candidate = weights - rate * step
            if objective(candidate) <= current - 1e-4 * rate * np.dot(gradient, step):
                weights = candidate
                break
            rate *= 0.5
        else:
            raise RuntimeError("逻辑回归未找到下降方向，请重置数据后重试。")
    return weights


class OverfitExample:
    """回归/分类实验，degree 为 1~6，seed 固定时实验可复现。

    regression=False 与截图一致，首先展示分类数据且等待点击拟合。
    show=False 适合生成静态图片或测试。
    属性 X、y 为当前训练数据；metrics 为最近拟合的 train/test 误差。
    测试数据独立生成，不用于训练；反复参考它选阶数时，应视作验证集。
    """

    def __init__(self, regression=False, *, degree=1, seed=7, show=True):
        self._check_degree(degree)
        self.degree = int(degree)
        self.mode = "Regression" if regression else "Categorical"
        self._data = self._make_data(seed)
        self._original = {
            mode: (X.copy(), y.copy()) for mode, (X, y, _, _) in self._data.items()
        }
        self.weights = None
        self.metrics = {}
        self._drawn_mode = None
        self._drawn_weights = None
        self._contours = []
        self._grid_features = None
        self._plot_x = np.linspace(-1, 1, 400)
        self._grid_x, self._grid_y = np.meshgrid(
            np.linspace(-1, 1, 160), np.linspace(-1, 1, 160)
        )
        with plt.ioff():
            self._build_ui(regression)
        if show:
            if "inline" in matplotlib.get_backend().lower():
                warnings.warn(
                    "当前为静态绘图后端。交互前请安装 ipympl，重启内核并运行 "
                    "%matplotlib widget，然后重新创建 ofit。",
                    UserWarning, stacklevel=2,
                )
            plt.show()

    def _build_ui(self, regression):
        self.fig = plt.figure(figsize=(9, 7))
        self.ax = self.fig.add_axes([0.11, 0.45, 0.84, 0.45])
        self.status = self.fig.text(0.11, 0.375, "", fontsize=10, va="top")
        self.fig.text(
            0.11, 0.025,
            "Try degrees 1, 2 and 6 on the SAME data. Lower test error is better.\n"
            "Dotted orange: ideal rule. Green: fitted model. Reset restores this mode.",
            fontsize=9, color="dimgray",
        )
        degree_ax = self.fig.add_axes([0.11, 0.09, 0.12, 0.17])
        degree_ax.set_title("Degree", fontsize=10, loc="left")
        self.degree_radio = RadioButtons(degree_ax, list(map(str, range(1, 7))), active=self.degree - 1)
        # Notebook 中悬停变色也会触发整张画布同步，取消无必要的悬停刷新。
        self.fit_button = Button(self.fig.add_axes([0.28, 0.20, 0.15, 0.07]), "fit data", color="#54b9ff", hovercolor="#54b9ff")
        self.reset_button = Button(self.fig.add_axes([0.28, 0.10, 0.15, 0.07]), "Reset data", color="0.85", hovercolor="0.85")
        mode_ax = self.fig.add_axes([0.49, 0.13, 0.25, 0.14])
        self.mode_radio = RadioButtons(mode_ax, ["Regression", "Categorical"], active=0 if regression else 1)
        class_ax = self.fig.add_axes([0.80, 0.13, 0.15, 0.14])
        class_ax.set_title("New class", fontsize=10)
        self.class_radio = RadioButtons(class_ax, ["y=1", "y=0"])
        # RadioButtons 默认先同步 draw()，回调又 draw_idle()，会产生两次刷新。
        # 统一由回调请求一次延迟刷新，交给交互后端合并连续事件。
        for radio in (self.degree_radio, self.mode_radio, self.class_radio):
            radio.drawon = False
        self.degree_radio.on_clicked(self._on_degree)
        self.mode_radio.on_clicked(self._on_mode)
        self.class_radio.on_clicked(self._on_class)
        self.fit_button.on_clicked(self.fit)
        self.reset_button.on_clicked(self.reset)
        self._click_id = self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self._draw()

    @staticmethod
    def _check_degree(degree):
        if isinstance(degree, bool) or not isinstance(degree, (int, np.integer)) or not 1 <= degree <= 6:
            raise ValueError("degree 必须是 1 到 6 之间的整数。")

    @staticmethod
    def _make_data(seed):
        rng = np.random.default_rng(seed)
        # 回归：真实关系是二次函数，噪声不是需要学习的规律。
        Xr = np.linspace(-0.85, 0.85, 12)[:, None]
        yr = Xr[:, 0] ** 2 + rng.normal(0, 0.20, len(Xr))
        Xrt = rng.uniform(-1, 1, (1000, 1))
        yrt = Xrt[:, 0] ** 2 + rng.normal(0, 0.20, len(Xrt))
        # 分类：理想的 P(y=1)=0.5 边界为 x1=x0²；边界附近含标签噪声。
        # 独立随机流使分类样本不受回归数据量影响。
        rng = np.random.default_rng(seed + 1)
        Xc = rng.uniform(-1, 1, (55, 2))
        yc = (rng.random(len(Xc)) < _sigmoid(8 * (Xc[:, 1] - Xc[:, 0] ** 2))).astype(float)
        Xct = rng.uniform(-1, 1, (1500, 2))
        yct = (rng.random(len(Xct)) < _sigmoid(8 * (Xct[:, 1] - Xct[:, 0] ** 2))).astype(float)
        return {"Regression": (Xr, yr, Xrt, yrt), "Categorical": (Xc, yc, Xct, yct)}

    @property
    def X(self):
        return self._data[self.mode][0]

    @property
    def y(self):
        return self._data[self.mode][1]

    def _invalidate(self):
        self.weights = None
        self.metrics = {}
        self._draw()

    def set_degree(self, degree):
        """更换阶数；原拟合失效，需要重新点击 fit data。"""
        self._check_degree(degree)
        if self.degree_radio.value_selected != str(degree):
            self.degree_radio.set_active(int(degree) - 1)

    def _on_degree(self, label):
        if self.degree == int(label):
            return
        self.degree = int(label)
        self._invalidate()

    def _on_class(self, label):
        self.fig.canvas.draw_idle()

    def set_mode(self, mode):
        """切换 Regression/Categorical；两种模式各自保留添加的数据。"""
        if mode not in self._data:
            raise ValueError("mode 必须是 'Regression' 或 'Categorical'。")
        if mode != self.mode:
            self.mode_radio.set_active(0 if mode == "Regression" else 1)

    def _on_mode(self, label):
        if self.mode == label:
            return
        self.mode = label
        self._invalidate()

    def reset(self, event=None):
        """恢复当前模式的初始训练数据，保留阶数及独立测试集。"""
        X, y = self._original[self.mode]
        _, _, Xt, yt = self._data[self.mode]
        self._data[self.mode] = (X.copy(), y.copy(), Xt, yt)
        self._invalidate()

    def add_point(self, x, y, *, label=1):
        """添加绘图坐标处的样本。分类时 label 为 0/1，回归时 y 为目标值。"""
        if not np.isfinite([x, y]).all():
            raise ValueError("坐标必须为有限数值。")
        if self.mode == "Regression":
            row, target = [x], y
        else:
            if label not in (0, 1):
                raise ValueError("分类标签必须为 0 或 1。")
            row, target = [x, y], label
        X, targets, Xt, yt = self._data[self.mode]
        self._data[self.mode] = (np.vstack([X, row]), np.append(targets, target), Xt, yt)
        self._invalidate()

    def _on_click(self, event):
        if event.inaxes is not self.ax or event.button != 1 or event.xdata is None or event.ydata is None:
            return
        toolbar = getattr(self.fig.canvas, "toolbar", None)
        if toolbar is not None and getattr(toolbar, "mode", ""):
            return  # 平移或缩放时，不意外添加数据。
        label = 0 if "shift" in (event.key or "").lower() else int(self.class_radio.value_selected[-1])
        self.add_point(event.xdata, event.ydata, label=label)

    def fit(self, event=None):
        """拟合当前训练集并更新图像，返回误差字典。"""
        X, y, Xt, yt = self._data[self.mode]
        A = _features(X, self.degree)
        At = _features(Xt, self.degree)
        if self.mode == "Regression":
            # 最小二乘直接求解，不将测试数据参与训练。
            self.weights = np.linalg.lstsq(A, y, rcond=None)[0]
            self.metrics = {
                "train_mse": float(np.mean((A @ self.weights - y) ** 2)),
                "test_mse": float(np.mean((At @ self.weights - yt) ** 2)),
            }
        else:
            self.weights = _fit_logistic(A, y)
            self.metrics = {}
            for name, features, targets in [("train", A, y), ("test", At, yt)]:
                z = features @ self.weights
                self.metrics[name + "_log_loss"] = float(np.mean(np.logaddexp(0, z) - targets * z))
                self.metrics[name + "_accuracy"] = float(np.mean((z >= 0) == targets))
        self._draw()
        return self.metrics.copy()

    def predict(self, X):
        """输入形状 (n,1)/(n,2)，回归返回数值，分类返回类别 1 的概率。"""
        if self.weights is None:
            raise RuntimeError("请先调用 fit() 或点击 fit data。")
        X = np.asarray(X, dtype=float)
        columns = 1 if self.mode == "Regression" else 2
        if X.ndim != 2 or X.shape[1] != columns or not np.isfinite(X).all():
            raise ValueError(f"X 必须是形状 (n, {columns}) 的有限数值数组。")
        z = _features(X, self.degree) @ self.weights
        return z if self.mode == "Regression" else _sigmoid(z)

    def _setup_plot(self):
        """仅在切换回归/分类时重建坐标轴，日常交互复用已有图形对象。"""
        ax = self.ax
        ax.clear()
        self._contours = []
        self._drawn_weights = None
        self._legend_fitted = None
        x = self._plot_x
        ax.plot(x, x ** 2, ":", color="orangered", label="ideal", linewidth=2)
        if self.mode == "Regression":
            self._samples = ax.scatter([], [], s=32, color="dodgerblue", label="training data", zorder=3)
            self._fit_line, = ax.plot([], [], color="seagreen", label="fitted model", linewidth=2)
            ax.set(xlabel="x", ylabel="y", ylim=(-0.65, 1.65))
            instruction = "Click on plot to add a training point; then click fit data."
        else:
            self._classes = {}
            for label, marker, color in [(1, "x", "red"), (0, "o", "dodgerblue")]:
                self._classes[label] = ax.scatter([], [], marker=marker, color=color, s=27, label=f"y={label}", zorder=3)
            ax.set(xlabel=r"$x_0$", ylabel=r"$x_1$", ylim=(-1.05, 1.10))
            instruction = "Click to add data. Hold Shift for y=0, or select New class below."
        ax.set_xlim(-1.05, 1.05)
        ax.set_title(f"Overfitting Example: {self.mode} data with noise", pad=30)
        ax.text(0.0, 1.025, instruction, transform=ax.transAxes, fontsize=9, color="dodgerblue")
        ax.grid(alpha=0.15)
        self._drawn_mode = self.mode

    def _draw(self):
        # ipympl + plt.ion() 下，每次 set_* 都可能发出一次 draw 消息。
        # 暂停自动刷新，将所有图形修改合并，再显式发送一次 draw_idle。
        # 上下文退出后恢复用户原本的交互模式，不影响其他图表。
        with plt.ioff():
            self._update_plot()
        self.fig.canvas.draw_idle()

    def _update_plot(self):
        if self._drawn_mode != self.mode:
            self._setup_plot()
        fitted = self.weights is not None
        status = "Select a degree, then click fit data."
        if self.mode == "Regression":
            self._samples.set_offsets(np.column_stack([self.X[:, 0], self.y]))
            self._fit_line.set_visible(fitted)
            if fitted:
                if self.weights is not self._drawn_weights:
                    self._fit_line.set_data(self._plot_x, self.predict(self._plot_x[:, None]))
                status = f"MSE: train = {self.metrics['train_mse']:.4f}    test = {self.metrics['test_mse']:.4f}"
        else:
            for label, artist in self._classes.items():
                artist.set_offsets(self.X[self.y == label])
            # 仅拟合结果变化时更新边界，选标签或刷新画布不再重复计算网格。
            if self.weights is not self._drawn_weights:
                for contour in self._contours:
                    contour.remove()
                self._contours = []
                if fitted:
                    if self._grid_features is None:
                        points = np.column_stack([self._grid_x.ravel(), self._grid_y.ravel()])
                        self._grid_features = _features(points, 6)
                    columns = (self.degree + 1) * (self.degree + 2) // 2
                    z = (self._grid_features[:, :columns] @ self.weights).reshape(self._grid_x.shape)
                    self._contours.append(self.ax.contourf(
                        self._grid_x, self._grid_y, z, levels=[-np.inf, 0, np.inf],
                        colors=["#dceeff", "#ffe3dd"], alpha=0.5,
                    ))
                    if z.min() < 0 < z.max():
                        self._contours.append(self.ax.contour(
                            self._grid_x, self._grid_y, z, levels=[0],
                            colors=["seagreen"], linewidths=2,
                        ))
            if fitted:
                status = (
                    f"Cross-entropy: train = {self.metrics['train_log_loss']:.4f}    test = {self.metrics['test_log_loss']:.4f}\n"
                    f"Accuracy: train = {self.metrics['train_accuracy']:.1%}    test = {self.metrics['test_accuracy']:.1%}"
                )
        self._drawn_weights = self.weights
        if fitted != self._legend_fitted:
            handles, labels = self.ax.get_legend_handles_labels()
            if self.mode == "Regression" and not fitted:
                handles, labels = handles[:-1], labels[:-1]
            if self.mode == "Categorical" and fitted:
                handles.append(Line2D([0], [0], color="seagreen", linewidth=2))
                labels.append("fitted boundary (p=0.5)")
            self.ax.legend(handles, labels, loc="lower right", fontsize=8)
            self._legend_fitted = fitted
        self.status.set_text(f"Degree {self.degree}   |   Training samples: {len(self.y)}\n{status}")


def overfit_example(regression=False, *, degree=1, seed=7, show=True):
    """与截图相同的入口：ofit = overfit_example(False)。"""
    return OverfitExample(regression, degree=degree, seed=seed, show=show)


if __name__ == "__main__":
    example = overfit_example()
