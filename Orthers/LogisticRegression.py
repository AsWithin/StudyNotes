import numpy as np
import copy
import math
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

X_train = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])


# sigmoid函数
def sigmoid(z):
    # 与 1 / (1 + exp(-z)) 等价，避免极端负数使 exp(-z) 溢出。
    return np.exp(-np.logaddexp(0, -np.asarray(z)))


# 逻辑回归的成本函数
def compute_cost_logistic(X, y, w, b):
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        z_i = np.dot(X[i], w) + b
        # 单个样本的交叉熵：log(1 + exp(z_i)) - y[i] * z_i
        cost += np.logaddexp(0, z_i) - y[i] * z_i
    cost /= m
    return cost


# 计算梯度函数
def compute_gradient_logistic(X, y, w, b):
    m, n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.0

    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i], w) + b)
        err_i = f_wb_i - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_i * X[i, j]
        dj_db += err_i
    dj_dw /= m
    dj_db /= m
    return dj_dw, dj_db


# 梯度下降函数
def gradient_descent(X, y, w_in, b_in, alpha, num_iters):
    J_history = []
    w = copy.deepcopy(w_in)
    b = copy.deepcopy(b_in)
    # 下标 0 表示尚未更新；随后记录每次更新后的参数，供绘图回放。
    parameter_history = [(w.copy(), b)]

    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient_logistic(X, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db
        J_history.append(compute_cost_logistic(X, y, w, b))
        parameter_history.append((w.copy(), b))
        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i + 1:5d}: Cost {J_history[-1]}   ")
    return w, b, J_history, parameter_history


def plot_static(X_train, y_train, w_out, b_out, J_history):
    """静态图：最终决策边界、全部训练成本、最后 100 次训练成本。

    在笔记本中定义此函数后，直接调用即可，不需要交互式绘图后端。
    """
    fig, (ax_boundary, ax_cost, ax_tail) = plt.subplots(
        1, 3, figsize=(15, 4.5), constrained_layout=True
    )

    # 1. 真实样本与训练完成后的决策边界。
    for label, marker, color in [(0, "o", "tab:blue"), (1, "x", "tab:red")]:
        selected = y_train == label
        ax_boundary.scatter(
            X_train[selected, 0], X_train[selected, 1],
            marker=marker, color=color, s=75, label=f"y={label}",
        )

    # 决策边界：w[0] * x0 + w[1] * x1 + b = 0。
    if abs(w_out[1]) > 1e-12:
        x0 = np.array([0.0, 4.0])
        x1 = -(w_out[0] * x0 + b_out) / w_out[1]
        ax_boundary.plot(x0, x1, color="black", label="Decision boundary (P=0.5)")
    elif abs(w_out[0]) > 1e-12:
        ax_boundary.axvline(-b_out / w_out[0], color="black", label="Decision boundary (P=0.5)")
    ax_boundary.set(
        title="Final decision boundary", xlabel=r"$x_0$", ylabel=r"$x_1$",
        xlim=(0, 4), ylim=(0, 3.5),
    )
    ax_boundary.legend(fontsize=8)

    # 2. J_history[0] 是第 1 次更新后的成本，因此横坐标从 1 开始。
    steps = np.arange(1, len(J_history) + 1)
    ax_cost.plot(steps, J_history, color="tab:orange")
    ax_cost.set(title="Cost: all iterations", xlabel="Iteration", ylabel="Cost J(w, b)")

    # 3. 切片保留真实训练编号，例如 9901~10000，而不是重新编号为 0~99。
    ax_tail.plot(steps[-100:], J_history[-100:], color="tab:orange")
    ax_tail.set(title="Cost: last 100 iterations", xlabel="Iteration", ylabel="Cost J(w, b)")
    ax_tail.ticklabel_format(axis="y", style="plain", useOffset=False)
    for ax in (ax_boundary, ax_cost, ax_tail):
        ax.grid(alpha=0.25)
    plt.show()
    return fig


def plot_dynamic(X_train, y_train, J_history, parameter_history):
    iters = len(J_history)
    # 三幅图共享同一个训练进度：概率与边界、完整成本、最后 100 次成本。
    # 先完成训练再回放，播放速度不会影响梯度下降。
    fig, (ax_boundary, ax_cost, ax_tail) = plt.subplots(1, 3, figsize=(15, 5.5))
    fig.subplots_adjust(left=0.055, right=0.98, bottom=0.25, top=0.86, wspace=0.4)

    x0_grid, x1_grid = np.meshgrid(np.linspace(0, 4, 120), np.linspace(0, 3.5, 110))
    probability_image = ax_boundary.pcolormesh(
        x0_grid, x1_grid, np.full_like(x0_grid, 0.5),
        cmap="Blues", vmin=0, vmax=1, shading="auto", alpha=0.65,
    )
    fig.colorbar(probability_image, ax=ax_boundary, label="P(y=1)", fraction=0.045)
    for label, marker, color in [(0, "o", "tab:blue"), (1, "x", "tab:red")]:
        selected = y_train == label
        ax_boundary.scatter(
            X_train[selected, 0], X_train[selected, 1],
            marker=marker, color=color, s=75, label=f"y={label}", zorder=3,
        )
    boundary_line, = ax_boundary.plot([], [], color="black", lw=2, label="P=0.5")
    ax_boundary.set(xlim=(0, 4), ylim=(0, 3.5), xlabel=r"$x_0$", ylabel=r"$x_1$")
    ax_boundary.legend(loc="upper right", fontsize=8)

    # costs[0] 是初始成本；costs[k] 对应第 k 次更新后的参数。
    initial_w, initial_b = parameter_history[0]
    costs = np.r_[compute_cost_logistic(X_train, y_train, initial_w, initial_b), J_history]
    steps = np.arange(iters + 1)
    tail_start = max(1, iters - 99)  # 10000 次训练时显示 9901..10000，恰好 100 个点。
    for axes, start, title in [
        (ax_cost, 0, "Cost: all iterations"),
        (ax_tail, tail_start, "Cost: last 100 iterations"),
    ]:
        # 灰线给出完整轨迹作参考，彩色线表示当前回放到的位置。
        axes.plot(steps[start:], costs[start:], color="0.8", lw=2, label="Full history")
        axes.set(title=title, xlabel="Iteration", ylabel="Cost J(w, b)")
        axes.grid(alpha=0.25)
        axes.ticklabel_format(axis="y", style="plain", useOffset=False)
    cost_line, = ax_cost.plot([], [], color="tab:orange", lw=2)
    tail_line, = ax_tail.plot([], [], color="tab:orange", lw=2)
    cost_marker, = ax_cost.plot([], [], "o", color="tab:red")
    tail_marker, = ax_tail.plot([], [], "o", color="tab:red")
    tail_note = ax_tail.text(0.04, 0.93, "", transform=ax_tail.transAxes, va="top", fontsize=9)

    slider = Slider(fig.add_axes([0.13, 0.12, 0.60, 0.035]), "Iteration", 0, iters, valinit=0, valstep=1)
    play_button = Button(fig.add_axes([0.80, 0.09, 0.08, 0.07]), "Pause")
    restart_button = Button(fig.add_axes([0.89, 0.09, 0.08, 0.07]), "Restart")
    status_text = fig.suptitle("")


    def update_plot(value):
        k = int(value)
        w, b = parameter_history[k]
        probabilities = sigmoid(w[0] * x0_grid + w[1] * x1_grid + b)
        probability_image.set_array(probabilities.ravel())
        # w0*x0 + w1*x1 + b = 0。单独处理垂直边界及全零初始参数。
        if abs(w[1]) > 1e-12:
            x0 = np.array([0.0, 4.0])
            boundary_line.set_data(x0, -(w[0] * x0 + b) / w[1])
        elif abs(w[0]) > 1e-12:
            boundary_line.set_data([-b / w[0]] * 2, [0, 3.5])
        else:
            boundary_line.set_data([], [])
        ax_boundary.set_title("Initial: P=0.5 everywhere" if k == 0 else "Probability and decision boundary")
        cost_line.set_data(steps[:k + 1], costs[:k + 1])
        cost_marker.set_data([k], [costs[k]])
        if k >= tail_start:
            tail_line.set_data(steps[tail_start:k + 1], costs[tail_start:k + 1])
            tail_marker.set_data([k], [costs[k]])
            tail_note.set_text("")
        else:
            tail_line.set_data([], [])
            tail_marker.set_data([], [])
            tail_note.set_text(f"Replay has not reached step {tail_start}")
        status_text.set_text(f"Iteration {k}/{iters} | Cost = {costs[k]:.8f} | w = [{w[0]:.3f}, {w[1]:.3f}], b = {b:.3f}")
        fig.canvas.draw_idle()


    # 早期逐步播放；中段跳帧；最后 100 次逐步播放以观察微小变化。
    playback_steps = np.unique(np.r_[
        np.arange(0, min(100, iters) + 1),
        np.linspace(min(100, iters), tail_start, 180, dtype=int),
        np.arange(tail_start, iters + 1),
    ])
    playing = True
    timer = fig.canvas.new_timer(interval=50)


    def advance_frame():
        next_index = np.searchsorted(playback_steps, int(slider.val), side="right")
        if next_index < len(playback_steps):
            slider.set_val(int(playback_steps[next_index]))
        else:
            nonlocal playing
            playing = False
            timer.stop()
            play_button.label.set_text("Play")
            fig.canvas.draw_idle()


    def toggle_play(event):
        nonlocal playing
        playing = not playing
        if playing:
            if int(slider.val) == iters:
                slider.set_val(0)
            timer.start()
        else:
            timer.stop()
        play_button.label.set_text("Pause" if playing else "Play")
        fig.canvas.draw_idle()


    def restart(event):
        nonlocal playing
        slider.set_val(0)
        playing = True
        play_button.label.set_text("Pause")
        timer.start()


    slider.on_changed(update_plot)
    play_button.on_clicked(toggle_play)
    restart_button.on_clicked(restart)
    timer.add_callback(advance_frame)
    fig.canvas.mpl_connect("close_event", lambda event: timer.stop())
    update_plot(0)
    timer.start()
    plt.show()


def main():
    mode = input("请选择绘图模式（0：静态图，1：动态图）：").strip()
    while mode not in ("0", "1"):
        mode = input("请输入 0 或 1：").strip()

    w_tmp = np.zeros_like(X_train[0])
    b_tmp = 0.0
    alpha = 0.1
    iters = 10000

    w_out, b_out, J_history, parameter_history = gradient_descent(
        X_train, y_train, w_tmp, b_tmp, alpha, iters
    )
    print(f"\nupdated parameters: w:{w_out}, b:{b_out}")

    if mode == "0":
        plot_static(X_train, y_train, w_out, b_out, J_history)
    else:
        plot_dynamic(X_train, y_train, J_history, parameter_history)


if __name__ == "__main__":
    main()
