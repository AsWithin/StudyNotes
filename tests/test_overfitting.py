"""运行：python -m unittest discover -s tests -v。无需交互后端。"""

import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseEvent
import numpy as np

from Orthers.Overfitting import overfit_example


class OverfittingTests(unittest.TestCase):
    def setUp(self):
        self.demo = overfit_example(show=False)

    def tearDown(self):
        plt.close("all")

    def click(self, ax, x, y, key=None):
        self.demo.fig.canvas.draw()
        px, py = ax.transData.transform((x, y))
        for name in ("button_press_event", "button_release_event"):
            event = MouseEvent(name, self.demo.fig.canvas, px, py, button=1, key=key)
            self.demo.fig.canvas.callbacks.process(name, event)

    def test_default_data_demonstrates_overfitting(self):
        for mode, error in [("Regression", "mse"), ("Categorical", "log_loss")]:
            with self.subTest(mode=mode):
                self.demo.set_mode(mode)
                scores = {}
                for degree in range(1, 7):
                    self.demo.set_degree(degree)
                    scores[degree] = self.demo.fit()
                    self.assertTrue(all(np.isfinite(list(scores[degree].values()))))
                self.assertLess(scores[2]["test_" + error], scores[1]["test_" + error])
                self.assertLess(scores[6]["train_" + error], scores[2]["train_" + error])
                self.assertGreater(scores[6]["test_" + error], scores[2]["test_" + error])

    def test_real_mouse_events_and_fit_button(self):
        n = len(self.demo.y)
        self.click(self.demo.ax, 0.1, 0.2)
        self.assertEqual(self.demo.y[-1], 1)
        self.click(self.demo.ax, 0.2, 0.3, key="shift")
        self.assertEqual(self.demo.y[-1], 0)
        self.demo.class_radio.set_active(1)
        self.click(self.demo.ax, 0.3, 0.4)
        self.assertEqual(self.demo.y[-1], 0)
        self.assertEqual(len(self.demo.y), n + 3)
        self.click(self.demo.fit_button.ax, 0.5, 0.5)
        self.assertIsNotNone(self.demo.weights)
        self.assertEqual(len(self.demo.y), n + 3)
        self.click(self.demo.reset_button.ax, 0.5, 0.5)
        self.assertEqual(len(self.demo.y), n)
        self.assertIsNone(self.demo.weights)

    def test_mode_data_persistence_and_reset(self):
        original = self.demo.X.copy()
        test_data = self.demo._data["Categorical"][2].copy()
        self.demo.add_point(0.5, -0.5, label=0)
        self.demo.set_mode("Regression")
        n = len(self.demo.y)
        self.click(self.demo.ax, 0.5, 1.2)
        self.assertAlmostEqual(self.demo.y[-1], 1.2)
        self.demo.set_mode("Categorical")
        self.assertEqual(len(self.demo.X), len(original) + 1)
        self.demo.reset()
        np.testing.assert_array_equal(self.demo.X, original)
        np.testing.assert_array_equal(self.demo._data["Categorical"][2], test_data)
        self.demo.set_mode("Regression")
        self.assertEqual(len(self.demo.y), n + 1)

    def test_prediction_and_invalidation(self):
        with self.assertRaises(RuntimeError):
            self.demo.predict([[0, 0]])
        self.demo.fit()
        p = self.demo.predict([[0, 0], [0.1, 0.5]])
        self.assertTrue(np.all((0 <= p) & (p <= 1)))
        self.demo.set_degree(6)
        self.assertIsNone(self.demo.weights)
        self.assertEqual(self.demo.metrics, {})
        with self.assertRaises(ValueError):
            self.demo.set_degree(7)
        with self.assertRaises(ValueError):
            self.demo.add_point(0, 0, label=2)

    def test_reproducibility_and_repeated_fits(self):
        other = overfit_example(show=False)
        np.testing.assert_array_equal(self.demo.X, other.X)
        self.demo.set_degree(6)
        first = self.demo.fit()
        artists = len(self.demo.ax.collections)
        second = self.demo.fit()
        self.assertEqual(first, second)
        self.assertEqual(len(self.demo.ax.collections), artists)

    def test_interactive_updates_do_not_flood_draw_requests(self):
        # 模拟 Notebook 开启交互模式；避免一次操作发送几十次刷新请求。
        with plt.ion(), patch.object(self.demo.fig.canvas, "draw_idle") as redraw:
            for action in (
                self.demo.fit,
                lambda: self.demo.set_degree(6),
                lambda: self.demo.add_point(0.2, 0.3),
                lambda: self.demo.set_mode("Regression"),
            ):
                redraw.reset_mock()
                action()
                self.assertGreaterEqual(redraw.call_count, 1)
                self.assertLessEqual(redraw.call_count, 2)
                self.assertTrue(plt.isinteractive())

    def test_plot_data_and_old_boundary_are_updated(self):
        self.demo.fit()
        self.demo.add_point(0.2, 0.3, label=0)
        for label in (0, 1):
            np.testing.assert_array_equal(
                self.demo._classes[label].get_offsets(),
                self.demo.X[self.demo.y == label],
            )
        # 数据变化后，旧决策边界和旧分类底色都应消失，仅剩两组散点。
        self.assertEqual(len(self.demo.ax.collections), 2)
        self.demo.set_mode("Regression")
        self.demo.fit()
        self.demo.add_point(0.3, 1.2)
        self.assertFalse(self.demo._fit_line.get_visible())
        np.testing.assert_array_equal(
            self.demo._samples.get_offsets(),
            np.column_stack([self.demo.X[:, 0], self.demo.y]),
        )


if __name__ == "__main__":
    unittest.main()
