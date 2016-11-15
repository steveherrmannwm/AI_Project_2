
import GradientDescentOld as gd
import numpy as np
import LinearModel
# def f(x = 1, y = 1, w = np.array([1])):
#     return 6 * w - 2
#
# if __name__ == "__main__":
#     x = np.array([-3, -2, -1, 0, 1, 2, 3])
#     w, wtrace = gd.GradientDescent().GD(f, 0.1, 10, np.array([0]), np.array([[1]]), np.array([1]), True)
#     print w, wtrace
#     gd.plot_trace(x, "Test trace", f, wtrace)
lm = LinearModel.LinearModel()
model = lm.res('train', np.array([[0,0],[0,1],[1,1]]), np.array([-1,1,1]), 0.1, 100, False, 'perceptron', 0.1, False)
print model