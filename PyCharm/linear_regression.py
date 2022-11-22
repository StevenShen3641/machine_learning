import numpy as np
import matplotlib.pyplot as plt


def compute_model_output(x, w, b):
    """
    Computes the prediction of a linear model
    Args:
      x (ndarray (m,)): Data, m examples
      w,b (scalar)    : model parameters
    Returns
      y (ndarray (m,)): target values
    """
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b

    return f_wb


if __name__ == "__main__":
    plt.style.use('./deeplearning.mplstyle')
    x_train = np.array([1.0, 2.0])  # input feature
    y_train = np.array([300.0, 500.0])  # target variable
    print(f"x_train = {x_train}")
    print(f"y_train = {y_train}")

    # Number of training examples m
    print(f"x_train.shape: {x_train.shape}")
    print(f"y_train.shape: {y_train.shape}")

    m = x_train.shape[0]  # length of the first dimension, same as the example number
    print(f"Number of training examples is: {m}")

    # or we can use len() as well
    # m = len(x_train)
    # print(f"Number of training examples is: {m}")

    # Since Python is zero index, we start the example notation from 0
    for i in [0, 1]:
        x_i = x_train[i]
        y_i = y_train[i]
        print(f"(x^({i}), y^({i})) = ({x_i}, {y_i})")

    # plot the data points
    plt.scatter(x_train, y_train, marker='x', c='r')  # marker and c show the points as red crosses
    # set the title
    plt.title("Housing Prices")
    # set the y-axis label
    plt.ylabel('Price (in 1000s of dollars)')
    # set the x-axis label
    plt.xlabel('Size (1000 sqft)')
    plt.show()

    w = 200
    b = 100

    tmp_f_wb = compute_model_output(x_train, w, b)

    # plot our model prediction, using plot() to draw a line
    plt.plot(x_train, tmp_f_wb, c='b', label='Our Prediction')
    # plot the data points
    plt.scatter(x_train, y_train, marker='x', c='r', label='Actual Values')
    # set the title
    plt.title("Housing Prices")
    # set the y-axis label
    plt.ylabel('Price (in 1000s of dollars)')
    # set the x-axis label
    plt.xlabel('Size (1000 sqft)')
    plt.legend()
    plt.show()

    # choose x to be 1.2, estimate y
    x_i = 1.2
    cost_1200sqft = w * x_i + b

    print(f"${cost_1200sqft:.0f} thousand dollars")
