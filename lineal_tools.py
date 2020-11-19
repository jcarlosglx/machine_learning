import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from pandas import DataFrame


def plot_twenty_points(y_test, y_prediction):
    """Display the result of the model, show the first 20 points

    Args:
        y_test (Array): An array of the axis Y traint
        y_prediction (Array): An array of the axis Y predicted by the model
    """
    axis_x = list(range(20))
    axis_y = y_test[:20]
    axis_y2 = y_prediction[:20]
    fig, ax = plt.subplots()
    ax.plot(axis_x, axis_y, "-", axis_x, axis_y2, "o")
    fig.set_size_inches(15, 8)
    plt.show()


def get_error_model(name_model, train_data, test_data, alphas):
    """Return the error given by a model for each alpha

    Args:
        name_model (Class): A class for machine learning
        train_data (List): A list with the element on training data in the
        form: (train_x, train_y)
        test_data (List): A list with the element on training data in the
        form: (train_x, train_y)
        alphas (List): A list with the alphas to the test

    Returns:
        List: A list of errors for each alpha
    """
    x_train, y_train = train_data
    x_test, y_test = test_data
    error_list = []
    for alpha in alphas:
        model = name_model(alpha=alpha)
        model.fit(x_train, y_train)
        y_prediction = model.predict(x_test)
        error_model = round(mean_squared_error(y_test, y_prediction), 3)
        error_list.append(error_model)
    return error_list


def train_model(name_model, train_data, test_data):
    """Train a model and show the results

    Args:
        name_model (Class): A class for machine learning
        train_data (List): A list with the element on training data in the
        form: (train_x, train_y)
        test_data (List): A list with the element on training data in the
        form: (train_x, train_y)

    Returns:
        Object: An object from the class trained
    """
    x_train, y_train = train_data
    x_test, y_test = test_data
    model = name_model()
    model.fit(x_train, y_train)
    y_prediction = model.predict(x_test)
    plot_twenty_points(y_test, y_prediction)
    return model


def print_alpha_errors(name_model, train_data, test_data, alphas):
    """Print the first five error in the model and the best alpha for the model

    Args:
        name_model (Class): A class for machine learning
        train_data (List): A list with the element on training data in the
        form: (train_x, train_y)
        test_data (List): A list with the element on training data in the
        form: (train_x, train_y)
        alphas (List): A list with the alphas to the test
    """
    print("Errors for {}".format(str(name_model)))
    mse_list = get_error_model(name_model, train_data, test_data, alphas)
    alpha_error = DataFrame({"mse": mse_list, "alpha": alphas})
    print(alpha_error.head())
    best_alpha = alpha_error[alpha_error["mse"] == alpha_error["mse"].min()]
    print("Best Alpha: \n{}\n".format(best_alpha))
