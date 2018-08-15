import matplotlib.pyplot as plt



def plot_gp(xtest, predictions, std=None, xtrain=None, ytrain=None,  title=None, save_name=None):

    xtest, predictions = xtest.squeeze(), predictions.squeeze()


    fig, ax = plt.subplots()

    # Plot the training data
    if (xtrain is not None) and (ytrain is not None):
        xtrain, ytrain = xtrain.squeeze(), ytrain.squeeze()
        ax.scatter(xtrain, ytrain, s=100, color='r', label='Training Data')

    # plot the testing data
    ax.plot(xtest, predictions, linewidth=5,
            color='k', label='Predictions')

    # plot the confidence interval
    if std is not None:
        std = std.squeeze()
        upper_bound = predictions + 1.960 * std
        lower_bound = predictions - 1.960 * std

        ax.fill_between(xtest, upper_bound, lower_bound,
                        color='red', alpha=0.2, label='95% Condidence Interval')
    ax.legend()
    if title is not None:
        ax.set_title(title)

    if save_name:
        pass
    else:
        plt.show()

    return fig
