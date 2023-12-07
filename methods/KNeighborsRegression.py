from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
from matplotlib import pyplot as plt


def createKNRModel(X_train, y_train, verbose=False):
    """
    Create a K Nearest Neighbors regression model with the given data.

    Parameters
    ----------
    X : pandas.DataFrame
        The features of the data.
        df[['data_in1', 'data_in2', ...]]
    y : pandas.Series
        The target variable of the data.
        df['data_out']
    verbose : bool, optional
        Whether to print the RMSE of the model. The default is False.
    """

    # X = X.to_numpy()
    # y = y.to_numpy()
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()

    # Splitting the data into training and testing sets
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.2, random_state=60
    # )


    # Creating the KNN regression model
    knn_model = KNeighborsRegressor(
        n_neighbors=4
    )  # Adjust the number of neighbors as needed

    # Training the model
    knn_model.fit(X_train, y_train)

    # Making predictions
    # y_pred = knn_model.predict(X_test)

    # Evaluating the model (for regression, you might use metrics like mean squared error)
    # if verbose:
    #     mse = mean_squared_error(y_test, y_pred)
    #     print("Mean Squared Error:", mse)

    # Return the model
    return knn_model


def plotKNRModel(knn_model, X, y):
    """
    Plot the model's predictions and the expected results.

    Parameters
    ----------
    knn_model : sklearn.neighbors.KNeighborsRegressor
        The model to plot.
    X : pandas.DataFrame
        The features of the data.
        df[['data_in1', 'data_in2', ...]]
    y : pandas.Series
        The target variable of the data.
        df['data_out']
    """

    # make full predictions with all data
    predictions = knn_model.predict(X)

    # plot predictions and expected results
    x_axis = [i for i in range(len(y))]
    plt.plot(x_axis, y, label="Expected")
    plt.plot(x_axis, predictions, label="Predicted")
    plt.legend()
    plt.show()


# ==================================================
# =========== MINIMAL IMPLEMENTATION ===============
# ==================================================
# # load dataframe
# df = pd.read_csv("monthly_data_csv.csv").dropna()

# # create X and y
# X = df[["RH", "TMP"]]
# y = df["PM10"]

# # create model
# knn_model = createKNRModel(X, y, verbose=True)

# # plot model
# plotKNRModel(knn_model, X, y)
