import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor


def createDTRModel(X_train, y_train, verbose=False):
    """
    Create a Decision Tree Regressor with the given data.

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

    # Splitting the data into training and testing sets
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.2, random_state=60
    # )

    # Creating the Decision Tree Regressor model
    dt_regressor = DecisionTreeRegressor(
        criterion="friedman_mse", random_state=60, splitter="best"
    )

    # Training the model
    dt_regressor.fit(X_train, y_train)

    # Making predictions
    # y_pred = dt_regressor.predict(X_test)

    # Evaluating the model (for regression, you might use metrics like mean squared error)
    # if verbose:
    #     mse = mean_squared_error(y_test, y_pred)
    #     print("Mean Squared Error:", mse)

    # Return the model
    return dt_regressor


def plotDTRModel(dt_regressor, X, y):
    """
    Plot the model's predictions and the expected results.

    Parameters
    ----------
    dt_regressor : sklearn.tree.DecisionTreeRegressor
        The model to plot.
    X : pandas.DataFrame
        The features of the data.
        df[['data_in1', 'data_in2', ...]]
    y : pandas.Series
        The target variable of the data.
        df['data_out']
    """

    # make full predictions with all data
    predictions = dt_regressor.predict(X)

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
# dt_regressor = createDTRModel(X, y, verbose=True)

# # plot model
# plotDTRModel(dt_regressor, X, y)
