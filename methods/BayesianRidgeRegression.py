import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import BayesianRidge
from matplotlib import pyplot as plt


def createBRRModel(X, y, verbose=False):
    """
    Create a Bayesian Ridge Regressor model with the given data.

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

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=60
    )

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize the Bayesian Ridge Regression model
    model = BayesianRidge()

    # Fit the model to the training data
    model.fit(X_train_scaled, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test_scaled)

    # Calculate and print the Root Mean Squared Error (RMSE) as a measure of performance
    if verbose:
        rmse = mean_squared_error(y_test, y_pred)
        print(f"Root Mean Squared Error (RMSE): {rmse}")

    # return the model
    return model, scaler


def plotBRRModel(brr_model, scaler, X, y):
    """
    Plot the model's predictions and the expected results.

    Parameters
    ----------
    brr_model : sklearn.linear_model.BayesianRidge
        The model to plot.
    scaler : sklearn.preprocessing.StandardScaler
        The scaler used to scale the data.
    X : pandas.DataFrame
        The features of the data.
        df[['data_in1', 'data_in2', ...]]
    y : pandas.Series
        The target variable of the data.
        df['data_out']
    """

    # make full predictions with all data
    predictions = brr_model.predict(scaler.transform(X))

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
# brr_model, scaler = createBRRModel(X, y, verbose=True)

# # plot model
# plotBRRModel(brr_model, scaler, X, y)
