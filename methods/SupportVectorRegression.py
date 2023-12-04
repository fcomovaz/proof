import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from matplotlib import pyplot as plt


def createSVRModel(X, y, verbose=False):
    """
    Create a Support Vector Regressor model with the given data.

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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize the Support Vector Regression model
    # You can adjust hyperparameters like C (regularization parameter), kernel, epsilon, etc.
    model = SVR(kernel="rbf", C=40.0, epsilon=1.55)

    # Fit the model to the training data
    model.fit(X_train_scaled, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test_scaled)

    # Calculate and print the Root Mean Squared Error (RMSE) as a measure of performance
    if verbose:
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        print(f"Root Mean Squared Error (RMSE): {rmse}")

    # return the model
    return model, scaler


def plotSVRModel(svr_model, scaler, X, y):
    """
    Plot the model's predictions and the expected results.

    Parameters
    ----------
    svr_model : sklearn.svm.SVR
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
    predictions = svr_model.predict(scaler.transform(X))

    # plot predictions and expected results
    x_axis = range(len(y))
    # calculate mse
    mse = mean_squared_error(y, predictions)
    print("MSE: %.5f" % mse)
    plt.plot(x_axis, y, label="Actual")
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
# svr_model, scaler = createSVRModel(X, y, verbose=True)

# # plot model
# plotSVRModel(svr_model, scaler, X, y)

