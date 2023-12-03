import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt


def createKRRModel(X, y, verbose=False):
    """
    Create a Kernel Ridge Regressor model with the given data.

    @param X: The independent variables. df[['data_in1', 'data_in2', ...]]
    @param y: The dependent variable. df['data_out']
    @param verbose: If True, print the mean squared error of the model.
    """

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=60
    )

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize the Kernel Ridge Regression model with a radial basis function (RBF) kernel
    model = KernelRidge(alpha=0.4, kernel="rbf", gamma=0.056)

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


def plotKRRModel(krr_model, scaler, X, y):
    """
    Plot the model's predictions and the expected results.

    @param rf_model: The model to plot.
    @param X: The independent variables. df[['data_in1', 'data_in2', ...]]
    @param y: The dependent variable. df['data_out']
    """

    # make full predictions with all data
    predictions = krr_model.predict(scaler.transform(X))

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
# krr_model, scaler = createKRRModel(X, y, verbose=True)

# # plot model
# plotKRRModel(krr_model, scaler, X, y)
