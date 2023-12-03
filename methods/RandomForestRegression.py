from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
from matplotlib import pyplot as plt


def createRFRModel(X, y, verbose=False):
    """
    Create a Random Forest Regressor model with the given data.

    @param X: The independent variables. df[['data_in1', 'data_in2', ...]]
    @param y: The dependent variable. df['data_out']
    @param verbose: If True, print the mean squared error of the model.
    """

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    # Inicializar el modelo de Random Forest Regresión
    rf_model = RandomForestRegressor(
        n_estimators=60, random_state=0, criterion="poisson"
    )

    # Train the model
    rf_model.fit(X_train, y_train)

    # Use the forest's predict method on the test data
    predictions = rf_model.predict(X_test)

    # Calculate the mean squared error of the model
    if verbose:
        mse = mean_squared_error(y_test, predictions)
        print(f"Error cuadrático medio: {mse}")

    # Return the model
    return rf_model


def plotRFRModel(rf_model, X, y):
    """
    Plot the model's predictions and the expected results.

    @param rf_model: The model to plot.
    @param X: The independent variables. df[['data_in1', 'data_in2', ...]]
    @param y: The dependent variable. df['data_out']
    """

    # make full predictions with all data
    predictions = rf_model.predict(X)

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
# X = df[["TMP", "RH"]]
# y = df["PM10"]

# # create model
# rf_model = createRFRModel(X, y, verbose=True)

# # plot model
# plotRFRModel(rf_model, X, y)
