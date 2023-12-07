from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
from matplotlib import pyplot as plt


def createLRModel(X, y, verbose=False):
    """
    Create a Linear Regression model with the given data.

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

    # Create a Linear Regression model object
    model = LinearRegression()

    # Train the model using the training sets
    model.fit(X_train, y_train)

    # Make predictions using the testing set
    y_pred = model.predict(X_test)

    # Calculate the Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)
    if verbose:
        print(mse)

    # Return the model
    return model


def plotLRModel(model, X, y):
    """
    Plot the model's predictions and the expected results.

    Parameters
    ----------
    model : sklearn.linear_model.LinearRegression
        The model to plot.
    X : pandas.DataFrame
        The features of the data.
        df[['data_in1', 'data_in2', ...]]
    y : pandas.Series
        The target variable of the data.
        df['data_out']
    """

    # plot
    x_total = [x for x in range(0, len(y))]
    y_predic = model.predict(X)
    plt.plot(x_total, y, label="Real")
    plt.plot(x_total, y_predic, label="Predicted")
    plt.legend()
    plt.show()


# ==================================================
# =========== MINIMAL IMPLEMENTATION ===============
# ==================================================
# load dataframe
df = pd.read_csv("seasonal_data_csv.csv").dropna()

# create X and y
X = df[["TMP", "RH"]]
y = df["PM10"]

# create model
rf_model = createLRModel(X, y, verbose=True)

# plot model
plotLRModel(rf_model, X, y)
