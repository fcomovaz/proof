from sklearn.metrics import *
import numpy as np
import pandas as pd


def error_dataframe(y, y_pred_models):
    """
    This functions takes the y_test and y_pred_models and returns a dataframe with the error metrics for each model.

    Parameters
    ----------
    y : numpy array
        array of the original data to be compared
    y_pred_models : list
        list of numpy arrays of the predicted data from each model
    """

    # unpackage y_pred_models
    lrm = y_pred_models[0]
    brr = y_pred_models[1]
    dtr = y_pred_models[2]
    enr = y_pred_models[3]
    gbr = y_pred_models[4]
    knr = y_pred_models[5]
    krr = y_pred_models[6]
    rfr = y_pred_models[7]
    svr = y_pred_models[8]
    fga = y_pred_models[9]

    # Calculate MSE
    mse_lrm = mean_squared_error(y, lrm)
    mse_brr = mean_squared_error(y, brr)
    mse_dtr = mean_squared_error(y, dtr)
    mse_enr = mean_squared_error(y, enr)
    mse_gbr = mean_squared_error(y, gbr)
    mse_knr = mean_squared_error(y, knr)
    mse_krr = mean_squared_error(y, krr)
    mse_rfr = mean_squared_error(y, rfr)
    mse_svr = mean_squared_error(y, svr)
    mse_fga = mean_squared_error(y, fga)

    # Calculate MAE
    mae_lrm = mean_absolute_error(y, lrm)
    mae_brr = mean_absolute_error(y, brr)
    mae_dtr = mean_absolute_error(y, dtr)
    mae_enr = mean_absolute_error(y, enr)
    mae_gbr = mean_absolute_error(y, gbr)
    mae_knr = mean_absolute_error(y, knr)
    mae_krr = mean_absolute_error(y, krr)
    mae_rfr = mean_absolute_error(y, rfr)
    mae_svr = mean_absolute_error(y, svr)
    mae_fga = mean_absolute_error(y, fga)

    # Calculate R2
    r2_lrm = r2_score(y, lrm)
    r2_brr = r2_score(y, brr)
    r2_dtr = r2_score(y, dtr)
    r2_enr = r2_score(y, enr)
    r2_gbr = r2_score(y, gbr)
    r2_knr = r2_score(y, knr)
    r2_krr = r2_score(y, krr)
    r2_rfr = r2_score(y, rfr)
    r2_svr = r2_score(y, svr)
    r2_fga = r2_score(y, fga)

    # Calculate MAPE
    mape_lrm = np.mean(np.abs((y - lrm) / y)) * 100
    mape_brr = np.mean(np.abs((y - brr) / y)) * 100
    mape_dtr = np.mean(np.abs((y - dtr) / y)) * 100
    mape_enr = np.mean(np.abs((y - enr) / y)) * 100
    mape_gbr = np.mean(np.abs((y - gbr) / y)) * 100
    mape_knr = np.mean(np.abs((y - knr) / y)) * 100
    mape_krr = np.mean(np.abs((y - krr) / y)) * 100
    mape_rfr = np.mean(np.abs((y - rfr) / y)) * 100
    mape_svr = np.mean(np.abs((y - svr) / y)) * 100
    mape_fga = np.mean(np.abs((y - fga) / y)) * 100

    # Calculate SMAPE
    smape_lrm = np.mean(np.abs((y - lrm) / ((np.abs(y) + np.abs(lrm)) / 2))) * 100
    smape_brr = np.mean(np.abs((y - brr) / ((np.abs(y) + np.abs(brr)) / 2))) * 100
    smape_dtr = np.mean(np.abs((y - dtr) / ((np.abs(y) + np.abs(dtr)) / 2))) * 100
    smape_enr = np.mean(np.abs((y - enr) / ((np.abs(y) + np.abs(enr)) / 2))) * 100
    smape_gbr = np.mean(np.abs((y - gbr) / ((np.abs(y) + np.abs(gbr)) / 2))) * 100
    smape_knr = np.mean(np.abs((y - knr) / ((np.abs(y) + np.abs(knr)) / 2))) * 100
    smape_krr = np.mean(np.abs((y - krr) / ((np.abs(y) + np.abs(krr)) / 2))) * 100
    smape_rfr = np.mean(np.abs((y - rfr) / ((np.abs(y) + np.abs(rfr)) / 2))) * 100
    smape_svr = np.mean(np.abs((y - svr) / ((np.abs(y) + np.abs(svr)) / 2))) * 100
    smape_fga = np.mean(np.abs((y - fga) / ((np.abs(y) + np.abs(fga)) / 2))) * 100

    # Calculate MSLE
    msle_lrm = mean_squared_log_error(y, lrm)
    msle_brr = mean_squared_log_error(y, brr)
    msle_dtr = mean_squared_log_error(y, dtr)
    msle_enr = mean_squared_log_error(y, enr)
    msle_gbr = mean_squared_log_error(y, gbr)
    msle_knr = mean_squared_log_error(y, knr)
    msle_krr = mean_squared_log_error(y, krr)
    msle_rfr = mean_squared_log_error(y, rfr)
    msle_svr = mean_squared_log_error(y, svr)
    msle_fga = mean_squared_log_error(y, fga)

    # Create DataFrame
    errors_columns = ["method", "rmse", "mae", "r2", "mape", "smape", "msle"]
    error_df = pd.DataFrame(columns=errors_columns)

    # insert errors
    error_df.loc[0] = ["lrm", mse_lrm, mae_lrm, r2_lrm, mape_lrm, smape_lrm, msle_lrm]
    error_df.loc[1] = ["brr", mse_brr, mae_brr, r2_brr, mape_brr, smape_brr, msle_brr]
    error_df.loc[2] = ["dtr", mse_dtr, mae_dtr, r2_dtr, mape_dtr, smape_dtr, msle_dtr]
    error_df.loc[3] = ["enr", mse_enr, mae_enr, r2_enr, mape_enr, smape_enr, msle_enr]
    error_df.loc[4] = ["gbr", mse_gbr, mae_gbr, r2_gbr, mape_gbr, smape_gbr, msle_gbr]
    error_df.loc[5] = ["knr", mse_knr, mae_knr, r2_knr, mape_knr, smape_knr, msle_knr]
    error_df.loc[6] = ["krr", mse_krr, mae_krr, r2_krr, mape_krr, smape_krr, msle_krr]
    error_df.loc[7] = ["rfr", mse_rfr, mae_rfr, r2_rfr, mape_rfr, smape_rfr, msle_rfr]
    error_df.loc[8] = ["svr", mse_svr, mae_svr, r2_svr, mape_svr, smape_svr, msle_svr]
    error_df.loc[9] = ["fga", mse_fga, mae_fga, r2_fga, mape_fga, smape_fga, msle_fga]

    # Return DataFrame
    return error_df
