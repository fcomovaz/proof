# for accesing to files and folders
import os

# for data manipulation
import numpy as np
import pandas as pd


# this program will take selected dataframes from certain year and put it in dataframe
# for example will take XXXXTMP.csv, XXXXRH.csv, XXXXPM10.csv and put it in XXXX.csv
# csv to create
folder_name = "seasonal_data_csv"

df_columns = ["TMP", "RH", "PM10"]
# create the dataframe
df = pd.DataFrame(columns=["TMP", "RH", "PM10"])

# set the years to get the data
years = [str(i) for i in range(2013, 2024)]

# each column in the files is a station, so choose one station
station = "CUA"

# columns to append to the dataframe
tmp = []
rh = []
pm10 = []

for year in years:
    tmp_file = year + "TMP.csv"
    RH_file = year + "RH.csv"
    pm10_file = year + "PM10.csv"

    # read each file from the folder
    tmp_df = pd.read_csv(folder_name + "/" + tmp_file)
    rh_df = pd.read_csv(folder_name + "/" + RH_file)
    pm10_df = pd.read_csv(folder_name + "/" + pm10_file)

    # this is a big df, just select the columns of the station
    tmp.append(tmp_df[station])
    rh.append(rh_df[station])
    pm10.append(pm10_df[station])

# append the columns to the dataframe
df["TMP"] = pd.concat(tmp, ignore_index=True)
df["RH"] = pd.concat(rh, ignore_index=True)
df["PM10"] = pd.concat(pm10, ignore_index=True)

# save the dataframe into a csv file
df.to_csv(folder_name + ".csv", index=False)
