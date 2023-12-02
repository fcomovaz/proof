# for accesing to files and folders
import os

# for data manipulation
import numpy as np
import pandas as pd

# folder name
folder_name = "seasonal_data_csv"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# get all the files from monthly_data_csv
files = os.listdir("monthly_data_csv")

# create a relation between months and seasons in north hemisphere
# 3 = winter, 0 = spring, 1 = summer, 2 = autumn
seasons = {
    3: ["12", "01", "02"],
    0: ["03", "04", "05"],
    1: ["06", "07", "08"],
    2: ["09", "10", "11"],
}

# add the seasons to the dataframe into a new column called SEASON
for f in files:
    df = pd.read_csv("monthly_data_csv/" + f)

    # convert fecha to datetime
    df["FECHA"] = pd.to_datetime(df["FECHA"])
    df["FECHA"] = df["FECHA"].dt.strftime("%m")
    df["SEASON"] = np.nan
    for season, months in seasons.items():
        df.loc[df["FECHA"].isin(months), "SEASON"] = season

    df.to_csv(folder_name + "/" + f[:-4] + ".csv", index=False)

    # print the progress in the same line as 9/100 files processed
    print(
        "\r" + str(files.index(f) + 1) + "/" + str(len(files)) + " files processed",
        end="",
    )



# get the average per month of each file and save it into a new folder
folder_name = "seasonal_data_csv"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# get all the files from seasonal_data_csv
files = os.listdir("seasonal_data_csv")

# get the average per month of each file and save it into a new folder
for f in files:
    df = pd.read_csv("seasonal_data_csv/" + f)

    df = df.groupby(["SEASON"]).mean()
    df.reset_index(inplace=True)

    df.to_csv(folder_name + "/" + f[:-4] + ".csv", index=False)

    # print the progress in the same line as 9/100 files processed
    print(
        "\r" + str(files.index(f) + 1) + "/" + str(len(files)) + " files processed",
        end="",
    )
