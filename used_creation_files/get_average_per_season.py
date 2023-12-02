# for accesing to files and folders
import os

# for data manipulation
import numpy as np
import pandas as pd


# get the average per month of each file and save it into a new folder
folder_name = "seasonal_data_csv"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# get all the files from seasonal_data_csv
files = os.listdir("seasonal_data_csv")

# get the average per month of each file and save it into a new folder
for f in files:
    df = pd.read_csv("seasonal_data_csv/" + f, delimiter=',')

    df = df.groupby(["SEASON"]).mean()
    df.reset_index(inplace=True)

    df.to_csv(folder_name + "/" + f[:-4] + ".csv", index=False)

    # print the progress in the same line as 9/100 files processed
    print(
        "\r" + str(files.index(f) + 1) + "/" + str(len(files)) + " files processed",
        end="",
    )
