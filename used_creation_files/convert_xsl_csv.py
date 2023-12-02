# for creating files and folders
import os

# for reading and writing excel files
import pandas as pd
import numpy as np

# convert all files from raw_data_xls to csv and save it into a new folder
folder_name = "raw_data_csv"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# get all the files from raw_data_xls
files = os.listdir("raw_data_xls")

# convert all the files to csv
for f in files:
    # read the file
    df = pd.read_excel("raw_data_xls/" + f)

    # # in these files Nan values are represented as "-99.0"
    df = df.replace(-99, np.nan)

    # create a new file
    df.to_csv(folder_name + "/" + f[:-4] + ".csv", index=False)

    # print the progress in the same line as 9/100 files processed
    print(
        "\r" + str(files.index(f) + 1) + "/" + str(len(files)) + " files processed",
        end="",
    )
