import tfs_cleaning
from tfs_ops_settings import *
from tfs_cleaning import *
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import dask as dd
from scipy import stats
import plotly.express as px
import os


def retrieve_volumes_data(file_path: str):

    volumes = pd.read_csv(file_path)

    return volumes


def analyze_volumes(volumes: pd.DataFrame):

    print(volumes.head(10))

    # --------------- Calculations ---------------

    trp_id = volumes["trp_id"][0]
    data_shape = volumes.shape

    percentile_25 = np.percentile(volumes["volume"], 25)
    percentile_50 = np.percentile(volumes["volume"], 50)
    percentile_75 = np.percentile(volumes["volume"], 75)
    percentile_95 = np.percentile(volumes["volume"], 95)
    percentile_99 = np.percentile(volumes["volume"], 99)

    percentile_25_by_year = volumes.groupby(volumes["year"], as_index=False)["volume"].quantile(0.25)
    percentile_50_by_year = volumes.groupby(volumes["year"], as_index=False)["volume"].quantile(0.50)
    percentile_75_by_year = volumes.groupby(volumes["year"], as_index=False)["volume"].quantile(0.75)
    percentile_95_by_year = volumes.groupby(volumes["year"], as_index=False)["volume"].quantile(0.95)
    percentile_99_by_year = volumes.groupby(volumes["year"], as_index=False)["volume"].quantile(0.99)

    outliers_by_year = {y: volumes[(volumes["volume"] > np.percentile(volumes[volumes["year"] == y]["volume"], 75)) & (volumes["volume"] < np.percentile(volumes[volumes["year"] == y]["volume"], 25)) & (volumes["year"] == y)] for y in volumes["year"].unique()} #Return all values which are greater than the 75th percentile and that are registered in the year taken in consideration (during the for loop in the dict comprehension)


    # --------------- Insights printing ---------------

    print(f"\n\n************* Exploratory Data Analysis for TRP: {trp_id} *************")

    print("TRP ID: ", trp_id, "\n")
    print("Data shape: ", data_shape)

    print("Percentiles for the whole distribution (all years): ")
    print("25th percentile: ", percentile_25)
    print("50th percentile: ", percentile_50)
    print("75th percentile: ", percentile_75)
    print("95th percentile: ", percentile_95)
    print("99th percentile: ", percentile_99)
    print("\n")

    print("Inter-Quartile Range (IQR) for the whole distribution (all years): ", percentile_75-percentile_25)
    print("Quartile Deviation for the whole distribution (all years): ", percentile_75-percentile_25)


    print("Percentiles by year")
    print(percentile_25_by_year, "\n")
    print(percentile_50_by_year, "\n")
    print(percentile_75_by_year, "\n")
    print(percentile_95_by_year, "\n")
    print(percentile_99_by_year, "\n")
    print("\n")


    print("Number of outliers (data over the 75th percentile for its year's data) by year")
    for y in sorted(outliers_by_year.keys()): print(f"Year: {y} | Outliers: {len(outliers_by_year[y])}")

    print("\n")








    print("\n\n")


    return None























