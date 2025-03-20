import pandas as pd
from scipy.ndimage import rotate

import tfs_cleaning
from tfs_ops_settings import *
from tfs_cleaning import *
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import dask as dd
from scipy import stats
from scipy.stats import ttest_ind
import plotly
import plotly.express as px
import os
import inspect
from functools import wraps


def savePlots(plotFunction):

    def checkPlots(plotNames, plots):
        if isinstance(plotNames, list) and isinstance(plots, list):
            return True
        else:
            #print("\033[91mCheckPlots: object obtained are not lists\033[0m")
            return False

    def checkPlotsTypeAndSave(plotName, plots, filePath):
        if isinstance(plots, (plt.Figure, plt.Axes, sns.axisgrid.FacetGrid, sns.axisgrid.PairGrid, list)):
            plt.savefig(f"{filePath}{plotName}.png", dpi=300)
            print(f"{plotName} Exported Correctly")

        elif isinstance(plots, plotly.graph_objs._figure.Figure):
            plots.write_html(f"{filePath}{plotName}.html")
            print(f"{plotName} Exported Correctly")

        else:
            try:
                plt.savefig(f"{filePath}{plotName}.png", dpi=300)
                print(f"{plotName} Exported Correctly")
            except:
                print("\033[91mExporting the plots wasn't possible, the returned type is not included in the decorator function\033[0m")

        return None

    @wraps(plotFunction)
    def wrapper(*args, **kwargs):

        plotsNames, generatedPlots, filePath = plotFunction(*args, **kwargs)
        #print("File path: " + filePath)

        if checkPlots(plotsNames, generatedPlots) is True:

            for plotName, plot in zip(plotsNames, generatedPlots):
                checkPlotsTypeAndSave(plotName, plot, filePath)

        elif checkPlots(plotsNames, generatedPlots) is False:
            #print("Saving Single Graph...")
            checkPlotsTypeAndSave(plotsNames, generatedPlots, filePath)

        else:
            print(f"\033[91mExporting the plots wasn't possible, here's the data types obtained by the decorator: PlotNames: {type(plotsNames)}, Generated Plots (could be a list of plots): {type(generatedPlots)}, File Path: {type(filePath)}\033[0m")

        return None

    return wrapper


@savePlots
def ShapiroWilkTest(targetFeatureName, data, shapiroWilkPlotsPath):

    plotName = targetFeatureName + inspect.currentframe().f_code.co_name

    print(f"Shapiro-Wilk Normality test on {targetFeatureName}")
    _, SWH0PValue = stats.shapiro(data) #Executing the Shapiro-Wilk Normality Test - This method returns a 'scipy.stats._morestats.ShapiroResult' class object with two parameters inside, the second is the H0 P-Value
    #print(type(stats.shapiro(data)))
    print(f"Normality probability (H0 Hypothesis P-Value): {SWH0PValue}")

    fig, ax = plt.subplots()
    SWQQPlot = stats.probplot(data, plot=ax)
    ax.set_title(f"Probability plot for {targetFeatureName}")

    return plotName, SWQQPlot, shapiroWilkPlotsPath


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
    print(percentile_25_by_year.rename(columns={"volume": "percentile_25"}), "\n")
    print(percentile_50_by_year.rename(columns={"volume": "percentile_50"}), "\n")
    print(percentile_75_by_year.rename(columns={"volume": "percentile_75"}), "\n")
    print(percentile_95_by_year.rename(columns={"volume": "percentile_95"}), "\n")
    print(percentile_99_by_year.rename(columns={"volume": "percentile_99"}), "\n")
    print("\n")


    print("Number of outliers (data over the 75th percentile for its year's data) by year:")
    for y in sorted(outliers_by_year.keys()): print(f"Year: {y} | Number of Outliers: {len(outliers_by_year[y])}")

    print("\n")


    print("Volumes mean: ", np.round(np.mean(volumes["volume"]), 2))
    print("Volumes median: ", np.round(np.mean(volumes["volume"]), 2))
    print("Volumes standard deviation: ", np.round(np.std(volumes["volume"]), 2))
    print("Volumes standard variance: ", np.round(np.var(volumes["volume"]), 2))
    print("\n")


    for y in sorted(outliers_by_year.keys()):
        print(f"Volumes mean for year {y}: ", np.round(np.mean(volumes[volumes["year"] == y]["volume"]), 2))
        print(f"Volumes median for year {y}: ", np.round(np.mean(volumes[volumes["year"] == y]["volume"]), 2))
        print(f"Volumes standard deviation for year {y}: ", np.round(np.std(volumes[volumes["year"] == y]["volume"]), 2))
        print(f"Volumes standard variance for year {y}: ", np.round(np.var(volumes[volumes["year"] == y]["volume"]), 2))
        print("\n")

    #Checking if the data distribution is normal
    swt_path = get_shapiro_wilk_plots_path()
    ShapiroWilkTest("volume", volumes["volume"], swt_path)

    plt.clf()

    print("\n\n")


    volume_hour_corr = np.corrcoef(volumes["volume"], volumes["hour"])
    volume_day_corr = np.corrcoef(volumes["volume"], volumes["day"])
    volume_week_corr = np.corrcoef(volumes["volume"], volumes["week"])
    volume_month_corr = np.corrcoef(volumes["volume"], volumes["month"])
    volume_year_corr = np.corrcoef(volumes["volume"], volumes["year"])


    print("Correlations between variables:")
    print("By hour: \n", volume_hour_corr, "\n")
    print("By day: \n", volume_day_corr, "\n")
    print("By week: \n", volume_week_corr, "\n")
    print("By month: \n", volume_month_corr, "\n")
    print("By year: \n", volume_year_corr, "\n")


    dates = [f"{y}-{m}-{d}" for y, m, d in zip(volumes["year"], volumes["month"], volumes["day"])]

    volumes["date"] = pd.to_datetime(dates)



    @savePlots
    def volume_trend_grouped_by_years():

        plot_path = get_eda_plots_folder_path()

        plt.figure(figsize=(16,9))

        for y in sorted(volumes["year"].unique()):

            year_data = volumes[volumes["year"] == y].groupby("date", as_index=False)["volume"].sum().sort_values(by="date", ascending=True)
            print(year_data)
            plt.plot(range(0, len(year_data)), "volume", data=year_data) #To make the plots overlap they must have the same exact data on the x axis.
            #In this case for example they must have the same days number on the x axis so that matplotlib know where to plot the data and thus, this can overlap too if if has the same x axis value

        plt.ylabel("Volume")
        plt.xlabel("Time (days)")

        plt.grid()


        return f"{trp_id}_volume_trend_grouped_by_years", plt, plot_path


    volume_trend_grouped_by_years()
    plt.clf()











    return None























