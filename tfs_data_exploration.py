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
    print(percentile_25_by_year, "\n")
    print(percentile_50_by_year, "\n")
    print(percentile_75_by_year, "\n")
    print(percentile_95_by_year, "\n")
    print(percentile_99_by_year, "\n")
    print("\n")


    print("Number of outliers (data over the 75th percentile for its year's data) by year")
    for y in sorted(outliers_by_year.keys()): print(f"Year: {y} | Outliers: {len(outliers_by_year[y])}")

    print("\n")


    print("Volumes mean: ", np.mean(volumes["volume"]))
    print("Volumes median: ", np.mean(volumes["volume"]))
    print("Volumes standard deviation: ", np.std(volumes["volume"]))
    print("Volumes standard variance: ", np.var(volumes["volume"]))

    for y in sorted(outliers_by_year.keys()):
        print(f"Volumes mean for year {y}: ", np.mean(volumes[volumes["year"] == y]["volume"]))
        print(f"Volumes median for year {y}: ", np.mean(volumes[volumes["year"] == y]["volume"]))
        print(f"Volumes standard deviation for year {y}: ", np.std(volumes[volumes["year"] == y]["volume"]))
        print(f"Volumes standard variance for year {y}: ", np.var(volumes[volumes["year"] == y]["volume"]))

    print("\n\n")

    #Checking if the data distribution is normal
    swt_path = get_shapiro_wilk_plots_path()
    ShapiroWilkTest("volume", volumes["volume"], swt_path)




    return None























