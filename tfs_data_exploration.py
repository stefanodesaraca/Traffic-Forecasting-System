import pandas as pd
import tfs_cleaning
from tfs_ops_settings import *
from tfs_cleaning import *
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
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


def retrieve_avg_speed_data(file_path: str):

    speeds = pd.read_csv(file_path)

    return speeds


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

    print(f"\n\n************* Traffic volumes - Exploratory Data Analysis for TRP: {trp_id} *************")

    print("TRP ID: ", trp_id, "\n")
    print("Data shape: ", data_shape)

    print("Percentiles for the whole distribution (all years): ")
    print("Traffic volume 25th percentile: ", percentile_25)
    print("Traffic volume 50th percentile: ", percentile_50)
    print("Traffic volume 75th percentile: ", percentile_75)
    print("Traffic volume 95th percentile: ", percentile_95)
    print("Traffic volume 99th percentile: ", percentile_99)
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


    print("Traffic volumes - Correlations between variables:")
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
            #print(year_data)
            plt.plot(range(0, len(year_data)), "volume", data=year_data) #To make the plots overlap they must have the same exact data on the x axis.
            #In this case for example they must have the same days number on the x axis so that matplotlib know where to plot the data and thus, this can overlap too if if has the same x axis value

        plt.ylabel("Volume")
        plt.xlabel("Time (days)")

        plt.legend(labels=sorted(volumes["year"].unique()), loc="upper right")

        plt.title(f"Traffic volumes aggregated by day for different years | TRP: {trp_id}")

        plt.grid(axis="y")


        return f"{trp_id}_volume_trend_grouped_by_years", plt, plot_path


    @savePlots
    def volume_trend_by_week():

        plot_path = get_eda_plots_folder_path()

        plt.figure(figsize=(16, 9))

        for y in sorted(volumes["year"].unique()):
            week_data = volumes[volumes["year"] == y][["volume", "year", "week"]].groupby(["week"], as_index=False)["volume"].median().sort_values(by="week", ascending=True)

            #print(week_data)
            plt.plot(range(0, len(week_data)), "volume", data=week_data)

        plt.ylabel("Volume")
        plt.xlabel("Week")


        plt.legend(labels=sorted(volumes["year"].unique()), loc="upper right")

        plt.title(f"Median traffic volumes aggregated by week for different years | TRP: {trp_id}")

        plt.grid(axis="y")


        return f"{trp_id}_volume_trend_by_hour_day", plt, plot_path


    @savePlots
    def volumes_distribution_by_week_and_year():

        plot_path = get_eda_plots_folder_path()


        fig, axs = plt.subplots(len(volumes["year"].unique()), 1, figsize=(16, 9))
        plt.suptitle(f"{trp_id} Volumes distribution by week and year")

        for idx, y in enumerate(sorted(volumes["year"].unique())):

            for w in sorted(volumes[volumes["year"] == y]["week"].unique()):

                volumes_grouped = volumes[(volumes["year"] == y) & (volumes["week"] == w)]
                #print(volumes_grouped)

                axs[idx].boxplot(x=volumes_grouped["volume"], positions=[w])
                axs[idx].set_ylabel("Volumes")
                axs[idx].set_xlabel("Weeks")
                axs[idx].set_title(f"Traffic volumes distribution by week for {y}")

        plt.subplots_adjust(hspace=0.5)


        return f"{trp_id}_volumes_distribution_by_week_and_year", plt, plot_path


    plots_list = [volume_trend_grouped_by_years, volume_trend_by_week, volumes_distribution_by_week_and_year]
    all((plots_list[i](), plt.clf()) for i in range(len(plots_list)))


    return None


def analyze_avg_speeds(speeds: pd.DataFrame):

    print(speeds.head(10))

    # --------------- Calculations ---------------

    trp_id = speeds["trp_id"][0]
    data_shape = speeds.shape

    percentile_25 = np.percentile(speeds["mean_speed"], 25)
    percentile_50 = np.percentile(speeds["mean_speed"], 50)
    percentile_75 = np.percentile(speeds["mean_speed"], 75)
    percentile_95 = np.percentile(speeds["mean_speed"], 95)
    percentile_99 = np.percentile(speeds["mean_speed"], 99)

    percentile_25_by_year = speeds.groupby(speeds["year"], as_index=False)["mean_speed"].quantile(0.25)
    percentile_50_by_year = speeds.groupby(speeds["year"], as_index=False)["mean_speed"].quantile(0.50)
    percentile_75_by_year = speeds.groupby(speeds["year"], as_index=False)["mean_speed"].quantile(0.75)
    percentile_95_by_year = speeds.groupby(speeds["year"], as_index=False)["mean_speed"].quantile(0.95)
    percentile_99_by_year = speeds.groupby(speeds["year"], as_index=False)["mean_speed"].quantile(0.99)

    outliers_by_year = {y: speeds[(speeds["mean_speed"] > np.percentile(speeds[speeds["year"] == y]["mean_speed"], 75)) & (speeds["volume"] < np.percentile(speeds[speeds["year"] == y]["mean_speed"], 25)) & (speeds["year"] == y)] for y in speeds["year"].unique()}  # Return all values which are greater than the 75th percentile and that are registered in the year taken in consideration (during the for loop in the dict comprehension)


    # --------------- Insights printing ---------------

    print(f"\n\n************* Average speeds - Exploratory Data Analysis for TRP: {trp_id} *************")

    print("TRP ID: ", trp_id, "\n")
    print("Data shape: ", data_shape)

    print("Percentiles for the whole distribution (all years): ")
    print("Average speed 25th percentile: ", percentile_25)
    print("Average speed 50th percentile: ", percentile_50)
    print("Average speed 75th percentile: ", percentile_75)
    print("Average speed 95th percentile: ", percentile_95)
    print("Average speed 99th percentile: ", percentile_99)
    print("\n")

    print("Inter-Quartile Range (IQR) for the whole distribution (all years): ", percentile_75 - percentile_25)
    print("Quartile Deviation for the whole distribution (all years): ", percentile_75 - percentile_25)

    print("Percentiles by year")
    print(percentile_25_by_year.rename(columns={"mean_speed": "percentile_25"}), "\n")
    print(percentile_50_by_year.rename(columns={"mean_speed": "percentile_50"}), "\n")
    print(percentile_75_by_year.rename(columns={"mean_speed": "percentile_75"}), "\n")
    print(percentile_95_by_year.rename(columns={"mean_speed": "percentile_95"}), "\n")
    print(percentile_99_by_year.rename(columns={"mean_speed": "percentile_99"}), "\n")
    print("\n")

    print("Number of outliers (data over the 75th percentile for its year's data) by year:")
    for y in sorted(outliers_by_year.keys()): print(f"Year: {y} | Number of Outliers: {len(outliers_by_year[y])}")

    print("\n")


    print("Average speeds mean: ", np.round(np.mean(speeds["mean_speed"]), 2))
    print("Average speeds median: ", np.round(np.mean(speeds["mean_speed"]), 2))
    print("Average speeds standard deviation: ", np.round(np.std(speeds["mean_speed"]), 2))
    print("Average speeds standard variance: ", np.round(np.var(speeds["mean_speed"]), 2))
    print("\n")

    for y in sorted(outliers_by_year.keys()):
        print(f"Average speeds mean for year {y}: ", np.round(np.mean(speeds[speeds["year"] == y]["mean_speed"]), 2))
        print(f"Average speeds median for year {y}: ", np.round(np.mean(speeds[speeds["year"] == y]["mean_speed"]), 2))
        print(f"Average speeds standard deviation for year {y}: ", np.round(np.std(speeds[speeds["year"] == y]["volume"]), 2))
        print(f"Average speeds standard variance for year {y}: ", np.round(np.var(speeds[speeds["year"] == y]["mean_speed"]), 2))
        print("\n")

    # Checking if the data distribution is normal
    swt_path = get_shapiro_wilk_plots_path()
    ShapiroWilkTest("mean_speed", speeds["mean_speed"], swt_path)

    plt.clf()

    print("\n\n")

    volume_hour_corr = np.corrcoef(speeds["mean_speed"], speeds["hour"])
    volume_day_corr = np.corrcoef(speeds["mean_speed"], speeds["day"])
    volume_week_corr = np.corrcoef(speeds["mean_speed"], speeds["week"])
    volume_month_corr = np.corrcoef(speeds["mean_speed"], speeds["month"])
    volume_year_corr = np.corrcoef(speeds["mean_speed"], speeds["year"])

    print("Average speeds - Correlations between variables:")
    print("By hour: \n", volume_hour_corr, "\n")
    print("By day: \n", volume_day_corr, "\n")
    print("By week: \n", volume_week_corr, "\n")
    print("By month: \n", volume_month_corr, "\n")
    print("By year: \n", volume_year_corr, "\n")

































    return None








#TODO VERIFY THAT DATES STORED IN THE AVG SPEED FILES ARE "%Y-%m-%d" FORMAT AND NOT WITH / BETWEEN THE Y, M AND D











