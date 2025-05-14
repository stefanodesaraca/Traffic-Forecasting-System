import pandas as pd
import dask.dataframe as dd
import tfs_cleaning
import numpy as np
from numpy.linalg import eigvals
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import plotly
import plotly.express as px
import os
import inspect
from functools import wraps
from typing import Any
import traceback
import logging

from tfs_cleaning import *

tab10 = sns.color_palette("tab10")


def savePlots(plotFunction):
    def checkPlots(plotNames, plots):
        return bool(isinstance(plotNames, list) and isinstance(plots, list))

    def checkPlotsTypeAndSave(plotName, plots, filePath):
        if isinstance(plots, (plt.Figure, plt.Axes, sns.axisgrid.FacetGrid, sns.axisgrid.PairGrid, list)):
            plt.savefig(f"{filePath}{plotName}.png", dpi=300)
            print(f"{plotName} exported correctly")

        elif isinstance(plots, plotly.graph_objs._figure.Figure):
            plots.write_html(f"{filePath}{plotName}.html")
            print(f"{plotName} exported correctly")

        else:
            try:
                plt.savefig(f"{filePath}{plotName}.png", dpi=300)
                print(f"{plotName} exported correctly")
            except Exception as e:
                logging.error(traceback.format_exc())
                print(f"\033[91mExporting the plots wasn't possible, the returned type is not included in the decorator function. Error: {e}\033[0m")

        return None

    @wraps(plotFunction)
    def wrapper(*args, **kwargs):
        plotsNames, generatedPlots, filePath = plotFunction(*args, **kwargs)
        # print("File path: " + filePath)

        if checkPlots(plotsNames, generatedPlots) is True:
            for plotName, plot in zip(plotsNames, generatedPlots):
                checkPlotsTypeAndSave(plotName, plot, filePath)

        elif checkPlots(plotsNames, generatedPlots) is False:
            # print("Saving Single Graph...")
            checkPlotsTypeAndSave(plotsNames, generatedPlots, filePath)

        else:
            print(f"\033[91mExporting the plots wasn't possible, here's the data types obtained by the decorator: PlotNames: {type(plotsNames)}, Generated Plots (could be a list of plots): {type(generatedPlots)}, File Path: {type(filePath)}\033[0m")

        return None

    return wrapper


@savePlots
def ShapiroWilkTest(targetFeatureName, data, shapiroWilkPlotsPath):
    plotName = targetFeatureName + inspect.currentframe().f_code.co_name

    print(f"Shapiro-Wilk Normality test on {targetFeatureName}")
    _, SWH0PValue = stats.shapiro(data)  # Executing the Shapiro-Wilk Normality Test - This method returns a 'scipy.stats._morestats.ShapiroResult' class object with two parameters inside, the second is the H0 P-Value
    # print(type(stats.shapiro(data)))
    print(f"Normality probability (H0 Hypothesis P-Value): {SWH0PValue}")

    fig, ax = plt.subplots()
    SWQQPlot = stats.probplot(data, plot=ax)
    ax.set_title(f"Probability plot for {targetFeatureName}")

    return plotName, SWQQPlot, shapiroWilkPlotsPath


def analyze_volumes(volumes: pd.DataFrame) -> None:

    # --------------- Calculations ---------------

    trp_id = volumes["trp_id"][0]

    percentile_25 = np.percentile(volumes["volume"], 25)
    percentile_75 = np.percentile(volumes["volume"], 75)

    external_values_by_year = {
        y: volumes[(volumes["volume"] > np.percentile(volumes[volumes["year"] == y]["volume"], 75)) & (volumes["volume"] < np.percentile(volumes[volumes["year"] == y]["volume"], 25)) & (volumes["year"] == y)]
        for y in volumes["year"].unique()
    }  # Return all values which are greater than the 75th percentile and that are registered in the year taken in consideration (during the for loop in the dict comprehension)

    # --------------- Insights printing ---------------

    print(f"\n\n************* Traffic volumes - Exploratory Data Analysis for TRP: {trp_id} *************")

    print("TRP ID: ", trp_id, "\n")
    print("Data shape: ", volumes.shape, "\n")
    print("Data types: \n", volumes.dtypes, "\n")

    print("Number of negative values: ", len(volumes[volumes["volume"] < 0]))
    print("Number of zeros: ", len(volumes[volumes["volume"] == 0]))

    print("Percentiles for the whole distribution (all years): ")
    print("Traffic volume 25th percentile: ", percentile_25)
    print("Traffic volume 50th percentile: ", np.percentile(volumes["volume"], 50))
    print("Traffic volume 75th percentile: ", percentile_75)
    print("Traffic volume 95th percentile: ", np.percentile(volumes["volume"], 95))
    print("Traffic volume 99th percentile: ", np.percentile(volumes["volume"], 99))
    print("\n")

    print("Inter-Quartile Range (IQR) for the whole distribution (all years): " , percentile_75 - percentile_25)
    print("Quartile Deviation for the whole distribution (all years): " , percentile_75 - percentile_25)

    print("\nPercentiles by year")
    print(volumes.groupby(volumes["year"], as_index=False)["volume"].quantile(0.25).rename(columns={"volume": "percentile_25"}), "\n")
    print(volumes.groupby(volumes["year"], as_index=False)["volume"].quantile(0.50).rename(columns={"volume": "percentile_50"}), "\n")
    print(volumes.groupby(volumes["year"], as_index=False)["volume"].quantile(0.75).rename(columns={"volume": "percentile_75"}), "\n")
    print(volumes.groupby(volumes["year"], as_index=False)["volume"].quantile(0.95).rename(columns={"volume": "percentile_95"}), "\n")
    print(volumes.groupby(volumes["year"], as_index=False)["volume"].quantile(0.99).rename(columns={"volume": "percentile_99"}), "\n")
    print("\n")

    print("Number of external values (data over the 75th percentile for its year's data) by year:")
    for y in sorted(external_values_by_year.keys()):
        print(f"Year: {y} | Number of external values: {len(external_values_by_year[y])}")

    print("\n")

    print("Volumes mean: ", np.round(np.mean(volumes["volume"]), 2))
    print("Volumes median: ", np.round(np.median(volumes["volume"]), 2))
    print("Volumes standard deviation: ", np.round(np.std(volumes["volume"]), 2))
    print("Volumes standard variance: ", np.round(np.var(volumes["volume"]), 2))
    print("\n")

    for y in sorted(external_values_by_year.keys()):
        print(f"Volumes mean for year {y}: ", np.round(np.mean(volumes[volumes["year"] == y]["volume"]), 2))
        print(f"Volumes median for year {y}: ", np.round(np.median(volumes[volumes["year"] == y]["volume"]), 2))
        print(f"Volumes standard deviation for year {y}: ", np.round(np.std(volumes[volumes["year"] == y]["volume"]), 2))
        print(f"Volumes variance for year {y}: ", np.round(np.var(volumes[volumes["year"] == y]["volume"]), 2))
        print("\n")

    # Checking if the data distribution is normal
    ShapiroWilkTest("volume", volumes["volume"], read_metainfo_key(keys_map=["folder_paths", "eda", "traffic_shapiro_wilk_test", "path"]))
    plt.clf()

    print("\n\n")

    print("Traffic volumes - Correlations between variables:")
    print("By hour: \n", np.corrcoef(volumes["volume"], volumes["hour"]), "\n")
    print("By day: \n", np.corrcoef(volumes["volume"], volumes["day"]), "\n")
    print("By week: \n", np.corrcoef(volumes["volume"], volumes["week"]), "\n")
    print("By month: \n", np.corrcoef(volumes["volume"], volumes["month"]), "\n")
    print("By year: \n", np.corrcoef(volumes["volume"], volumes["year"]), "\n")

    print("Traffic volumes - Correlations dataframe-wise (numerical variables only): ")
    print(volumes.corr(numeric_only=True), "\n")

    volumes["date"] = pd.to_datetime([f"{y}-{m}-{d}" for y, m, d in zip(volumes["year"], volumes["month"], volumes["day"], strict=True)]) #TODO WE ALREADY HAVE THE "DATE" COLUMN RIGHT? WHY IS THIS STILL HERE. I'LL CHECK BETTER

    @savePlots
    def volume_trend_grouped_by_years():

        plt.figure(figsize=(16, 9))
        for y in sorted(volumes["year"].unique()):
            year_data = volumes[volumes["year"] == y].groupby("date", as_index=False)["volume"].sum().sort_values(by="date", ascending=True)
            # print(year_data)
            plt.plot(range(0, len(year_data)), "volume", data=year_data, marker="o")  # To make the plots overlap they must have the same exact data on the x axis.
            # In this case for example they must have the same days number on the x axis so that matplotlib know where to plot the data and thus, this can overlap too if if has the same x axis value

        plt.grid()
        plt.ylabel("Volume")
        plt.xlabel("Time (days)")
        plt.legend(labels=sorted(volumes["year"].unique()), loc="upper right")
        plt.title(f"Traffic volumes aggregated (sum) by day for different years | TRP: {trp_id}")

        return f"{trp_id}_volume_trend_grouped_by_years", plt, read_metainfo_key(keys_map=["folder_paths", "eda", f"{get_active_ops()}_plots", "subfolders", "traffic_volumes", "path"])

    @savePlots
    def volume_trend_by_week():

        plt.figure(figsize=(16, 9))
        for y in sorted(volumes["year"].unique()):
            week_data = volumes[volumes["year"] == y][["volume", "year", "week"]].groupby(["week"], as_index=False)["volume"].median().sort_values(by="week", ascending=True)

            # print(week_data)
            plt.plot(range(0, len(week_data)), "volume", data=week_data, marker="o")

        plt.grid()
        plt.ylabel("Volume")
        plt.xlabel("Week")
        plt.legend(labels=sorted(volumes["year"].unique()), loc="upper right")
        plt.title(f"Traffic volumes aggregated (median) by week for different years | TRP: {trp_id}")

        return f"{trp_id}_volume_trend_by_hour_day", plt, read_metainfo_key(keys_map=["folder_paths", "eda", f"{get_active_ops()}_plots", "subfolders", "traffic_volumes", "path"])

    @savePlots
    def volumes_distribution_by_week_and_year():

        fig, axs = plt.subplots(len(volumes["year"].unique()), 1, figsize=(16, 9))
        plt.suptitle(f"{trp_id} Volumes distribution by week and year")

        for idx, y in enumerate(sorted(volumes["year"].unique())):
            for w in sorted(volumes[volumes["year"] == y]["week"].unique()):
                volumes_grouped = volumes[(volumes["year"] == y) & (volumes["week"] == w)]
                # print(volumes_grouped)

                axs[idx].boxplot(x=volumes_grouped["volume"], positions=[w])
                axs[idx].set_ylabel("Volumes")
                axs[idx].set_xlabel("Weeks")
                axs[idx].set_title(f"Traffic volumes distribution by week for {y}")

        plt.subplots_adjust(hspace=0.5)

        return f"{trp_id}_volumes_distribution_by_week_and_year", plt, read_metainfo_key(keys_map=["folder_paths", "eda", f"{get_active_ops()}_plots", "subfolders", "traffic_volumes", "path"])

    @savePlots
    def correlation_heatmap():
        return (f"{trp_id}_volumes_corr_heatmap",
                sns.heatmap(volumes.corr(numeric_only=True), annot=True, fmt=".2f").set_title(f"Traffic volumes - TRP: {trp_id} - Correlation heatmap"),
                read_metainfo_key(keys_map=["folder_paths", "eda", f"{get_active_ops()}_plots", "traffic_volumes_eda_plots", "path"]))

    all((i(), plt.clf()) for i in (volume_trend_grouped_by_years, volume_trend_by_week, volumes_distribution_by_week_and_year, correlation_heatmap))

    return None


def analyze_avg_speeds(speeds: pd.DataFrame) -> None:

    # --------------- Calculations ---------------

    trp_id = speeds["trp_id"][0]

    percentile_25 = np.percentile(speeds["mean_speed"], 25)
    percentile_75 = np.percentile(speeds["mean_speed"], 75)

    external_values_by_year = {
        y: speeds[(speeds["mean_speed"] > np.percentile(speeds[speeds["year"] == y]["mean_speed"], 75)) & (speeds["mean_speed"] < np.percentile(speeds[speeds["year"] == y]["mean_speed"], 25)) & (speeds["year"] == y)]
        for y in speeds["year"].unique()
    }  # Return all values which are greater than the 75th percentile and that are registered in the year taken in consideration (during the for loop in the dict comprehension)

    # --------------- Insights printing ---------------

    print(f"\n\n************* Average speeds - Exploratory Data Analysis for TRP: {trp_id} *************")

    print("TRP ID: ", trp_id, "\n")
    print("Data shape: ", speeds.shape, "\n")
    print("Data types: \n", speeds.dtypes, "\n")

    print("Percentiles for the whole distribution (all years): ")
    print("Average speed 25th percentile: ", percentile_25)
    print("Average speed 50th percentile: ", np.percentile(speeds["mean_speed"], 50))
    print("Average speed 75th percentile: ", percentile_75)
    print("Average speed 95th percentile: ", np.percentile(speeds["mean_speed"], 95))
    print("Average speed 99th percentile: ", np.percentile(speeds["mean_speed"], 99))
    print("\n")
    print("Number of negative values: ", len(speeds[speeds["mean_speed"] < 0]))
    print("Number of zeros: ", len(speeds[speeds["mean_speed"] == 0]))

    print("Inter-Quartile Range (IQR) for the whole distribution (all years): ", percentile_75 - percentile_25)
    print("Quartile Deviation for the whole distribution (all years): ", percentile_75 - percentile_25)

    print("\nPercentiles by year")
    print(speeds.groupby(speeds["year"], as_index=False)["mean_speed"].quantile(0.25).rename(columns={"mean_speed": "percentile_25"}), "\n")
    print(speeds.groupby(speeds["year"], as_index=False)["mean_speed"].quantile(0.50).rename(columns={"mean_speed": "percentile_50"}), "\n")
    print(speeds.groupby(speeds["year"], as_index=False)["mean_speed"].quantile(0.75).rename(columns={"mean_speed": "percentile_75"}), "\n")
    print(speeds.groupby(speeds["year"], as_index=False)["mean_speed"].quantile(0.95).rename(columns={"mean_speed": "percentile_95"}), "\n")
    print(speeds.groupby(speeds["year"], as_index=False)["mean_speed"].quantile(0.99).rename(columns={"mean_speed": "percentile_99"}), "\n")
    print("\n")

    print("Number of external values (data over the 75th percentile for its year's data) by year:")
    for y in sorted(external_values_by_year.keys()):
        print(f"Year: {y} | Number of external values: {len(external_values_by_year[y])}")

    print("\n")

    print("Average speeds mean: ", np.round(np.mean(speeds["mean_speed"]), 2))
    print("Average speeds median: ", np.round(np.median(speeds["mean_speed"]), 2))
    print("Average speeds standard deviation: ", np.round(np.std(speeds["mean_speed"]), 2))
    print("Average speeds variance: ", np.round(np.var(speeds["mean_speed"]), 2))
    print("\n")

    for y in sorted(external_values_by_year.keys()):
        print(f"Average speeds mean for year {y}: ", np.round(np.mean(speeds[speeds["year"] == y]["mean_speed"]), 2),)
        print(f"Average speeds median for year {y}: ", np.round(np.median(speeds[speeds["year"] == y]["mean_speed"]), 2))
        print(f"Average speeds standard deviation for year {y}: ", np.round(np.std(speeds[speeds["year"] == y]["mean_speed"]), 2))
        print(f"Average speeds standard variance for year {y}: ", np.round(np.var(speeds[speeds["year"] == y]["mean_speed"]), 2))
        print("\n")

    # Checking if the data distribution is normal
    swt_path = read_metainfo_key(keys_map=["folder_paths", "eda", "traffic_shapiro_wilk_test", "path"])
    ShapiroWilkTest("mean_speed", speeds["mean_speed"], swt_path)
    plt.clf()

    print("\n\n")

    print("Average speeds - Correlations between variables:")
    print("By hour: \n", np.corrcoef(speeds["mean_speed"], speeds["hour_start"]), "\n")
    print("By day: \n", np.corrcoef(speeds["mean_speed"], speeds["day"]), "\n")
    print("By week: \n", np.corrcoef(speeds["mean_speed"], speeds["week"]), "\n")
    print("By month: \n", np.corrcoef(speeds["mean_speed"], speeds["month"]), "\n")
    print("By year: \n", np.corrcoef(speeds["mean_speed"], speeds["year"]), "\n")

    print("Average speeds - Correlations dataframe-wise (numerical variables only): ")
    print(speeds.corr(numeric_only=True), "\n")

    @savePlots
    def speeds_trend_grouped_by_years():

        plt.figure(figsize=(16, 9))
        for y in sorted(speeds["year"].unique()):
            year_data = speeds[speeds["year"] == y].groupby("date", as_index=False)["mean_speed"].mean().sort_values(by="date", ascending=True)
            # print(year_data)
            plt.plot(range(0, len(year_data)), "mean_speed", data=year_data, marker="o")  # To make the plots overlap they must have the same exact data on the x axis.

        plt.grid()
        plt.ylabel("Average speed")
        plt.xlabel("Time (days)")
        plt.legend(labels=sorted(speeds["year"].unique()), loc="upper right")
        plt.title(f"Average speeds aggregated by day for different years | TRP: {trp_id}")

        return f"{trp_id}_avg_speeds_trend_grouped_by_years", plt, read_metainfo_key(keys_map=["folder_paths", "eda", f"{get_active_ops()}_plots", "subfolders", "avg_speeds", "path"])

    @savePlots
    def speeds_trend_by_week():

        plt.figure(figsize=(16, 9))
        for y in sorted(speeds["year"].unique()):
            week_data = speeds[speeds["year"] == y][["mean_speed", "year", "week"]].groupby(["week"], as_index=False)["mean_speed"].median().sort_values(by="week", ascending=True)

            plt.plot(range(0, len(week_data)), "mean_speed", data=week_data, marker="o")

        plt.grid()
        plt.ylabel("Median of the average speed")
        plt.xlabel("Week")
        plt.legend(labels=sorted(speeds["year"].unique()), loc="upper right")
        plt.title(f"Median of the average speeds by week for different years | TRP: {trp_id}")

        return f"{trp_id}_avg_speed_trend_by_hour_day", plt, read_metainfo_key(keys_map=["folder_paths", "eda", f"{get_active_ops()}_plots", "subfolders", "avg_speeds", "path"])

    @savePlots
    def speeds_distribution_by_week_and_year():

        fig, axs = plt.subplots(len(speeds["year"].unique()), 1, figsize=(16, 9))
        plt.suptitle(f"{trp_id} speeds distribution by week and year")

        for idx, y in enumerate(sorted(speeds["year"].unique())):
            for w in sorted(speeds[speeds["year"] == y]["week"].unique()):
                speeds_grouped = speeds[(speeds["year"] == y) & (speeds["week"] == w)]
                # print(speeds_grouped)

                axs[idx].boxplot(x=speeds_grouped["mean_speed"], positions=[w])
                axs[idx].set_ylabel("Average speed")
                axs[idx].set_xlabel("Weeks")
                axs[idx].set_title(f"Average speed distribution by week for {y}")

        plt.subplots_adjust(hspace=0.5)

        return f"{trp_id}_avg_speed_distribution_by_week_and_year", plt, read_metainfo_key(keys_map=["folder_paths", "eda", f"{get_active_ops()}_plots", "subfolders", "avg_speeds", "path"])

    @savePlots
    def correlation_heatmap():
        return (f"{trp_id}_avg_speed_corr_heatmap",
                sns.heatmap(speeds.corr(numeric_only=True), annot=True, fmt=".2f").set_title(f"Traffic volumes - TRP: {trp_id} - Correlation heatmap"),
                read_metainfo_key(keys_map=["folder_paths", "eda", f"{get_active_ops()}_plots", "avg_speeds_eda_plots"]))

    all((i(), plt.clf()) for i in (speeds_trend_grouped_by_years, speeds_trend_by_week, speeds_distribution_by_week_and_year, correlation_heatmap))

    return None


def volumes_data_multicollinearity_test(volumes: pd.DataFrame) -> None:
    volumes = volumes.drop(columns=["volume", "date", "trp_id"])
    volumes_col_names = list(volumes.columns)
    # print(volumes_col_names)

    print()

    # ----------------- VIF -----------------

    volumes_vif = {}

    for i in range(len(volumes_col_names)):
        y = volumes.iloc[:, volumes.columns == volumes_col_names[i]]
        X = volumes.iloc[:, volumes.columns != volumes_col_names[i]]

        model = sm.OLS(y, X)

        model_results = model.fit()
        r2 = model_results.rsquared

        vif = round(1 / (1 - r2), 2)

        volumes_vif[volumes_col_names[i]] = vif

        print(f"R^2 value of variable: {volumes_col_names[i]} = ", r2)
        print(f"VIF (Variance Inflation Factor) value of variable: {volumes_col_names[i]} = ", vif, "\n")

    print("----------------- Traffic volumes - VIFs -----------------")

    pprint.pprint(volumes_vif)
    print()

    # ----------------- Condition index -----------------

    volumes_corr_matrix = volumes.corr()

    eigenvalues = eigvals(volumes_corr_matrix)
    condition_index = max(eigenvalues) / min(eigenvalues)

    print(f"Traffic volumes - Condition Index: {condition_index}\n\n")

    return None


def avg_speeds_data_multicollinearity_test(speeds: pd.DataFrame) -> None:
    speeds = speeds.drop(columns=["mean_speed", "percentile_85", "trp_id", "date"], axis=1)
    speeds_col_names = list(speeds.columns)
    # print(speeds_col_names)

    speeds_vif = {}

    for i in range(len(speeds_col_names)):
        y = speeds.iloc[:, speeds.columns == speeds_col_names[i]]
        X = speeds.iloc[:, speeds.columns != speeds_col_names[i]]

        model = sm.OLS(y, X)

        model_results = model.fit()
        r2 = model_results.rsquared

        vif = round(1 / (1 - r2), 2)

        speeds_vif[speeds_col_names[i]] = vif

        print(f"R^2 value of variable: {speeds_col_names[i]} = ", r2)
        print(f"VIF (Variance Inflation Factor) value of variable: {speeds_col_names[i]} = ", vif, "\n")

    print("----------------- Average speeds - VIFs -----------------")

    pprint.pprint(speeds_vif)
    print()

    # ----------------- Condition index -----------------

    speeds_corr_matrix = speeds.corr()

    eigenvalues = eigvals(speeds_corr_matrix)
    condition_index = max(eigenvalues) / min(eigenvalues)

    print(f"Average speeds - Condition Index: {condition_index}\n\n")

    return None


# TODO ANALYZE TRAFFIC VOLUMES AND AVERAGE SPEED COMBINED (ONLY FOR THE TRPs WHICH HAVE BOTH FOR THE SAME TIME PERIOD

# TODO VERIFY THAT DATES STORED IN THE AVG SPEED FILES ARE "%Y-%m-%d" FORMAT AND NOT WITH / BETWEEN THE Y, M AND D
