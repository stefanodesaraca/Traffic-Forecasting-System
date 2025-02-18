from OpsSettings import read_current_ops_file
import os

cwd = os.getcwd()
ops_folder = "ops"

#This function creates the traffic forecasting system folders for the current operation
def create_tfs_folders():

    current_ops = read_current_ops_file()

    main_folders = [f"{current_ops}_data", f"{current_ops}_eda", f"{current_ops}_rn_graph", f"{current_ops}_ml"]
    data_subfolders = ["traffic_volumes", "average_speed", "travel_times"]
    rn_graph_subfolders = [f"{current_ops}_edges", f"{current_ops}_arches", f"{current_ops}_graph_analysis", f"{current_ops}_shortest_paths"]
    ml_subfolders = [f"{current_ops}_models", f"{current_ops}_models_performance", f"{current_ops}_ml_reports"]


    for mf in main_folders:
        os.makedirs(f"{cwd}/{ops_folder}/{current_ops}/{mf}", exist_ok=True)

    #Data subfolders
    for dsf in data_subfolders:
        os.makedirs(f"{cwd}/{ops_folder}/{current_ops}/{current_ops}_data/{dsf}", exist_ok=True)

    #Graph subfolders
    for gsf in rn_graph_subfolders:
        os.makedirs(f"{cwd}/{ops_folder}/{current_ops}/{current_ops}_rn_graph/{gsf}", exist_ok=True)

    #Machine learning subfolders
    for mlsf in ml_subfolders:
        os.makedirs(f"{cwd}/{ops_folder}/{current_ops}/{current_ops}_ml/{mlsf}", exist_ok=True)


    return None





































