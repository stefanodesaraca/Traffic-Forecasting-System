import os
from cleantext import clean

cwd = os.getcwd()

ops_folder = "ops"
os.makedirs(ops_folder, exist_ok=True) #Creating the operations folder

current_ops_filename = "current_ops"

#The user sets the current operation
def write_current_ops_file(ops_name: str):

    ops_name = clean(ops_name, to_ascii=True, no_emoji=True)

    with open(f"{current_ops_filename}.txt", "w") as ops_file:
        ops_file.write(ops_name)

    return None


#Reading operations file, it indicates which road network we're taking into consideration
def read_current_ops_file():

    try:
        with open(f"{current_ops_filename}.txt", "r") as ops_file:
            op = ops_file.read()

    except FileNotFoundError:
        print("\033[91mOperations File Not Found\033[0m")

    return op


#If the user wants to create a new operation, this function will be called
def create_ops_folder(ops_name: str):

    ops_name = clean(ops_name, to_ascii=True, no_emoji=True)
    os.makedirs(f"{ops_folder}/{ops_name}", exist_ok=True)

    return None


def del_ops_folder(ops_name: str):

    try:
        os.rmdir(ops_name)
        print(f"{ops_name} Operation Folder Deleted")

    except FileNotFoundError:
        print("\033[91mOperation Folder Not Found\033[0m")


    return None


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














































