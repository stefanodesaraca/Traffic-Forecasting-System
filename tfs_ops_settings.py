import os
from cleantext import clean

cwd = os.getcwd()

ops_folder = "ops"
os.makedirs(ops_folder, exist_ok=True) #Creating the operations folder

active_ops_filename = "active_ops"


def clean_text(text: str) -> str:

    text = clean(text, no_emoji=True, no_currency_symbols=True)
    text = text.replace(" ", "_")
    text = text.lower()

    return text


#The user sets the current operation
def write_active_ops_file(ops_name: str) -> None:

    ops_name = clean_text(ops_name)

    with open(f"{active_ops_filename}.txt", "w") as ops_file:
        ops_file.write(ops_name)

    return None


#Reading operations file, it indicates which road network we're taking into consideration
def read_active_ops_file():

    try:
        with open(f"{active_ops_filename}.txt", "r") as ops_file:
            op = ops_file.read()

    except FileNotFoundError:
        print("\033[91mOperations File Not Found\033[0m")

    return op


def del_active_ops_file() -> None:

    try:
        os.remove(f"{active_ops_filename}.txt")

    except FileNotFoundError:
        print("\033[91mCurrent Operation File Not Found\033[0m")

    return None


#If the user wants to create a new operation, this function will be called
def create_ops_folder(ops_name: str) -> None:

    ops_name = clean_text(ops_name)
    os.makedirs(f"{ops_folder}/{ops_name}", exist_ok=True)

    main_folders = [f"{ops_name}_data", f"{ops_name}_eda", f"{ops_name}_rn_graph", f"{ops_name}_ml"]
    data_subfolders = ["traffic_volumes", "average_speed", "travel_times", "trp_metadata"]
    data_sub_subfolders = ["raw", "clean"] #To isolate raw data from the clean one
    eda_subfolders = [f"{ops_name}_shapiro_wilk_test", f"{ops_name}_plots"]
    eda_sub_subfolders = ["traffic_volumes", "avg_speeds"]
    rn_graph_subfolders = [f"{ops_name}_edges", f"{ops_name}_arches", f"{ops_name}_graph_analysis", f"{ops_name}_shortest_paths"]
    ml_subfolders = ["models_parameters", "models", "models_performance", "ml_reports"]
    ml_sub_subfolders = ["traffic_volumes", "average_speed"]
    ml_sub_sub_subfolders = [road_category for road_category in ["E", "R", "F", "K", "P"]]


    for mf in main_folders:
        os.makedirs(f"{cwd}/{ops_folder}/{ops_name}/{mf}", exist_ok=True)

    # Data subfolders
    for dsf in data_subfolders:
        os.makedirs(f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_data/{dsf}", exist_ok=True)

        #Data sub-subfolders
        for dssf in data_sub_subfolders:
            if dsf != "trp_metadata": os.makedirs(f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_data/{dsf}/{dssf}_{dsf}", exist_ok=True)

    for e in eda_subfolders:
        os.makedirs(f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_eda/{e}", exist_ok=True)

        for esub in eda_sub_subfolders:
            os.makedirs(f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_eda/{e}/{esub}_eda_plots", exist_ok=True)

    # Graph subfolders
    for gsf in rn_graph_subfolders:
        os.makedirs(f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_rn_graph/{gsf}", exist_ok=True)

    # Machine learning subfolders
    for mlsf in ml_subfolders:
        os.makedirs(f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_ml/{ops_name}_{mlsf}", exist_ok=True)

        #Machine learning sub-subfolders
        for mlssf in ml_sub_subfolders:
            os.makedirs(f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_ml/{ops_name}_{mlsf}/{ops_name}_{mlssf}_{mlsf}", exist_ok=True)

            for mlsssf in ml_sub_sub_subfolders:
                os.makedirs(f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_ml/{ops_name}_{mlsf}/{ops_name}_{mlssf}_{mlsf}/{ops_name}_{mlsssf}_{mlssf}_{mlsf}", exist_ok=True)

    return None


def del_ops_folder(ops_name: str) -> None:

    try:
        os.rmdir(ops_name)
        print(f"{ops_name} Operation Folder Deleted")

    except FileNotFoundError:
        print("\033[91mOperation Folder Not Found\033[0m")


    return None
















































