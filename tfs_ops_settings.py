from xml.etree.ElementTree import indent

from tfs_utils import *
import os
from cleantext import clean
import json

cwd = os.getcwd()
ops_folder = "ops"
os.makedirs(ops_folder, exist_ok=True) #Creating the operations folder
active_ops_filename = "active_ops"
metainfo_filename = "metainfo"


def clean_text(text: str) -> str:
    text = clean(text, no_emoji=True, no_currency_symbols=True)
    text = text.replace(" ", "_")
    text = text.lower()
    return text


#The user sets the current operation
def write_active_ops_file(ops_name: str) -> None:
    ops_name = clean_text(ops_name)
    assert os.path.isfile(f"{ops_folder}/{ops_name}") is True, f"{ops_name} operation folder not found. Create an operation with that name first."
    with open(f"{active_ops_filename}.txt", "w") as ops_file: ops_file.write(ops_name)
    return None


#Reading operations file, it indicates which road network we're taking into consideration
def read_active_ops_file():
    try:
        with open(f"{active_ops_filename}.txt", "r") as ops_file: op = ops_file.read()
        return op
    except FileNotFoundError:
        print("\033[91mOperations File Not Found\033[0m")
        exit(code=1)


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

    write_metainfo(ops_name)

    main_folders = [f"data", f"eda", f"rn_graph", f"ml"]
    data_subfolders = ["traffic_volumes", "average_speed", "travel_times", "trp_metadata"]
    data_sub_subfolders = ["raw", "clean"] #To isolate raw data from the clean one
    eda_subfolders = [f"{ops_name}_shapiro_wilk_test", f"{ops_name}_plots"]
    eda_sub_subfolders = ["traffic_volumes", "avg_speeds"]
    rn_graph_subfolders = [f"{ops_name}_edges", f"{ops_name}_arches", f"{ops_name}_graph_analysis", f"{ops_name}_shortest_paths"]
    ml_subfolders = ["models_parameters", "models", "models_performance", "ml_reports"]
    ml_sub_subfolders = ["traffic_volumes", "average_speed"]
    ml_sub_sub_subfolders = [road_category for road_category in ["E", "R", "F", "K", "P"]]

    with open(f"{ops_folder}/{ops_name}/{metainfo_filename}.json", "r") as m: metainfo = json.load(m)
    metainfo["folder_paths"] = {} #Setting/resetting the folders path dictionary to either write it for the first time or reset the previous one to adapt it with new updated folders, paths, etc.

    for mf in main_folders:
        main_f = f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_{mf}"
        os.makedirs(main_f, exist_ok=True)
        metainfo["folder_paths"][mf] = {}

    # Data subfolders
    for dsf in data_subfolders:
        data_sub = f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_data/{dsf}"
        os.makedirs(data_sub, exist_ok=True)
        metainfo["folder_paths"]["data"][dsf] = {"path": data_sub,
                                                "subfolders": {}}

        #Data sub-subfolders
        for dssf in data_sub_subfolders:
            if dsf != "trp_metadata":
                data_2sub = f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_data/{dsf}/{dssf}_{dsf}"
                os.makedirs(data_2sub, exist_ok=True)
                metainfo["folder_paths"]["data"][dsf]["subfolders"][dssf] = {"path": data_2sub}

    for e in eda_subfolders:
        eda_sub = f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_eda/{e}"
        os.makedirs(eda_sub, exist_ok=True)
        metainfo["folder_paths"]["eda"][e] = {"path": eda_sub,
                                              "subfolders": {}}

        for esub in eda_sub_subfolders:
            if e != f"{ops_name}_shapiro_wilk_test":
                eda_2sub = f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_eda/{e}/{esub}_eda_plots"
                os.makedirs(eda_2sub, exist_ok=True)
                metainfo["folder_paths"]["eda"][e]["subfolders"][esub] = {"path": eda_2sub}

    # Graph subfolders
    for gsf in rn_graph_subfolders:
        gsf_sub = f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_rn_graph/{gsf}"
        os.makedirs(gsf_sub, exist_ok=True)
        metainfo["folder_paths"]["rn_graph"][gsf] = {"path": gsf_sub,
                                                     "subfolders": None}

    # Machine learning subfolders
    for mlsf in ml_subfolders:
        ml_sub = f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_ml/{ops_name}_{mlsf}"
        os.makedirs(ml_sub, exist_ok=True)
        metainfo["folder_paths"]["ml"][mlsf] = {"path": ml_sub,
                                                "subfolders": {}}

        #Machine learning sub-subfolders
        for mlssf in ml_sub_subfolders:
            ml_2sub = f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_ml/{ops_name}_{mlsf}/{ops_name}_{mlssf}_{mlsf}"
            os.makedirs(ml_2sub, exist_ok=True)
            metainfo["folder_paths"]["ml"][mlsf]["subfolders"][mlssf] = {"path": ml_2sub,
                                                                         "subfolders": {}}

            for mlsssf in ml_sub_sub_subfolders:
                ml_3sub = f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_ml/{ops_name}_{mlsf}/{ops_name}_{mlssf}_{mlsf}/{ops_name}_{mlsssf}_{mlssf}_{mlsf}"
                os.makedirs(ml_3sub, exist_ok=True)
                metainfo["folder_paths"]["ml"][mlsf]["subfolders"][mlssf]["subfolders"][mlsssf] = {"path": ml_3sub}

    with open(f"{ops_folder}/{ops_name}/{metainfo_filename}.json", "w") as m: json.dump(metainfo, m, indent=4)


    return None


def del_ops_folder(ops_name: str) -> None:
    try:
        os.rmdir(ops_name)
        print(f"{ops_name} Operation Folder Deleted")
    except FileNotFoundError:
        print("\033[91mOperation Folder Not Found\033[0m")
    return None


def write_metainfo(ops_name: str) -> None:

    target_folder = f"{ops_folder}/{ops_name}/"
    assert os.path.isdir(target_folder) is True, f"{target_folder} folder not found. Have you created the operation first?"

    if os.path.isdir(target_folder) is True:
        metainfo = {
            "common": {
                "n_raw_traffic_volumes": None,
                "n_clean_traffic_volumes": None,
                "n_raw_average_speeds": None,
                "n_clean_average_speeds": None,
                "raw_volumes_size": None,
                "clean_volumes_size": None,
                "raw_average_speeds_size": None,
                "clean_average_speeds_size": None
            },
            "traffic_volumes": {
                "start_date_iso": None,
                "end_date_iso": None,
                "start_year": None,
                "start_month": None,
                "start_day": None,
                "start_hour": None,
                "end_year": None,
                "end_month": None,
                "end_day": None,
                "end_hour": None,
                "n_days": None,
                "n_months": None,
                "n_years:": None,
                "n_weeks": None,
                "raw_filenames": [],
                "raw_filepaths": [],
                "clean_filenames": [],
                "clean_filepaths": [],
                "n_rows": []
            },
            "average_speeds": {
                "start_date_iso": None,
                "end_date_iso": None,
                "start_year": None,
                "start_month": None,
                "start_day": None,
                "start_hour": None,
                "end_year": None,
                "end_month": None,
                "end_day": None,
                "end_hour": None,
                "n_days": None,
                "n_months": None,
                "n_years:": None,
                "n_weeks": None,
                "raw_filenames": [],
                "raw_filepaths": [],
                "clean_filenames": [],
                "clean_filepaths": [],
                "n_rows": []
            },
            "metadata_files": [],
            "folder_paths": {},
            "forecasting": {
                "target_datetimes": {
                    "V": None,
                    "AS": None
                }
            },
            "by_trp_id": {
                "trp_ids" : {} #TODO ADD IF A RAW FILES HAS A CORRESPONDING CLEAN ONE (FOR BOTH TV AND AVG SPEEDS)
            }
        }

        with open(target_folder + metainfo_filename + ".json", "w") as tf: json.dump(metainfo, tf, indent=4)

    return None












































