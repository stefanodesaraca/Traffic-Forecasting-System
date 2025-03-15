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

    print(volumes)



    return None























