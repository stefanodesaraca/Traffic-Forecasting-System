import shutil
from contextlib import contextmanager
from itertools import islice
from typing import Literal, Any, Generator
import numpy as np
import pandas as pd
from pydantic.types import PositiveInt
import dask
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
from functools import lru_cache, wraps

import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.express as px

from exceptions import TargetVariableNotFoundError, MissingDataError
from definitions import GlobalDefinitions

pd.set_option("display.max_columns", None)


@contextmanager
def dask_cluster_client(scheduler_address: str | None = None, direct_to_workers: bool = False, processes: bool = False):
    """
    - Initializing a client to support parallel backend computing and to be able to visualize the Dask client dashboard
    - Check localhost:8787 to watch real-time processing
    - By default, the number of workers is obtained by dask using the standard os.cpu_count()
    - More information about Dask local clusters here: https://docs.dask.org/en/stable/deploying-python.html
    """
    from dask.distributed import shuffle
    shuffle.p2p_barrier_timeout = 120  # increase from 30s to 2 minutes

    cluster = None
    if scheduler_address:
        client = Client(address=scheduler_address + ":8786",
                        timeout="60s",
                        direct_to_workers=direct_to_workers)
        # Creating a zip of the entire src/ folder
        shutil.make_archive("src", "zip", "src")
        # Upload the whole archive to workers
        client.upload_file("src.zip")
        dask.config.set({"dataframe.shuffle.method": "tasks"})

        print(client.scheduler_info())  # shows scheduler & workers
        print(client.run(lambda: __import__('sys').executable))  # shows python executable on each worker
        print(client.run(lambda: __import__('socket').gethostname()))  # confirm hosts
    else:
        cluster = LocalCluster(processes=processes)
        client = Client(cluster)
    try:
        yield client
    finally:
        client.close()
        if not scheduler_address:
            cluster.close()


def check_target(target: str, errors: bool = False) -> bool:
    if target not in GlobalDefinitions.TARGET_DATA.keys() and target not in GlobalDefinitions.TARGET_DATA.values():
        if errors:
            raise TargetVariableNotFoundError(f"Wrong target variable: {target}")
        return False
    return True


def check_municipality_id(municipality_id: PositiveInt) -> bool:
    if not isinstance(municipality_id, PositiveInt):
        raise ValueError(f"{municipality_id} is not a positive int")
    return True


def to_pg_array(py_list: list[str] | tuple[str]) -> str:
    return "{" + ",".join(py_list) + "}"


def ZScore(df: dd.DataFrame, column: str) -> dd.DataFrame:
        df["z_score"] = (df[column] - df[column].mean()) / df[column].std()
        return df[(df["z_score"] > -3) & (df["z_score"] < 3)].drop(columns="z_score").persist()


def sin_encoder(data: dd.Series | dd.DataFrame | PositiveInt, timeframe: int) -> dd.Series | dd.DataFrame:
    """
    Apply sine transformation for cyclical encoding.

    Parameters
    ----------
    data : dd.Series or dd.DataFrame
        The data where to execute the transformation.
    timeframe : int
        A number of days indicating the timeframe for the cyclical transformation.

    Returns
    -------
    dd.Series or dd.DataFrame
        The sine-transformed data.

    Notes
    -----
    The order of the function parameters has a specific reason. Since this function
    will be used with Dask's map_partition() (which takes a function and its parameters
    as input), it's important that the first parameter of sin_transformer is actually
    the data where to execute the transformation itself by map_partition().

    For more information about Dask's map_partition() function:
    https://docs.dask.org/en/stable/generated/dask.dataframe.DataFrame.map_partitions.html
    """
    return np.sin(data * (2.0 * np.pi / timeframe))


def cos_encoder(data: dd.Series | dd.DataFrame | PositiveInt, timeframe: int) -> dd.Series | dd.DataFrame:
    """
    Apply cosine transformation for cyclical encoding.

    Parameters
    ----------
    data : dd.Series or dd.DataFrame
        The data where to execute the transformation.
    timeframe : int
        A number of days indicating the timeframe for the cyclical transformation.

    Returns
    -------
    dd.Series or dd.DataFrame
        The cosine-transformed data.

    Notes
    -----
    The order of the function parameters has a specific reason. Since this function
    will be used with Dask's map_partition() (which takes a function and its parameters
    as input), it's important that the first parameter of sin_transformer is actually
    the data where to execute the transformation itself by map_partition().

    For more information about Dask's map_partition() function:
    https://docs.dask.org/en/stable/generated/dask.dataframe.DataFrame.map_partitions.html
    """
    return np.cos((data - 1) * (2.0 * np.pi / timeframe))


def arctan2_decoder(sin_val: float, cos_val: float) -> int | float:  # TODO VERIFY IF IT'S ACTUALLY AN INT (IF SO REMOVE | float)
    """
    Decode cyclical features using arctan2 function.

    Parameters
    ----------
    sin_val : float
        The sine component value.
    cos_val : float
        The cosine component value.

    Returns
    -------
    int or float
        The decoded angle in degrees.
    """
    angle_rad = np.arctan2(sin_val, cos_val)
    return (angle_rad * 360) / (2 * np.pi)


def split_by_target(data: dd.DataFrame, target: str, mode: Literal[0, 1]) -> tuple[dd.DataFrame, dd.DataFrame, dd.DataFrame, dd.DataFrame] | tuple[dd.DataFrame, dd.DataFrame]:
    """
    Splits the Dask DataFrame into training and testing sets based on the target column and mode.

    Parameters:
        data: dd.DataFrame
        target: str ("volume" or "mean_speed")
        mode: the mode which indicates the kind of split it's intended to execute.
                0 - Stands for the classic 4 section train-test-split (X_train, X_test, y_train, y_test)
                1 - Indicates a forecasted specific train-test-split (X, y)

    Returns:
        X_train, X_test, y_train, y_test
    """

    X = data.drop(columns=[target])
    y = data[[target]]

    if mode == 1:
        return X.persist(), y.persist()
    elif mode == 0:
        n_rows = data.shape[0].compute()
        p_70 = int(n_rows * 0.70)
        return (dd.from_pandas(X.head(p_70)),
                dd.from_pandas(X.tail(n_rows - p_70)),
                dd.from_pandas(y.head(p_70)),
                dd.from_pandas(y.tail(n_rows - p_70)))
    else:
        raise ValueError(f"{mode} is not a valid splitting mode")


def merge(dfs: list[dd.DataFrame]) -> dd.DataFrame:
    """
    Dask Dataframes merger function
    Parameters:
        dfs: a list of Dask Dataframes to concatenate
    """
    try:
        return (dd.concat(dfs, axis=0)
                .repartition(partition_size=GlobalDefinitions.DEFAULT_DASK_DF_PARTITION_SIZE)
                .sort_values(["zoned_dt_iso"], ascending=True)
                .persist())  # Sorting records by date
    except ValueError as e:
        raise MissingDataError(f"No data to concatenate. Error: {e}")


def get_n_items_from_gen(gen: Generator[Any, None, None], n: PositiveInt) -> Generator[list[list | tuple], None, None]:
    """Yield lists of up to n items from the generator."""
    while True:
        chunk = list(islice(gen, n))
        if not chunk:
            break
        yield chunk


def cached(maxsize: int | None = 128, typed: bool = False) -> Any:
    """
    Decorator that applies lru_cache, but allows per-call opt-out with enable_cache=False.
    """
    def decorator(func):
        cached_func = lru_cache(maxsize=maxsize, typed=typed)(func)
        @wraps(func)
        def wrapper(*args, enable_cache: bool = False, **kwargs):
            if enable_cache:
                return cached_func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator


def cached_async():
    """
    Async-compatible cache decorator with per-call enable/disable.
    """
    def decorator(func):
        cache = {}
        @wraps(func)
        async def wrapper(*args, enable_cache: bool = False, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            if enable_cache:
                if key in cache:
                    return cache[key]
                result = await func(*args, **kwargs)
                cache[key] = result
                return result
            else:
                return await func(*args, **kwargs)
        return wrapper
    return decorator


def save_plot(plotFunction):
    def save(plots, fp):
        if isinstance(plots, (plt.Figure, plt.Axes, sns.axisgrid.FacetGrid, sns.axisgrid.PairGrid, list)):
            plt.savefig(fp, dpi=300)
            print(f"{fp} exported correctly")
        elif isinstance(plots, plotly.graph_objs._figure.Figure):
            plots.write_html(fp)
            print(f"{fp} exported correctly")
        else:
            try:
                plt.savefig(f"{fp}", dpi=300)
                print(f"{fp} exported correctly")
            except Exception as e:
                print(
                    f"\033[91mExporting the plots wasn't possible, the returned type is not included in the decorator function. Error: {e}\033[0m")
        return None

    @wraps(plotFunction)
    def wrapper(*args, **kwargs):
        plots, fp = plotFunction(*args, **kwargs)
        if isinstance(plots, list):
            for plot in plots:
                save(plot, fp)
        else:
            save(plots, fp)
        return None

    return wrapper


def get_increments(n: int | float, k: PositiveInt) -> list[int | float]:
    step = n // k
    return [step * i for i in range(1, k + 1)]


def closest(numbers: list[int | float], k: int | float):
    return min(numbers, key=lambda n: abs(n - k))


def get_trait_length_by_road_category(links: list[callable]) -> dict[str, float]:
    lengths = {}
    for link in links:
        lengths[link["road_category"]] += link["length"]
    return lengths


def get_trait_main_road_category(grouped_trait: dict[str, float]) -> str:
    return max(grouped_trait, key=grouped_trait.get)


def _get_road_category_proportions(grouped_trait: dict[str, float]) -> dict[str, dict[str, float]]:
    total = sum(grouped_trait.values())
    stats = {}
    for category, length in grouped_trait.items():
        stats[category] = {
            "length": length,
            "percentage": (length / total) * 100
        }
    return stats
