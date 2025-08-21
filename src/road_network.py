import os
import json
import pickle
from pydantic.types import PositiveFloat, PositiveInt
import geopandas as gpd
import geojson
import networkx as nx
import datetime
from typing import Any, Iterator, Literal, Hashable
from tqdm import tqdm
from scipy.spatial.distance import euclidean, cityblock  # Scipy's cityblock distance is the Manhattan distance. Scipy distance docs: https://docs.scipy.org/doc/scipy/reference/spatial.distance.html#module-scipy.spatial.distance
from geopy.distance import geodesic # To calculate distance (in meters) between two sets of coordinates (lat-lon). Geopy distance docs: https://geopy.readthedocs.io/en/stable/#module-geopy.distance
from shapely import Point, LineString

from exceptions import WrongGraphProcessingBackendError
from utils import GlobalDefinitions
from loaders import BatchStreamLoader

Node = Hashable  # Any object usable as a node


class RoadNetwork:

    def __init__(self,
                 network_id: str,
                 name: str,
                 backend: str = "networkx",
                 broker: Any | None = None,
                 loader: BatchStreamLoader | None = None,
                 network_binary: bytes | None = None
    ):
        self._loader: BatchStreamLoader = loader
        self._broker: Any = broker #Synchronous DBBroker
        self.network_id: str = network_id
        self.name: str = name
        self._backend: str = backend
        self._network: nx.Graph | None = None

        if network_binary:
            self._network = pickle.loads(network_binary) #To load pre-computed graphs

        if self._backend not in GlobalDefinitions.GRAPH_PROCESSING_BACKENDS:
            raise WrongGraphProcessingBackendError(f"{self._backend} is not a valid graph processing backend. Try one of: {', '.join(GlobalDefinitions.GRAPH_PROCESSING_BACKENDS)}")
        #TODO SET ENVIRONMENT VARIABLES FOR CUDF?


    @property
    def graph(self) -> nx.Graph:
        """
        Returns the network's graph object.

        Returns:
            The road network's networkx graph object
        """
        return self._network


    def get_data(self) -> dict[Any, Any]:
        return self.__dict__


    def load_nodes(self) -> None:
        all(self._network.add_nodes_from((row["id"], row.to_dict().pop("id")) for row in partition) for partition in self._loader.get_nodes().partitions)
        return None


    def load_links(self) -> None:
        all(self._network.add_nodes_from((row["start_traffic_node_id"], row["end_traffic_node_id"], lambda row: row.to_dict().pop(item) for item in ["start_traffic_node_id", "end_traffic_node_id"]) for row in partition) for partition in self._loader.get_links().partitions)
        return None


    def _compute_edge_weight(self, edge: Node, is_forecast: bool = False, forecasting_horizon: datetime.datetime | None = None) -> float | int:



        #TODO FOR EVERY EDGE COMPUTE ITS WEIGHT WITH self._compute_edge_weight()
        #TODO CREATE A compute_edges_weights() THAT THE USER CAN CALL AND THUS COMPUTE THE WEIGHTS FOR THE WHOLE GRAPH. CALLING THE SAME METHOD SHOULD JUST UPDATE THE WEIGHTS SINCE edge["attr"] = ... JUST UPDATES THE ATTRIBUTE VALUE


        return ...


    def build(self, auto_load_nodes: bool = True, auto_load_links: bool = True, verbose: bool = True) -> None:

        if auto_load_nodes:
            if verbose:
                print("Loading nodes...")
            self.load_nodes()

        if auto_load_links:
            if verbose:
                print("Loading links...")
            self.load_links()

        if verbose:
            print("Road network graph created!")

        return None


    def degree_centrality(self) -> dict:
        """
        Returns the degree centrality for each node

        Returns:
            A dictionary comprehending the degree centrality of every node of the network
        """
        return nx.degree_centrality(self._network)


    def betweenness_centrality(self) -> dict:
        """
        Returns the betweennes centrality for each node
        """
        return nx.betweenness_centrality(self._network, seed=100)


    def eigenvector_centrality(self) -> dict:
        """
        Returns the eigenvector centrality for each node
        """
        return nx.eigenvector_centrality(self._network)


    def load_centrality(self) -> dict:
        """
        Returns the eigenvector centrality for each node
        """
        return nx.load_centrality(self._network)


    def to_json(self, filepath: str) -> None: #TODO -> to_db
        """
        Exports the graph (contained into the self._network attribute into a json file).
        """
        assert filepath.endswith(".json") is True, "File extension missing or not json"
        with open(filepath, "w", encoding="utf-8") as g_dumper:
            json.dump(nx.to_dict_of_dicts(self._network), g_dumper, indent=4)
        return None


    def from_json(self, filepath: str) -> None:
        """
        Loads a graph from a json file into the self._network attribute.
        """
        assert filepath.endswith(".json"), "File extension missing or not json"
        with open(filepath, "r", encoding="utf-8") as g_loader:
            self._network = nx.from_dict_of_dicts(json.load(g_loader))
        return None


    def get_astar_path(self, source: str, target: str, weight: str, heuristic: Literal["manhattan", "euclidean"]) -> list[str]:
        """
        Returns the shortest path calculated with the  algorithm.

        Parameters:
            source: the source vertex ID as a string
            target: the target vertex ID as a string
            weight: the attribute to consider as weight for the arches of the graph
            heuristic: the heuristic to use during the computation of the shortest path. Can be either "manhattan" or "Euclidean"

        Returns:
            A list of vertices, each linked by an arch
        """
        return nx.astar_path(G=self._network, source=source, target=target, heuristic={"manhattan": cityblock, "euclidean": euclidean}[heuristic], weight=weight)


    def get_dijkstra_path(self, source: str, target: str, weight: str) -> list[str]:
        """
        Returns the shortest path calculated with the  algorithm.

        Parameters:
            source: the source vertex ID as a string
            target: the target vertex ID as a string
            weight: the attribute to consider as weight for the arches of the graph

        Returns:
            A list of vertices, each linked by an arch
        """
        return nx.dijkstra_path(G=self._network, source=source, target=target, weight=weight)


    def find_trps_on_path(self, path: list[str]) -> Iterator[str]:
        """
        Finds all TRPs present along a path by checking each node.

        Parameters:
            path: a list of strings, each one representing a specific vertex on the path.

        Returns:
            A filter object which contains only the vertices which actually have a TRP associated with them.
        """

        return filter(lambda v: self._network[v]["has_trps"] == True, path)









