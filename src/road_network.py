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
import matplotlib.pyplot as plt

from exceptions import WrongGraphProcessingBackendError
from definitions import GlobalDefinitions
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
        self._network: nx.DiGraph | None = None

        if not network_binary:
            self._network = nx.DiGraph(**{"network_id": self.network_id, "name": self.name})
        else:
            self._network = pickle.loads(network_binary)  # To load pre-computed graphs

        if self._backend not in GlobalDefinitions.GRAPH_PROCESSING_BACKENDS:
            raise WrongGraphProcessingBackendError(f"{self._backend} is not a valid graph processing backend. Try one of: {', '.join(GlobalDefinitions.GRAPH_PROCESSING_BACKENDS)}")
        #TODO SET ENVIRONMENT VARIABLES FOR CUDF?



    def load_nodes(self) -> None:
        all(self._network.add_nodes_from((row.to_dict().pop("node_id"), row) for _, row in partition.iterrows()) for partition in self._loader.get_nodes().partitions)
        return None


    def load_links(self, county_ids_filter: list[str] | None = None) -> None:
        all(self._network.add_edges_from(
                (row["start_traffic_node_id"], row["end_traffic_node_id"],
                 {k: v for k, v in row.to_dict().items() if k not in ["start_traffic_node_id", "end_traffic_node_id"]})
                for _, row in partition.iterrows()
            ) for partition in self._loader.get_links(county_ids_filter=county_ids_filter).partitions)
        return None


    def build(self, auto_load_nodes: bool = True, auto_load_links: bool = True, county_ids_filter: list[str] | None = None, verbose: bool = True) -> None:

        if auto_load_nodes:
            if verbose:
                print("Loading nodes...")
            self.load_nodes()

        if auto_load_links:
            if verbose:
                print("Loading links...")
            self.load_links(county_ids_filter=county_ids_filter)

        #Removing isolated nodes since they're unreachable
        self._network.remove_nodes_from(list(i for i in nx.isolates(self._network))) #Using a generator to avoid eccessive memory consumption
        #This way we're also going to apply all filters if any have been set since if there isn't a link connecting a node to another that will be isolated
        # NOTE WHEN WE'LL HAVE THE ABILITY TO FILTER DIRECTLY AT THE SOURCE OF THE NODES (WHEN WE'LL HAVE THE MUNICIPALITY AND COUNTY DATA ON THE NODES) WE'LL JUST NOT LOAD THE ONES OUTSIDE OF THE FILTERS CONDITIONS

        if verbose:
            print("Road network graph created!")

        print(self._network)
        nx.draw(self._network, with_labels=True)
        plt.show()

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


    #@saveplot()
    def save_graph_image(self, fp: str) -> None:
        nx.draw(self._network, with_labels=True, node_size=1500, font_size=25, font_color="yellow", font_weight="bold")
        return ...


    def to_pickle(self) -> None:
        nx.write_gpickle(self._network, 'graph.pkl')
        return None


    def from_pickle(self) -> nx.DiGraph:
        return nx.read_gpickle(self._network, 'graph.pkl')


    def to_db(self) -> None:
        pickle.dumps(self._network)
        ...


    @staticmethod
    def from_db(fp: str) -> None:
        return pickle.loads(fp)






