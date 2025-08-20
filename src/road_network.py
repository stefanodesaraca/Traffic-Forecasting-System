from pydantic import BaseModel as PydanticBaseModel
from pydantic.types import PositiveFloat, PositiveInt
import geopandas as gpd
import geojson
import json
import pickle
import networkx as nx
import datetime
from typing import Any, Iterator, Literal
from tqdm import tqdm
from scipy.spatial.distance import euclidean, cityblock  # Scipy's cityblock distance is the Manhattan distance. Scipy distance docs: https://docs.scipy.org/doc/scipy/reference/spatial.distance.html#module-scipy.spatial.distance
from geopy.distance import geodesic # To calculate distance (in meters) between two sets of coordinates (lat-lon). Geopy distance docs: https://geopy.readthedocs.io/en/stable/#module-geopy.distance
from shapely import Point, LineString

# To allow arbitrary types in the creation of a Pydantic dataclass.
# In our use case this is done to allow the use of GeoPandas GeoDataFrame objects as type hints in the RoadNetwork class
class BaseModel(PydanticBaseModel):
    class Config:
        arbitrary_types_allowed = True


# The vertices are intersections, just like in most road network representations (like Google Maps, etc.)
class Node(BaseModel):
    vertex_id: str
    type: str  # Example: Feature
    is_roundabout: bool
    geometry: Point
    lat: float # TODO TO GET FROM Point
    lon: float # TODO TO GET FROM Point
    connected_traffic_link_ids: list[str]
    road_node_ids: list[str]
    n_incoming_links: PositiveInt
    n_outgoing_links: PositiveInt
    n_undirected_links: PositiveInt
    legal_turning_movements: list[dict[str, [str | list[str]]]]
    road_system_references: list[str]
    municipality_ids: list[str] | None = None  # TODO TO GET THIS ONE SINCE IT DOESN'T EXIST YET IN THE DATA AVAILABLE RIGHT NOW. FOR NOW IT WILL BE NONE


    def get_vertex_data(self) -> dict[Any, Any]:
        """
        Returns all attributes and respective values of the Vertex instance.
        """
        return self.__dict__


class Link(BaseModel):
    link_id: str
    type: str  # Example: Feature
    geometry: LineString
    lat: float
    lon: float
    year_applies_to: PositiveInt
    candidate_ids: list[str]
    road_system_references: list[str]  # A list of "short form" road references. An example could be a road reference (short form) contained in the road_reference_short_form attribute of a TrafficRegistrationPoint instance
    road_category: str  # One letter road category
    road_sequence_id: int  # In "roadPlacements"
    start_position: float  # In "roadPlacements"
    end_position: float  # In "roadPlacements"
    direction: str  # In "roadPlacements"
    start_position_with_placement_direction: float  # In "roadPlacements"
    end_position_with_placement_direction: float  # In "roadPlacements"
    functional_road_class: int
    function_class: str
    start_traffic_node_id: str
    end_traffic_node_id: str
    subsumed_traffic_node_ids: list[str]
    road_link_ids: list[str]
    road_node_ids: list[str]
    municipality_ids: list[int]
    county_ids: list[int]
    highest_speed_limit: PositiveInt
    lowest_speed_limit: PositiveInt
    max_lanes: PositiveInt
    min_lanes: PositiveInt
    has_only_public_transport_lanes: bool
    length: PositiveFloat
    traffic_direction_wrt_metering_direction: str
    is_norwegian_scenic_route: bool
    is_ferry_route: bool
    is_ramp: bool
    toll_station_ids: list[str]
    associated_trp_ids: list[str]
    traffic_volumes: list[dict[str, [str | float | int | None]]]
    urban_ratio: int | float | None
    number_of_establishments: int
    number_of_employees: PositiveInt
    number_of_inhabitants: PositiveInt
    has_anomalies: bool
    anomalies: list  # TODO YET TO DEFINE THE DATA TYPE OF THE ELEMENTS OF THIS LIST
    weight: PositiveFloat | None = None  # Only set when executing forecasting with weighted arches


    def get_arch_data(self) -> dict[Any, Any]:
        """
        Returns all attributes and respective values of the Arch instance.
        """
        return self.__dict__


class TrafficRegistrationPoint(BaseModel):
    trp_id: str
    arch_id: str #The ID of the arch (road link) where the TRP is located
    arch_road_link_reference: str # The road link reference number of the arch where the TRP is located
    name: str
    lat: float
    lon: float
    road_category: str
    road_category_extended: str
    road_reference_short_form: str  # This is a string which is identifies the road reference of the TRP
    road_sequence_id: int | float
    road_reference_history: list[dict[str, [str | None]]]
    relative_position: float
    county_name: str
    county_number: int
    geographic_number: int
    country_part_id: int
    country_part_name: str
    municipality_name: str
    municipality_number: int  # Municipality ID
    traffic_registration_type: str
    first_data: str | datetime.datetime
    first_data_with_quality_metrics: str | datetime.datetime
    latest_data_volume_by_day: str | datetime.datetime
    latest_data_volume_by_hour: str | datetime.datetime
    latest_data_volume_average_daily_by_year: str | datetime.datetime
    latest_data_volume_average_daily_by_season: str | datetime.datetime
    latest_data_volume_average_daily_by_month: str | datetime.datetime


    def get_single_trp_network_data(self) -> dict[Any, Any]:
        """
        Returns all attributes and respective values of the TrafficRegistrationPoint instance.
        """
        return self.__dict__



class RoadNetwork(BaseModel):
    network_id: str
    _vertices: list[Node] | None = None  # Optional parameter
    _arches: list[Link] | None = None  # Optional parameter
    _trps: list[TrafficRegistrationPoint] | None = None  # Optional parameter. This is the list of all TRPs located within the road network
    road_network_name: str
    _network: nx.Graph = nx.Graph()


    def get_network_data(self) -> dict[Any, Any]:
        return self.__dict__


    def load_vertices(self, vertices: list[Node] = None, municipality_id_filter: list[str] | None = None, **kwargs) -> None:
        """
        This function loads the vertices inside a RoadNetwork class instance.

        Parameters:
            vertices: a list of Vertex objects
            municipality_id_filter: a list of municipality IDs to use as filter to only keep vertices which are actually located within that municipality
            **kwargs: other attributes which might be needed in the process

        Returns:
            None
        """

        # If a RoadNetwork class instance has been created and already been provided with vertices it's important to ensure that the ones that are located outside
        #  the desired municipality get filtered
        if self._vertices is not None:
            self._vertices = [v for v in self._vertices if any(i in municipality_id_filter for i in v.municipality_ids) is False]  # Only keeping the vertex if all municipalities of the vertex aren't included in the ones to be filtered out
        else:
            self._vertices = [v for v in vertices if any(i in municipality_id_filter for i in v.municipality_ids) is False]

        return None


    def load_arches(self, arches: list[Link] = None, municipality_id_filter: list[str] | None = None, **kwargs) -> None:
        """
        This function loads the arches inside a RoadNetwork class instance.

        Parameters:
            arches: a list of Arch objects
            municipality_id_filter: a list of municipality IDs to use as filter to only keep arches which are actually located within that municipality
            **kwargs: other attributes which might be needed in the process

        Returns:
            None
        """

        # If a RoadNetwork class instance has been created and already been provided with arches it's important to ensure that the ones that are located outside
        # the desired municipality get filtered
        if self._arches is not None:
            self._arches = [a for a in self._arches if any(i in municipality_id_filter for i in a.municipality_ids) is False]  # Only keeping the arch if all municipalities of the arch itself aren't included in the ones to be filtered out
        else:
            self._arches = [a for a in arches if any(i in municipality_id_filter for i in a.municipality_ids) is False]

        return None


    def load_trps(self, trps: list[TrafficRegistrationPoint] = None, municipality_id_filter: list[str] | None = None, **kwargs) -> None:
        """
        This function loads the arches inside a RoadNetwork class instance.

        Parameters:
            trps: a list of TrafficRegistrationPoint objects
            municipality_id_filter: a list of municipality IDs to use as filter to only keep arches which are actually located within that municipality
            **kwargs: other attributes which might be needed in the process

        Returns:
            None
        """

        #If a RoadNetwork class instance has been created and already been provided with traffic registration points it's important to ensure that the ones that are located outside
        # the desired municipality get filtered
        if self._trps is not None:
            self._trps = [trp for trp in self._trps if trp.municipality_number not in municipality_id_filter]  # Only keeping the TRP if the municipality of the TRP isn't included in the ones to be filtered out
        else:
            self._trps = [trp for trp in trps if trp.municipality_number not in municipality_id_filter]

        return None


    def build(self, verbose: bool) -> None:
        """
        Loads vertices and arches into the network.

        Parameters:
            verbose: boolean parameters. Indicates the verbosity of the process.

        Returns:
            None
        """
        if verbose:
            print("Loading vertices...")
        for v in tqdm(self._vertices):
            self._network.add_node((v.vertex_id, v.get_vertex_data()))

        if verbose:
            print("Loading arches...")
        for a in tqdm(self._arches):
            self._network.add_edge(a.start_traffic_node_id, a.end_traffic_node_id, **a.get_arch_data())
        print()

        return None


    def get_graph(self) -> nx.Graph:
        """
        Returns the network's graph object.

        Returns:
            The road network's networkx graph object
        """
        return self._network


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


    def to_pickle(self, filepath: str) -> None:
        """
        Exports the content of self._network of the RoadNetwork instance.
        Only accepting pickle as exporting file format.
        """
        assert filepath.endswith(".pickle") is True or filepath.endswith(".pkl") is True, "File extension missing or not pickle"
        with open(filepath, "wb") as g_dumper:
            pickle.dump(self._network, g_dumper)
        return None


    def to_json(self, filepath: str) -> None: #TODO -> to_db
        """
        Exports the graph (contained into the self._network attribute into a json file).
        """
        assert filepath.endswith(".json") is True, "File extension missing or not json"
        with open(filepath, "w", encoding="utf-8") as g_dumper:
            json.dump(nx.to_dict_of_dicts(self._network), g_dumper, indent=4)
        return None


    def from_pickle(self, filepath: str) -> None:
        """
        Loads a previously exported graph into the RoadNetwork instance.
        Only accepting graphs exported in pickle format.
        """
        assert filepath.endswith(".pickle") or filepath.endswith(".pkl"), "File extension missing or not pickle"
        with open(filepath, "rb") as g_loader:
            self._network = pickle.load(g_loader)
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




# TODO FILTER ROAD NETWORK BY A LIST OF MUNICIPALITY IDs. SO ONE CAN CREATE A NETWORK WITH VERTICES OR ARCHES FROM ONLY SPECIFIC MUNICIPALITIES

# TODO WE COULD EXPORT EACH Vertex OR Arch OBJECT AS A STANDALONE FILE (OR RECORD IN A DB) WITH MORE INFORMATION (COLLECTED AFTER THE ANALYSIS) ABOUT THE Vertex OR THE Arch ITSELF

# TODO LOAD TRP METADATA FOR EACH NODE/LINK WHICH HAS A TRP CONNECTED TO IT

# TODO ADD CUDF BACKEND
