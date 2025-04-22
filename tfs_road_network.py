from pydantic import BaseModel as PydanticBaseModel
import geopandas as gpd
import geojson
import pickle
import networkx as nx
import datetime
from typing import Any
from tqdm import tqdm


#To allow arbitrary types in the creation of a Pydantic dataclass.
#In our use case this is done to allow the use of GeoPandas GeoDataFrame objects as type hints in the RoadNetwork class
class BaseModel(PydanticBaseModel):
    class Config:
        arbitrary_types_allowed = True



#The vertices are intersections, just like in most road network representations (like Google Maps, etc.)
class Vertex(BaseModel):
    vertex_id: str
    is_roundabout: bool
    type: str #Example: Feature
    geometry_type: str
    coordinates: list[float]
    lat: float
    lon: float
    connected_traffic_link_ids: list[str]
    road_node_ids: list[str]
    n_incoming_links: int
    n_outgoing_links: int
    n_undirected_links: int
    legal_turning_movements: list[dict[str, [str | list[str]]]]
    road_system_references: list[str]
    municipality_ids: list[str] = None #TODO TO GET THIS ONE SINCE IT DOESN'T EXIST YET IN THE DATA AVAILABLE RIGHT NOW. FOR NOW IT WILL BE NONE


    def get_vertex_data(self) -> dict[Any, Any]:
        return {attr: val for attr, val in self.__dict__.items()}



class Arch(BaseModel):
    arch_id: str
    type: str #Example: Feature
    geometry_type: str
    coordinates: list[list[float]]
    year_applies_to: int
    candidate_ids: list[str]
    road_system_references: list[str] #A list of "short form" road references. An example could be a road reference (short form) contained in the road_reference_short_form attribute of a TrafficRegistrationPoint instance
    road_category: str #One letter road category
    road_category_extended: str #Road category's full name. Examples: Europaveg, Riksveg
    road_sequence_id: int #In "roadPlacements"
    start_position: float #In "roadPlacements"
    end_position: float #In "roadPlacements"
    direction: str #In "roadPlacements"
    start_position_with_placement_direction: float #In "roadPlacements"
    end_position_with_placement_direction: float #In "roadPlacements"
    functional_road_class: int
    function_class: str
    start_traffic_node_id: str #TODO VERIFY IF THIS IS THE START OF THE LINK
    end_traffic_node_id: str #TODO VERIFY IF THIS IS THE END OF THE LINK
    subsumed_traffic_node_ids: list[str]
    road_link_ids: list[str]
    road_node_ids: list[str]
    municipality_ids: list[int]
    county_ids: list[int]
    highest_speed_limit: int
    lowest_speed_limit: int
    max_lanes: int
    min_lanes: int
    has_only_public_transport_lanes: bool
    length: float
    traffic_direction_wrt_metering_direction: str
    is_norwegian_scenic_route: bool
    is_ferry_route: bool
    is_ramp: bool
    toll_station_ids: list[str]
    associated_trp_ids: list[str]
    traffic_volumes: list[dict[str, [str | float | int | None]]]
    urban_ratio: int | float | None
    number_of_establishments: int
    number_of_employees: int
    number_of_inhabitants: int
    has_anomalies: bool
    anomalies: list #TODO YET TO DEFINE THE DATA TYPE OF THE ELEMENTS OF THIS LIST


    def get_arch_data(self) -> dict[Any, Any]:
        return {attr: val for attr, val in self.__dict__.items()}



class TrafficRegistrationPoint(BaseModel):
    trp_id: str
    name: str
    lat: float
    lon: float
    road_category: str
    road_category_extended: str
    road_reference_short_form: str #This is a string which is identifies the road reference of the TRP
    road_sequence_id: int | float
    road_reference_history: list[dict[str, [str | None]]]
    relative_position: float
    county_name: str
    county_number: int
    geographic_number: int
    country_part_id: int
    country_part_name: str
    municipality_name: str
    municipality_number: int
    traffic_registration_type: str
    first_data: str | datetime.datetime
    first_data_with_quality_metrics: str | datetime.datetime
    latest_data_volume_by_day: str | datetime.datetime
    latest_data_volume_by_hour: str | datetime.datetime
    latest_data_volume_average_daily_by_year: str | datetime.datetime
    latest_data_volume_average_daily_by_season: str | datetime.datetime
    latest_data_volume_average_daily_by_month: str | datetime.datetime



class RoadNetwork(BaseModel):
    network_id: str
    _vertices: list[Vertex] = None #Optional parameter
    _arches: list[Arch] = None #Optional parameter
    n_vertices: int
    n_arches: int
    n_trp: int
    road_network_name: str
    _network: nx.Graph = nx.Graph()


    def get_network_attributes_and_values(self) -> dict[Any, Any]:
        return {attr: val for attr, val in self.__dict__.items()}


    def load_vertices(self, vertices: list[Vertex] = None, municipality_id_filter: list[str] | None = None, **kwargs) -> None:
        """
        This function loads the vertices inside a RoadNetwork class instance.

        Parameters:
            vertices: a list of Vertex objects
            municipality_id_filter: a list of municipality IDs to use as filter to only keep vertices which are actually located within that municipality
            **kwargs: other attributes which might be needed in the process
        """

        # If a RoadNetwork class instance has been created and already been provided with vertices it's important to ensure that the ones that are located outside
        # of the desired municipality get filtered
        if self._vertices is not None:
            self._vertices = [v for v in self._vertices if any(i in municipality_id_filter for i in v.municipality_ids) is False] #Only keeping the vertex if all municipalities of the vertex aren't included in the ones to be filtered out
            return None
        else:
            self._vertices = [v for v in vertices if any(i in municipality_id_filter for i in v.municipality_ids) is False]
            return None


    def load_arches(self, arches: list[Arch] = None, municipality_id_filter: list[str] | None = None, **kwargs) -> None:
        """
        This function loads the arches inside a RoadNetwork class instance.

        Parameters:
            arches: a list of Arch objects
            municipality_id_filter: a list of municipality IDs to use as filter to only keep arches which are actually located within that municipality
            **kwargs: other attributes which might be needed in the process
        """

        #If a RoadNetwork class instance has been created and already been provided with arches it's important to ensure that the ones that are located outside
        # of the desired municipality get filtered
        if self._arches is not None:
            self._arches = [a for a in self._arches if any(i in municipality_id_filter for i in a.municipality_ids) is False] #Only keeping the vertex if all municipalities of the vertex aren't included in the ones to be filtered out
            return None
        else:
            self._arches = [a for a in arches if any(i in municipality_id_filter for i in a.municipality_ids) is False]
            return None


    def build(self, verbose: bool) -> None:
        if verbose is True: print("Loading vertices...")
        for v in tqdm(self._vertices): self._network.add_node((v.vertex_id, v.get_vertex_data()))

        if verbose is True: print("Loading arches...")
        for a in tqdm(self._arches): self._network.add_edge(a.start_traffic_node_id, a.end_traffic_node_id, **a.get_arch_data())
        print()

        return None


    def get_graph(self) -> nx.Graph:
        return self._network


    def degree_centrality(self) -> dict:
        """
        Returns the degree centrality for each node
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


    def load_graph(self, filepath: str) -> None:
        """
        Loads a previously exported graph into the RoadNetwork instance.
        Only accepting graphs exported in pickle format.
        """
        assert filepath.endswith(".pickle") is True or filepath.endswith(".pkl") is True, "File extension missing or not pickle"
        with open(filepath, "wb") as g_loader:
            self._network = pickle.load(g_loader)
        return None




#TODO FILTER ROAD NETWORK BY A LIST OF MUNICIPALITY IDs. SO ONE CAN CREATE A NETWORK WITH VERTICES OR ARCHES FROM ONLY SPECIFIC MUNICIPALITIES

#TODO WE COULD EXPORT EACH Vertex OR Arch OBJECT AS A STANDALONE FILE (OR RECORD IN A DB) WITH MORE INFORMATION (COLLECTED AFTER THE ANALYSIS) ABOUT THE Vertex OR THE Arch ITSELF

#TODO LOAD TRP METADATA FOR EACH NODE/LINK WHICH HAS A TRP CONNECTED TO IT

#TODO ADD CUDF BACKEND