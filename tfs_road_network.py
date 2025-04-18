from pydantic import BaseModel as PydanticBaseModel
import geopandas as gpd
import geojson
import networkx
import datetime


#To allow arbitrary types in the creation of a Pydantic dataclass.
#In our use case this is done to allow the use of GeoPandas GeoDataFrame objects as type hints in the RoadNetwork class
class BaseModel(PydanticBaseModel):
    class Config:
        arbitrary_types_allowed = True



#The edges are intersections, just like in most road network representations (like Google Maps, etc.)
class Edge(BaseModel):
    edge_id: str
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
    start_traffic_node_id: str
    end_traffic_node_id: str
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



class TrafficRegistrationPoint(BaseModel):
    trp_id: str
    name: str
    lat: float
    lon: float
    road_category: str
    road_category_extended: str
    road_reference_short_form: str #This is a string which is identifies the road reference of the TRP
    road_sequence_id: int | float
    relative_position: float
    county_name: str
    county_number: int
    geographic_number: int
    country_part_id: int
    country_part_name: str
    municipality_name: str
    municipality_number: int
    traffic_registration_type: str
    first_data: datetime.datetime
    first_data_with_quality_metrics: datetime.datetime
    latest_data_volume_by_day: datetime.datetime
    latest_data_volume_by_hour: datetime.datetime
    latest_data_volume_average_daily_by_year: datetime.datetime
    latest_data_volume_average_daily_by_season: datetime.datetime
    latest_data_volume_average_daily_by_month: datetime.datetime





class RoadNetwork(BaseModel):
    network_id: str
    edges: list[Edge] = None #Optional parameter
    arches: list[Arch] = None #Optional parameter
    n_edges: int
    n_arches: int
    n_trp: int
    road_network_name: str


    def load_edges(self, edges: list[Edge] = None, municipality_id_filter: list[str] | None = None, **kwargs):
        """
        This function loads the edges inside a RoadNetwork class instance.

        Parameters:
            edges: a list of Edge objects
            municipality_id_filter: a list of municipality IDs to use as filter to only keep edges which are actually located within that municipality
            **kwargs: other attributes which might be needed in the process
        """

        if self.edges is not None:
            self.edges = [e for e in self.edges if any(i in municipality_id_filter for i in e.municipality_ids) is False] #Only keeping the edge if all municipalities of the edge aren't included in the ones to be filtered out
            return self.edges
        else:
            edges = [e for e in edges if any(i in municipality_id_filter for i in e.municipality_ids) is False]
            return edges


    def load_arches(self, arches: list[Arch] = None, municipality_id_filter: list[str] | None = None, **kwargs):
        """
        This function loads the arches inside a RoadNetwork class instance.

        Parameters:
            arches: a list of Arch objects
            municipality_id_filter: a list of municipality IDs to use as filter to only keep arches which are actually located within that municipality
            **kwargs: other attributes which might be needed in the process
        """

        if self.arches is not None:
            self.arches = [a for a in self.arches if any(i in municipality_id_filter for i in a.municipality_ids) is False] #Only keeping the edge if all municipalities of the edge aren't included in the ones to be filtered out
            return self.arches
        else:
            arches = [a for a in arches if any(i in municipality_id_filter for i in a.municipality_ids) is False]
            return arches




















#TODO CREATE THE "generate_road_network_graph()" FUNCTION WHICH GATHERS EDGES AND ARCHES FROM THE CLASS ATTRIBUTES AND BUILDS THE R.N. GRAPH


#TODO FILTER ROAD NETWORK BY A LIST OF MUNICIPALITY IDs. SO ONE CAN CREATE A NETWORK WITH EDGES OR ARCHES FROM ONLY SPECIFIC MUNICIPALITIES



#TODO DEFINE THE TrafficRegistrationPoint CLASS

#TODO THE EDGES, THE LINKS AND THE TRAFFIC REIGSTRATION POINTS OBJECTS WILL BE CREATED IN SPECIFIC METHODS IN TEH tfs_utils.py FILE
# HERE WE'LL ONLY DEFINE THE LOGICS, ATTRIBUTES AND METHODS WHICH EACH CLASS REPRESENTS
# ONLY THE RoadNetwork CLASS WILL HAVE EVERYTHING (ALMOST) DEFINED INSIDE ITSELF, SO IT WILL HAVE METHODS WHICH WILL LET IT GENERATE THE NETWORK GIVEN A SET OF EDGES AND LINKS

#TODO WE COULD EXPORT EACH Edge OR Arch OBJECT AS A STANDALONE FILE (OR RECORD IN A DB) WITH MORE INFORMATION (COLLECTED AFTER THE ANALYSIS) ABOUT THE Edge OR THE Arch ITSELF



#Here we'll just define the road network graph, the operations on the graph will be defined in other files which will
#automatically find the right graph for the current operation, import it and then execute the analyses there