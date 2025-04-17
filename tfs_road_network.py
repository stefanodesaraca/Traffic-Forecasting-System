from pydantic import BaseModel as PydanticBaseModel
import geopandas as gpd
import geojson
import networkx

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



class Arch(BaseModel):
    arch_id: str
    type: str #Example: Feature
    geometry_type: str
    coordinates: list[list[float]]
    year_applies_to: int
    candidate_ids: list[str]
    road_system_references: list[str]
    road_category: str #One letter road category
    road_category_extended: str #Road category's full name. Example: Europaveg
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




class RoadNetwork(BaseModel):
    network_id: str
    edges: list[Edge]
    arches: list[Arch]
    n_edges: int
    n_arches: int
    n_trp: int
    road_network_name: str





#TODO DEFINE THE RoadNetwork CLASS
# DEFINE THE TrafficRegistrationPoint CLASS
# ALL OF THESE CLASSES WILL INHERIT FROM pydantic's BaseModel AND HAVE EACH THEIR OWN ATTRIBUTES WHICH DESCRIBE THE OBJECT ITSELF

#TODO THE EDGES, THE LINKS AND THE TRAFFIC REIGSTRATION POINTS OBJECTS WILL BE CREATED IN SPECIFIC METHODS IN TEH tfs_utils.py FILE
# HERE WE'LL ONLY DEFINE THE LOGICS, ATTRIBUTES AND METHODS WHICH EACH CLASS REPRESENTS
# ONLY THE RoadNetwork CLASS WILL HAVE EVERYTHING (ALMOST) DEFINED INSIDE ITSELF, SO IT WILL HAVE METHODS WHICH WILL LET IT GENERATE THE NETWORK GIVEN A SET OF EDGES AND LINKS

#TODO POTENTIALLY IN THE FUTURE WE COULD CREATE A CLASS Edge AND A CLASS Arch WHICH WILL CONTAIN INFORMATION ABOUT Edges AND Arches RESPECTIVELY. THEN WE COULD EXPORT EACH Edge OR Arch OBJECT AS A STANDALONE FILE (OR RECORD IN A DB) WITH MORE INFORMATION (COLLECTED AFTER THE ANALYSIS) ABOUT THE Edge OR THE Arch ITSELF



#Here we'll just define the road network graph, the operations on the graph will be defined in other files which will
#automatically find the right graph for the current operation, import it and then execute the analyses there