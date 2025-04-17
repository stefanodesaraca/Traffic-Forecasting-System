from pydantic import BaseModel as PydanticBaseModel
import geopandas as gpd
import geojson
import networkx

#To allow arbitrary types in the creation of a Pydantic dataclass.
#In our use case this is done to allow the use of GeoPandas GeoDataFrame objects as type hints in the RoadNetwork class
class BaseModel(PydanticBaseModel):
    class Config:
        arbitrary_types_allowed = True

class RoadNetwork(BaseModel):
    network_id: str
    edges: gpd.GeoDataFrame
    arches: gpd.GeoDataFrame
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