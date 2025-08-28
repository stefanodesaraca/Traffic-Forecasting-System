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


    def _parse_road_network_json(self, fp: str) -> dict[str, Any]:

        with open(fp, "r", encoding="utf-8") as f:
            road_system = json.load(f)

        #Each dictionary is a road object
        data = ({
            "id": road["id"],
            "href": road["href"],

            # Egenskaper (assuming we want them individually by name)
            "road_number": road["egenskaper"][0]["verdi"],
            "phase": road["egenskaper"][1]["verdi"],
            # The status of the road itself, example: built, under construction, historical, etc.
            "phase_id": road["egenskaper"][1].get("enum_id"),  # The id of the phase
            "road_category": road["egenskaper"][2]["verdi"],  # Road category #TODO TO CONVERT TO ID
            "road_category_enum_id": road["egenskaper"][2].get("enum_id"),  # Road category enum in the NVDB system

            # Geometri (Geometry)
            "geometry_wkt": road["geometri"]["wkt"],  # Well-known-text geometry of the whole road
            "geometry_srid": road["geometri"]["srid"],
            # Spatial Reference System Identifier, identifies the EPSG code which represents the UTM projection used
            "has_own_geometry": road["geometri"].get("egengeometri"),
            # Indicates whether the geometry of road inherits from another object in the NVDB

            # Lokasjon (Location)
            "municipality_ids": road["lokasjon"]["kommuner"],  # Municipalities which the road belongs to
            "counties": road["lokasjon"]["fylker"],  # Counties which the road belongs to

            # Kontraktsområder (Road Maintainers) (list of dicts)
            "maintainer_ids": (i["id"] for i in road["lokasjon"].get("kontraktsområder", [])),
            "maintainer_numbers": (i["nummer"] for i in road["lokasjon"].get("kontraktsområder", [])),
            "maintainer_names": (i["navn"] for i in road["lokasjon"].get("kontraktsområder", [])),

            # Vegforvaltere (Road Managers)
            "manager_enum_ids": (i["enumid"] for i in road["lokasjon"].get("vegforvaltere", [])),
            "manager_names": (i["vegforvalter"] for i in road["lokasjon"].get("vegforvaltere", [])),

            # Adresser (Addresses)
            "address_names": (i["navn"] for i in road["lokasjon"].get("adresser", [])),
            "address_codes": (i["adressekode"] for i in road["lokasjon"].get("adresser", [])),

            # Vegsystemreferanser (Road System Reference)
            # -- vegsystem --
            "road_system": (
                {
                "reference_road_category": ref["vegsystem"]["vegkategori"],
                "reference_fase": ref["vegsystem"]["fase"],
                "reference_number": ref["vegsystem"]["nummer"],
                }
                for ref in road["lokasjon"]["vegsystemreferanser"]
            ),

            # -- strekning --
            # Strekning(s) are portions of the road object from a descriptive perspective
            "stretches": (
                {
                    "main_stretch_id": ref["strekning"]["strekning"], # Strekning(s) are portions of the road object
                    "stretch_subsection_id": ref["strekning"]["delstrekning"], # Subsection of the stretch
                    "is_intersection_or_junction_arm": ref["strekning"]["arm"], # Indicates if it's an intersection or junction arm
                    "has_lanes_divider": ref["strekning"]["adskilte_løp"], # Indicates whether the lanes in opposite direction are physically separated or not
                    "reference_traffic_group": ref["strekning"]["trafikantgruppe"], # Reference traffic group
                    "reference_direction": ref["strekning"]["retning"], # Traffic direction from the reference direction perspective
                    "reference_start_meter": ref["strekning"]["fra_meter"], # Meter of the road start
                    "reference_end_meter": ref["strekning"]["til_meter"], # Meter of the road ending
                    "reference_short_form": ref["kortform"], # Road reference short form
                }
                for ref in road["lokasjon"]["vegsystemreferanser"]
            ),
            "overall_direction": road["metrertLokasjon"]["retning"],

            # Stedfestinger (NVDB Geographical Reference to a Specific Road)
            # Stedfestinger are portions of the road object which are needed to calculate the road segments of the object itself
            # Essentially they represent the shape and the location of the portion itself with a specific referencing system of the NVDB
            "physical_road_stretches": (
                {
                "georeference_type": stedfesting["type"],
                "road_link_sequence_id": stedfesting["veglenkesekvensid"],
                "georeference_start_position": stedfesting["startposisjon"],
                "georeference_end_position": stedfesting["sluttposisjon"],
                "road_link_sequence_direction": stedfesting["retning"],
                "lanes_object_location": stedfesting.get("kjørefelt", []),  # On which lanes the object is located
                "georeference_short_form": stedfesting["kortform"],
                } for stedfesting in road["lokasjon"]["stedfestinger"]
            ),
            # Lengde (Length)
            "length": road["lokasjon"]["lengde"],

            # Every segment of a single road object belongs only to that object
            "road_segments": ({
                "road_link_sequence_id": segment.get("veglenkesekvensid"),
                "georeference_start_position": segment.get("startposisjon"),
                "georeference_end_position": segment.get("sluttposisjon"),
                "length": segment.get("lengde"),
                "road_link_sequence_direction": segment.get("retning"),
                "segment_lanes": (f for f in segment.get("feltoversikt", [])),
                "road_link_type": segment.get("veglenkeType"),
                "detail_level": segment.get("detaljnivå"),
                "segment_road_type": segment.get("typeVeg"), #The type of road segment
                "validation_date": segment.get("startdato"), #Date when the segment has become valid

                # Nested dictionary: geometry
                "geometry": {
                    "wkt": segment.get("geometri", {}).get("wkt"),
                    "srid": segment.get("geometri", {}).get("srid")
                },

                # Nested dictionary: Vegsystemreferanse
                "road_system_reference": {
                    "road_system": {
                        "vegkategori": segment.get("vegsystemreferanse", {}).get("vegsystem", {}).get("vegkategori"),
                        "fase": segment.get("vegsystemreferanse", {}).get("vegsystem", {}).get("fase"),
                        "nummer": segment.get("vegsystemreferanse", {}).get("vegsystem", {}).get("nummer")
                    },
                    # Strekning: a major stretch of the road (like a long continuous part)
                    "stretch": {
                        "main_stretch_id": segment.get("vegsystemreferanse", {}).get("strekning", {}).get("strekning"),
                        "stretch_subsection_id": segment.get("vegsystemreferanse", {}).get("strekning", {}).get("delstrekning"), #Delstrekning: a subdivision of a stretch (these are official NVDB units)
                        "is_intersection_or_junction_arm": segment.get("vegsystemreferanse", {}).get("strekning", {}).get("arm"),
                        "has_lanes_divider": segment.get("vegsystemreferanse", {}).get("strekning", {}).get("adskilte_løp"),
                        "reference_traffic_group": segment.get("vegsystemreferanse", {}).get("strekning", {}).get("trafikantgruppe"),
                        "reference_direction": segment.get("vegsystemreferanse", {}).get("strekning", {}).get("retning"),
                        "reference_start_meter": segment.get("vegsystemreferanse", {}).get("strekning", {}).get("fra_meter"),
                        "reference_end_meter": segment.get("vegsystemreferanse", {}).get("strekning", {}).get("til_meter")
                    },
                    "reference_short_form": segment.get("vegsystemreferanse", {}).get("kortform")
                },

                # Lists
                "road_maintainers": (k for k in segment.get("kontraktsområder", [])),
                "road_administrators": (v for v in segment.get("vegforvaltere", [])),
                "associated_segment_ids": (a for a in segment.get("adresser", [])) if "adresser" in segment else None, #Segments associated to this one #TODO USEFUL TO COMPUTE THE GRAPH EDGES

                # Other top-level keys
                "municipality": segment.get("kommune"),
                "county": segment.get("fylke")
            } for segment in road["vegsegmenter"])

        } for road in road_system)


        return ...



    def _compute_edge_weight(self, edge: Node, is_forecast: bool = False, forecasting_horizon: datetime.datetime | None = None) -> float | int:



        #TODO FOR EVERY EDGE COMPUTE ITS WEIGHT WITH self._compute_edge_weight()
        #TODO CREATE A compute_edges_weights THAT THE USER CAN CALL AND THUS COMPUTE THE WEIGHTS FOR THE WHOLE GRAPH. CALLING THE SAME METHOD SHOULD JUST UPDATE THE WEIGHTS SINCE edge["attr"] = ... JUST UPDATES THE ATTRIBUTE VALUE


        return ...


    def get_data(self) -> dict[Any, Any]:
        return self.__dict__


    def load_nodes(self) -> None:
        all(self._network.add_nodes_from((row["id"], row.to_dict().pop("id")) for row in partition) for partition in self._loader.get_nodes().partitions)
        return None


    def load_links(self) -> None:
        all(self._network.add_edges_from(
                (row["start_traffic_node_id"], row["end_traffic_node_id"],
                 {k: v for k, v in row.to_dict().items() if k not in ["start_traffic_node_id", "end_traffic_node_id"]})
                for row in partition
            ) for partition in self._loader.get_links().partitions)
        return None


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









