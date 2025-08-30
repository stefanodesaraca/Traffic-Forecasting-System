import json
import pickle
import networkx as nx
from typing import Any, Iterator, Literal, Hashable
from pydantic.types import PositiveFloat, PositiveInt
from scipy.spatial.distance import cityblock, minkowski  # Scipy's cityblock distance is the Manhattan distance. Scipy distance docs: https://docs.scipy.org/doc/scipy/reference/spatial.distance.html#module-scipy.spatial.distance
from geopy.distance import geodesic # To calculate distance (in meters) between two sets of coordinates (lat-lon). Geopy distance docs: https://geopy.readthedocs.io/en/stable/#module-geopy.distance
import matplotlib.pyplot as plt
from pykrige import OrdinaryKriging

from exceptions import WrongGraphProcessingBackendError
from definitions import GlobalDefinitions, ProjectTables, ProjectMaterializedViews, IconStyles
from loaders import BatchStreamLoader
from utils import save_plot

Node = Hashable  # Any object usable as a node
Edge = Hashable  # Any object usable as an edge


class RoadNetwork:

    def __init__(self,
                 network_id: str,
                 name: str,
                 backend: str = "networkx",
                 db_broker: Any | None = None,
                 loader: BatchStreamLoader | None = None,
                 network_binary: bytes | None = None
    ):
        self._loader: BatchStreamLoader = loader
        self._db_broker: Any = db_broker #Synchronous DBBroker
        self.network_id: str = network_id
        self.name: str = name
        self._backend: str = backend
        self._network: nx.DiGraph | None = None

        if not network_binary:
            self._network = nx.DiGraph(**{"network_id": self.network_id, "name": self.name})
        else:
            self._network = pickle.loads(network_binary)  # To load pre-computed graphs


    def _find_trps_within_link_buffer_zone(self, link_id: str, buffer_zone_radius: PositiveInt = 3500) -> list:
        return self._db_broker.send_sql(f"""
                SELECT rgl_b.*
                FROM {ProjectTables.RoadGraphLinks} rgl_a
                JOIN {ProjectTables.RoadGraphLinks} rgl_b
                  ON rgl_a.link_id = %s
                  AND rgl_a.link_id <> rgl_b.link_id
                WHERE ST_DWithin(rgl_a.geom::geography, rgl_b.geom::geography, %s);
            """, execute_args=[link_id, buffer_zone_radius])


    def _find_link_aggregated_traffic_data(self, link_id: str, county_avg_weight: PositiveFloat = 0.25, municipality_avg_weight: PositiveFloat = 0.5, road_category_avg_weight: PositiveFloat = 0.25) -> dict[str, PositiveFloat | PositiveInt]:
        if sum([county_avg_weight, municipality_avg_weight, road_category_avg_weight]) > 1:
            raise ValueError("Weights sum must be at most 1")
        return self._db_broker.send_sql(f"""
            WITH link_agg_data AS (
                SELECT l.link_id,
                       rlm.municipality_id,
                       rlc.county_id,
                       l.road_category
                FROM "{ProjectTables.RoadGraphLinks.value}" l
                LEFT JOIN "{ProjectTables.RoadLink_Municipalities.value}" rlm
                    ON l.link_id = rlm.link_id
                LEFT JOIN "{ProjectTables.RoadLink_Counties.value}" rlc
                    ON l.link_id = rlc.link_id
            )
            
            SELECT 
                0.25 * county_avg.avg_{GlobalDefinitions.VOLUME} +
                0.5  * municipality_avg.avg_{GlobalDefinitions.VOLUME} +
                0.25 * road_category_avg.avg_{GlobalDefinitions.VOLUME} AS weighted_avg_{GlobalDefinitions.VOLUME}
                
                0.25 * county_avg.avg_{GlobalDefinitions.MEAN_SPEED} +
                0.5  * municipality_avg.avg_{GlobalDefinitions.MEAN_SPEED} +
                0.25 * road_category_avg.avg_{GlobalDefinitions.MEAN_SPEED} AS weighted_avg_{GlobalDefinitions.MEAN_SPEED}
                
            FROM link_agg_data l_agg
            LEFT JOIN "{ProjectMaterializedViews.TrafficDataByCountyMView.value}" county_avg
                ON l_agg.county_id = county_avg.county_id
            LEFT JOIN "{ProjectMaterializedViews.TrafficDataByMunicipalityMView.value}" municipality_avg
                ON l_agg.municipality_id = municipality_avg.municipality_id
            LEFT JOIN "{ProjectMaterializedViews.TrafficDataByRoadCategoryMView.value}" road_category_avg
                ON l_agg.road_category = road_category_avg.road_category;
        """, execute_args=[link_id], single=True)


    @staticmethod
    def _compute_edge_weight(edge_start: Node, edge_end: Node, attrs: dict) -> PositiveFloat | None:
        length: PositiveFloat = attrs.get("length")
        road_category: str = attrs.get("road_category")
        min_lanes: PositiveInt = attrs.get("min_lanes")
        max_lanes: PositiveInt = attrs.get("max_lanes")
        highest_speed_limit: PositiveInt = attrs.get("highest_speed_limit")
        lowest_speed_limit: PositiveInt = attrs.get("lowest_speed_limit")
        is_ferry_route: bool = attrs.get("is_ferry_route")













        #TODO DEPENDING ON THE ROAD CATEGORY THE AVERAGE VALUE WE'RE GOING TO TAKE INTO CONSIDERATION FOR THE AVERAGE SPEED OF THE ROAD CATEGORY WILL BE CUSTOMIZED BASED ON THE COUNTY AND THE MUNICIPALITY OF THE LINK
        #TODO WE'LL FIRST CREATE A COUNTY AVERAGE AND MUNICIPALITY AVERAGE (MULTIPLE MUNICIPALITY AVERAGES IF THE LINK BELONGS TO MULTIPLE MUNICIPALITIES) AND THEN GIVE MORE WEIGHT TO THE MUNICIPALITY ONES, BUT KEEPING STILL THE COUNTY AVERAGE





        return ...





    def load_nodes(self) -> None:
        all(self._network.add_nodes_from((row.to_dict().pop("node_id"), row) for _, row in partition.iterrows()) for partition in self._loader.get_nodes().partitions)
        return None


    def load_links(self, county_ids_filter: list[str] | None = None) -> None:
        all(self._network.add_edges_from(
                (row["start_traffic_node_id"], row["end_traffic_node_id"],
                 {k: v for k, v in row.to_dict().items() if k not in ["start_traffic_node_id", "end_traffic_node_id"]})
                for _, row in partition.iterrows()
            ) for partition in self._loader.get_links(county_ids_filter=county_ids_filter, has_trps=True).partitions)
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


    def find_route(self,
                   source: str,
                   destination: str,
                   avoid_only_public_transport_lanes: bool | None = None,
                   avoid_toll_stations: bool | None = None,
                   avoid_ferry_routes: bool | None = None
    ) -> list[tuple[str, str]]:

        # ---------- STEP 1 ----------




















        #TODO THE FORECASTED TRAVEL TIME WILL BE TOTAL LENGTH IN METERS OF THE WHOLE LINESTRING DIVIDED BY 85% OF THE MAX SPEED LIMIT + 30s * (85% OF THE TOTAL NUMBER OF NODES THAT THE USER WILL PASS THROUGH, SINCE EACH ONE IS AN INTERSECTION AND PROBABLY 85% HAVE A TRAFFIC LIGHT)

        #TODO TRY TO GET AT LEAST 2 TRPS
        # IF NONE ARE FOUND REPEAT THE RESEARCH, BUT INCREASE THE RADIUS BY 1500m

        return ...





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
        return nx.astar_path(G=self._network, source=source, target=target, heuristic={"manhattan": cityblock, "minkowski": minkowski}[heuristic], weight=weight)


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


    def to_pickle(self) -> None:
        with open(f'{self._network["network_id"]}.gpickle', 'wb') as fp:
            pickle.dump(self._network, fp, pickle.HIGHEST_PROTOCOL)
        return None


    def to_db(self) -> None:
        self._db_broker.send_sql(f"""
            INSERT INTO "{ProjectTables.RoadNetworks.value}" ("name", "binary_obj")
            VALUES (%s, %s)
            ON CONFLICT DO UPDATE SET "binary_obj" = EXCLUDED.binary_obj;;
        """, execute_args=[self._network["network_id"], pickle.dumps(self._network, pickle.HIGHEST_PROTOCOL)])
        return None


    @save_plot
    def save_graph_svg(self) -> tuple[plt, str]:
        fig, ax = plt.subplots(figsize=(8, 6))
        nx.draw(
            self._network,
            ax=ax,
            with_labels=True,
            node_size=1500,
            font_size=25,
            font_color="yellow",
            font_weight="bold",
            edge_color=[
                IconStyles.TRP_LINK_STYLE.value.get("icon_color")
                if self._network.nodes[node].get("has_trps") else "blue"
                for node in self._network.nodes()
            ]
        )
        return fig, f"{self._network['network_id']}.svg"




