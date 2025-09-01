import json
import pickle
import networkx as nx
from typing import Any, Iterator, Literal, Hashable
from pydantic.types import PositiveFloat, PositiveInt
from scipy.spatial.distance import cityblock, minkowski  # Scipy's cityblock distance is the Manhattan distance. Scipy distance docs: https://docs.scipy.org/doc/scipy/reference/spatial.distance.html#module-scipy.spatial.distance
import matplotlib.pyplot as plt
from pykrige import OrdinaryKriging

from definitions import GlobalDefinitions, ProjectTables, ProjectMaterializedViews, IconStyles
from loaders import BatchStreamLoader
from utils import save_plot

Node = Hashable  # Any object usable as a node


from pprint import pprint




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


    @staticmethod
    def _get_minkowski_dist(u: tuple[Any, ...], v: tuple[Any, ...], G: nx.DiGraph):
        return minkowski(G.nodes[u], G.nodes[v], p=1.0 if G.edges[u, v]["road_category"] in GlobalDefinitions.HIGH_SPEED_ROAD_CATEGORIES else 2.0)


    @staticmethod
    def _get_trait_length_by_road_category(links: list[callable]) -> dict[str, float]:
        lengths = {}
        for link in links:
            lengths[link["road_category"]] += link["length"]
        return lengths


    @staticmethod
    def _get_trait_main_road_category(grouped_trait: dict[str, float]) -> str:
        return max(grouped_trait, key=grouped_trait.get)


    @staticmethod
    def _get_road_category_proportions(grouped_trait: dict[str, float]) -> dict[str, dict[str, float]]:
        total = sum(grouped_trait.values())
        stats = {}
        for category, length in grouped_trait.items():
            stats[category] = {
                "length": length,
                "percentage": (length / total) * 100
            }
        return stats


    def _find_trps_within_link_buffer_zone(self, link_id: str, buffer_zone_radius: PositiveInt = 3500) -> list:
        return self._db_broker.send_sql(f"""
                SELECT rgl_b.*
                FROM "{ProjectTables.RoadGraphLinks}" rgl_a
                JOIN "{ProjectTables.RoadGraphLinks}" rgl_b
                  ON rgl_a.link_id = %s
                  AND rgl_a.link_id <> rgl_b.link_id
                WHERE ST_DWithin(rgl_a.geom::geography, rgl_b.geom::geography, %s);
            """, execute_args=[link_id, buffer_zone_radius])


    def _find_link_aggregated_traffic_data(self, link_id: str, county_avg_weight: PositiveFloat = 0.25, municipality_avg_weight: PositiveFloat = 0.5, road_category_avg_weight: PositiveFloat = 0.25) -> dict[str, PositiveFloat | PositiveInt]:
        if not sum([county_avg_weight, municipality_avg_weight, road_category_avg_weight]) == 1:
            raise ValueError("Weights sum must be exactly 1")
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
                WHERE l.link_id = %s
            ),
            county_avg AS (
                SELECT 
                    l_agg.link_id, 
                    AVG(c.avg_{GlobalDefinitions.VOLUME}_by_county) AS avg_{GlobalDefinitions.VOLUME},          
                    AVG(c.avg_{GlobalDefinitions.MEAN_SPEED}_by_county) AS avg_{GlobalDefinitions.MEAN_SPEED}
                FROM link_agg_data l_agg
                JOIN "{ProjectMaterializedViews.TrafficDataByCountyMView.value}" c
                  ON l_agg.county_id = c.county_id
                GROUP BY l_agg.link_id
            ),
            municipality_avg AS (
                SELECT 
                    l_agg.link_id, 
                    AVG(m.avg_{GlobalDefinitions.VOLUME}_by_municipality) AS avg_{GlobalDefinitions.VOLUME},
                    AVG(m.avg_{GlobalDefinitions.MEAN_SPEED}_by_municipality) AS avg_{GlobalDefinitions.MEAN_SPEED}
                FROM link_agg_data l_agg
                JOIN "{ProjectMaterializedViews.TrafficDataByMunicipalityMView.value}" m
                  ON l_agg.municipality_id = m.municipality_id
                GROUP BY l_agg.link_id
            ),
            road_category_avg AS (
                SELECT 
                    l_agg.link_id,
                    AVG(r.avg_{GlobalDefinitions.VOLUME}_by_road_category) AS avg_{GlobalDefinitions.VOLUME},
                    AVG(r.avg_{GlobalDefinitions.MEAN_SPEED}_by_road_category) AS avg_{GlobalDefinitions.MEAN_SPEED}
                FROM link_agg_data l_agg
                JOIN "{ProjectMaterializedViews.TrafficDataByRoadCategoryMView.value}" r
                  ON l_agg.road_category = r.road_category
                GROUP BY l_agg.link_id
            )
            
            SELECT
                l_agg.link_id,
                0.25 * county_avg.avg_{GlobalDefinitions.VOLUME} +
                0.5  * municipality_avg.avg_{GlobalDefinitions.VOLUME} +
                0.25 * road_category_avg.avg_{GlobalDefinitions.VOLUME} AS weighted_avg_{GlobalDefinitions.VOLUME},
                0.25 * county_avg.avg_{GlobalDefinitions.MEAN_SPEED} +
                0.5  * municipality_avg.avg_{GlobalDefinitions.MEAN_SPEED} +
                0.25 * road_category_avg.avg_{GlobalDefinitions.MEAN_SPEED} AS weighted_avg_{GlobalDefinitions.MEAN_SPEED}
            FROM (SELECT DISTINCT link_id FROM link_info) l_agg
            LEFT JOIN county_avg ON l_agg.link_id = county_avg.link_id
            LEFT JOIN municipality_avg ON l_agg.link_id = municipality_avg.link_id
            LEFT JOIN road_category_avg ON l_agg.link_id = road_category_avg.link_id;
        """, execute_args=[link_id], single=True)
        #Returning the average value of each target variable aggregated respectively by any counties, municipalities and road categories the link may belong to, this way we'll have a customized indicator of what are the "normal" (average) conditions on that road


    def _compute_edge_weight(self, edge_start: Node, edge_end: Node, attrs: dict) -> PositiveFloat | None:
        length: PositiveFloat = attrs.get("length")
        road_category: str = attrs.get("road_category")
        min_lanes: PositiveInt = attrs.get("min_lanes")
        max_lanes: PositiveInt = attrs.get("max_lanes")
        highest_speed_limit: PositiveInt = attrs.get("highest_speed_limit")
        lowest_speed_limit: PositiveInt = attrs.get("lowest_speed_limit")
        is_ferry_route: bool = attrs.get("is_ferry_route")

        neighborhood_radius = 3500 #Distance in meters from the link line, so the diameter of the whole buffer zone is neighborhood_radius * 2
        neighborhood_trps = []

        while len(neighborhood_trps) == 0:
            neighborhood_trps = self._find_trps_within_link_buffer_zone(attrs["link_id"], buffer_zone_radius=neighborhood_radius)
            neighborhood_radius += 1000









        #TODO DEPENDING ON THE ROAD CATEGORY THE AVERAGE VALUE WE'RE GOING TO TAKE INTO CONSIDERATION FOR THE AVERAGE SPEED OF THE ROAD CATEGORY WILL BE CUSTOMIZED BASED ON THE COUNTY AND THE MUNICIPALITY OF THE LINK
        #TODO WE'LL FIRST CREATE A COUNTY AVERAGE AND MUNICIPALITY AVERAGE (MULTIPLE MUNICIPALITY AVERAGES IF THE LINK BELONGS TO MULTIPLE MUNICIPALITIES) AND THEN GIVE MORE WEIGHT TO THE MUNICIPALITY ONES, BUT KEEPING STILL THE COUNTY AVERAGE





        return ...


    def load_nodes(self) -> None:
        all(self._network.add_nodes_from((row.to_dict().pop("node_id"), row) for _, row in partition.iterrows()) for partition in self._loader.get_nodes().partitions)
        return None


    def load_links(self,
                   county_ids_filter: list[str] | None = None,
                   has_only_public_transport_lanes: bool | None = None,
                   has_toll_stations: bool | None = None,
                   has_ferry_routes: bool | None = None) -> None:
            all(self._network.add_edges_from(
                    (row["start_traffic_node_id"], row["end_traffic_node_id"],
                     {k: v for k, v in row.to_dict().items() if k not in ["start_traffic_node_id", "end_traffic_node_id"]})
                    for _, row in partition.iterrows()
                ) for partition in self._loader.get_links(
                    county_ids_filter=county_ids_filter,
                    has_only_public_transport_lanes_filter=has_only_public_transport_lanes,
                    has_toll_stations=has_toll_stations,
                    has_ferry_routes=has_ferry_routes
                ).partitions
            )
            return None


    def build(self,
              auto_load_nodes: bool = True,
              auto_load_links: bool = True,
              county_ids_filter: list[str] | None = None,
              has_only_public_transport_lanes: bool | None = None,
              has_toll_stations: bool | None = None,
              has_ferry_routes: bool | None = None,
              verbose: bool = True) -> None:

        if auto_load_nodes:
            if verbose:
                print("Loading nodes...")
            self.load_nodes()

        if auto_load_links:
            if verbose:
                print("Loading links...")
            self.load_links(
                county_ids_filter=county_ids_filter,
                has_only_public_transport_lanes=has_only_public_transport_lanes,
                has_toll_stations=has_toll_stations,
                has_ferry_routes=has_ferry_routes,
            )

        # NOTE WHEN WE'LL HAVE THE ABILITY TO FILTER DIRECTLY AT THE SOURCE OF THE NODES (WHEN WE'LL HAVE THE MUNICIPALITY AND COUNTY DATA ON THE NODES) WE'LL JUST NOT LOAD THE ONES OUTSIDE OF THE FILTERS CONDITIONS

        if verbose:
            print("Road network graph created!")

        print(self._network)
        print(list(self._network.edges(data=True))[0])

        return None


    def find_route(self,
                   source: str,
                   destination: str,
                   has_only_public_transport_lanes: bool | None = None,
                   has_toll_stations: bool | None = None,
                   has_ferry_routes: bool | None = None
    ) -> list[tuple[str, str]]:

        has_only_public_transport_lanes_edges = None
        has_toll_stations_edges = None
        has_ferry_routes_edges = None

        if has_only_public_transport_lanes:
            self._network.remove_edges_from(has_only_public_transport_lanes_edges := [(u, v) for u, v, attrs in self._network.edges(data=True) if attrs['has_only_public_transport_lanes'] < 3])
        if has_toll_stations:
            self._network.remove_edges_from(has_toll_stations_edges := [(u, v) for u, v, attrs in self._network.edges(data=True) if attrs['has_toll_stations'] < 3])
        if has_ferry_routes:
            self._network.remove_edges_from(has_ferry_routes_edges := [(u, v) for u, v, attrs in self._network.edges(data=True) if attrs['has_ferry_routes'] < 3])

        #Removing isolated nodes since they're unreachable
        self._network.remove_nodes_from(unreachable := list(i for i in nx.isolates(self._network)))
        #This way we're also going to apply all filters if any have been set since if there isn't a link connecting a node to another that will be isolated


        # ---------- STEP 1 ----------

        sp = nx.astar_path(G=self._network, source=source, target=destination, heuristic=lambda u, v: self._get_minkowski_dist(u, v, self._network), weight="length") #TODO IF THE cityblock OR minkowski FUNCTIONS DON'T WORK BECAUSE THE WEIGHT THEY RECEIVE IS ACTUALLY THE ATTRS DICT THEN CREATE A WRPPER FUNCTION WHICH JUST RETURNS THE DISTANCE AND ACCEPTS THE ATTRS DICT AS WELL, BUT DOESN'T INSERT THAT WITHING THE DISTANCE CALCULATION FUNCTION


        print(sp)





        # Adding removed nodes back into the graph to avoid needing to re-build the whole graph again
        self._network.add_nodes_from(unreachable)

        # Adding removed edges back into the graph to avoid needing to re-build the whole graph again
        self._network.add_edges_from(has_only_public_transport_lanes_edges)
        self._network.add_edges_from(has_toll_stations_edges)
        self._network.add_edges_from(has_ferry_routes_edges)

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




