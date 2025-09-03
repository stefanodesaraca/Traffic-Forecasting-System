import json
import pickle
from itertools import chain
import numpy as np
import networkx as nx
from typing import Any, Generator, Mapping
from pydantic.types import PositiveFloat, PositiveInt
from scipy.spatial.distance import minkowski
import matplotlib.pyplot as plt
from shapely import wkt
import dask.dataframe as dd
import utm
from pykrige.ok import OrdinaryKriging

from definitions import GlobalDefinitions, ProjectTables, ProjectMaterializedViews, TrafficClasses, IconStyles
from loaders import BatchStreamLoader
from pipelines import MLPreprocessingPipeline, MLPredictionPipeline
from utils import save_plot





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
        self._network: nx.DiGraph | None = None #TODO CORRECTLY DEFINE ROAD DIRECTIONS, BUT WITH AN UNDIRECTED GRAPH EVERYTHING WORKS CORRECTLY

        if not network_binary:
            self._network = nx.DiGraph(**{"network_id": self.network_id, "name": self.name})
        else:
            self._network = pickle.loads(network_binary)  # To load pre-computed graphs


    @staticmethod
    def _get_minkowski_dist(u: tuple[Any, ...], v: tuple[Any, ...], G: nx.DiGraph):
        # Parsing WKT geometry into shapely Point
        u_geom = G.nodes[u]["geom"]
        v_geom = G.nodes[v]["geom"]

        u_point = wkt.loads(u_geom) if isinstance(u_geom, str) else u_geom
        v_point = wkt.loads(v_geom) if isinstance(v_geom, str) else v_geom

        # Converting lat/lon to UTM
        u_easting, u_northing, _, _ = utm.from_latlon(u_point.y, u_point.x)
        v_easting, v_northing, _, _ = utm.from_latlon(v_point.y, v_point.x)

        return minkowski([u_easting, u_northing], [v_easting, v_northing], p=2.0) #TODO 1.0 if G.edges[u, v]["road_category"] in GlobalDefinitions.HIGH_SPEED_ROAD_CATEGORIES else


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


    def _get_neighbor_links(self, link_id: str, buffer_zone_radius: PositiveInt) -> dict[str, Any]:
        return self._db_broker.send_sql(
            f""" SELECT rgl_b.* FROM "{ProjectTables.RoadGraphLinks}" rgl_a 
            JOIN "{ProjectTables.RoadGraphLinks}" rgl_b ON rgl_a.link_id = %s 
            AND rgl_a.link_id <> rgl_b.link_id 
            WHERE ST_DWithin(rgl_a.geom::geography, rgl_b.geom::geography, %s);""",
            execute_args=[link_id, buffer_zone_radius])


    def _get_neighbor_trps(self, link_id: str, buffer_zone_radius: PositiveInt = 3500) -> list[dict[str, Any]]:
        return self._db_broker.send_sql(f"""
            WITH neighbors AS (
                SELECT rgl_b.link_id AS neighbor_link_id
                FROM "{ProjectTables.RoadGraphLinks.value}" rgl_a
                JOIN "{ProjectTables.RoadGraphLinks.value}" rgl_b
                  ON rgl_a.link_id = %s
                 AND rgl_a.link_id <> rgl_b.link_id
                WHERE ST_DWithin(rgl_a.geom::geography, rgl_b.geom::geography, %s)
            )
            SELECT 
                trp.id,
                trp.road_category
                trp.lat
                trp.lon
            FROM neighbors n
            JOIN "{ProjectTables.RoadLink_TrafficRegistrationPoints.value}" rl_t
              ON n.neighbor_link_id = rl_t.link_id
            JOIN "{ProjectTables.TrafficRegistrationPoints.value}" trp
              ON rl_t.trp_id = trp.trp_id;
        """, execute_args=[link_id, buffer_zone_radius])


    def _get_link_aggregated_traffic_data(
            self, link_id: str,
            county_avg_weight: PositiveFloat = 0.25,
            municipality_avg_weight: PositiveFloat = 0.5,
            road_category_avg_weight: PositiveFloat = 0.25
    ) -> dict[str, PositiveFloat | PositiveInt]:
        if round(county_avg_weight + municipality_avg_weight + road_category_avg_weight, 6) != 1.0:
            raise ValueError("Weights sum must be exactly 1")
        return self._db_broker.send_sql(
            f"""
                WITH link_agg_data AS (
                    SELECT l.link_id, rlm.municipality_id, rlc.county_id, l.road_category
                    FROM "{ProjectTables.RoadGraphLinks.value}" l
                    LEFT JOIN "{ProjectTables.RoadLink_Municipalities.value}" rlm ON l.link_id = rlm.link_id
                    LEFT JOIN "{ProjectTables.RoadLink_Counties.value}" rlc ON l.link_id = rlc.link_id
                    WHERE l.link_id = %s
                ),
                county_avg AS (
                    SELECT l_agg.link_id,
                           AVG(c.avg_{GlobalDefinitions.VOLUME}_by_county) AS avg_{GlobalDefinitions.VOLUME},
                           AVG(c.avg_{GlobalDefinitions.MEAN_SPEED}_by_county) AS avg_{GlobalDefinitions.MEAN_SPEED}
                    FROM link_agg_data l_agg
                    JOIN "{ProjectMaterializedViews.TrafficDataByCountyMView.value}" c ON l_agg.county_id = c.county_id
                    GROUP BY l_agg.link_id
                ),
                municipality_avg AS (
                    SELECT l_agg.link_id,
                           AVG(m.avg_{GlobalDefinitions.VOLUME}_by_municipality) AS avg_{GlobalDefinitions.VOLUME},
                           AVG(m.avg_{GlobalDefinitions.MEAN_SPEED}_by_municipality) AS avg_{GlobalDefinitions.MEAN_SPEED}
                    FROM link_agg_data l_agg
                    JOIN "{ProjectMaterializedViews.TrafficDataByMunicipalityMView.value}" m ON l_agg.municipality_id = m.municipality_id
                    GROUP BY l_agg.link_id
                ),
                road_category_avg AS (
                    SELECT l_agg.link_id,
                           AVG(r.avg_{GlobalDefinitions.VOLUME}_by_road_category) AS avg_{GlobalDefinitions.VOLUME},
                           AVG(r.avg_{GlobalDefinitions.MEAN_SPEED}_by_road_category) AS avg_{GlobalDefinitions.MEAN_SPEED}
                    FROM link_agg_data l_agg
                    JOIN "{ProjectMaterializedViews.TrafficDataByRoadCategoryMView.value}" r ON l_agg.road_category = r.road_category
                    GROUP BY l_agg.link_id
                )
                SELECT
                    l_agg.link_id,
                    %s * county_avg.avg_{GlobalDefinitions.VOLUME} +
                    %s * municipality_avg.avg_{GlobalDefinitions.VOLUME} +
                    %s * road_category_avg.avg_{GlobalDefinitions.VOLUME} AS weighted_avg_{GlobalDefinitions.VOLUME},
                    %s * county_avg.avg_{GlobalDefinitions.MEAN_SPEED} +
                    %s * municipality_avg.avg_{GlobalDefinitions.MEAN_SPEED} +
                    %s * road_category_avg.avg_{GlobalDefinitions.MEAN_SPEED} AS weighted_avg_{GlobalDefinitions.MEAN_SPEED}
                FROM link_agg_data l_agg
                LEFT JOIN county_avg ON l_agg.link_id = county_avg.link_id
                LEFT JOIN municipality_avg ON l_agg.link_id = municipality_avg.link_id
                LEFT JOIN road_category_avg ON l_agg.link_id = road_category_avg.link_id;
            """,
            execute_args=[
                link_id,
                county_avg_weight, municipality_avg_weight, road_category_avg_weight,
                county_avg_weight, municipality_avg_weight, road_category_avg_weight,
            ],
            single=True
        )
        #Returning the average value of each target variable aggregated respectively by any counties, municipalities and road categories the link may belong to, this way we'll have a customized indicator of what are the "normal" (average) conditions on that road


    def _get_trp_predictions(self, trp_id: str, target: str, road_category: str) -> Generator[dd.DataFrame, None, None] | dd.DataFrame:
        #return (
        #    MLPredictionPipeline(trp_id=trp_id, target=target, road_category=road_category, model=model, preprocessing_pipeline=MLPreprocessingPipeline(), loader=self._loader, db_broker=self._db_broker).start()
        #    for model in self._db_broker.get_trained_model_objects(target=target, road_category=road_category)
        #)
        models = {m["name"]: pickle.loads(m["pickle_object"]) for m in self._db_broker.get_trained_model_objects(target=target, road_category=road_category)}
        return MLPredictionPipeline(trp_id=trp_id, target=target, road_category=road_category, model=models["HistGradientBoostingRegressor"], preprocessing_pipeline=MLPreprocessingPipeline(), loader=self._loader, db_broker=self._db_broker).start()


    @staticmethod
    def _get_increments(n: int | float, k: PositiveInt) -> list[int | float]:
        step = n // k
        return [step * i for i in range(1, k + 1)]


    @staticmethod
    def _closest(numbers: list[int | float], k: int | float):
        return min(numbers, key=lambda n: abs(n - k))


    def _compute_edge_weight(self, edge_start: callable, edge_end: callable, attrs: dict) -> PositiveFloat | None:
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
            neighborhood_trps = self._get_neighbor_trps(attrs["link_id"], buffer_zone_radius=neighborhood_radius)
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

        sp = nx.astar_path(G=self._network, source=source, target=destination, heuristic=lambda u, v: self._get_minkowski_dist(u, v, self._network), weight="length")
        sp_edges = [(u, v, self._network.get_edge_data(u, v)) for u, v in zip(sp, sp[1:])] #TODO TEST IF THIS WORKS WITH A GENERATOR AS WELL CALLING THE find_route() METHOD TWICE IN THE SAME RUNTIME

        print(sp)
        print(sp_edges)


        # ---------- STEP 2 ----------

        buffer_radius = 3500
        trps_per_edge = {}
        while True:
            for e in sp_edges:
                trps_per_edge[e[2]["link_id"]] = self._get_neighbor_trps(e[2]["link_id"], buffer_radius)
            if sum(len(v) for v in trps_per_edge) >= min(3, len(sp) // 3):
                break
            buffer_radius += 2000

        trps_along_sp: set[dict[str, Any]] = set(*chain(trps_per_edge.values())) # A set of dictionary where each dict is a TRP with its metadata (id, road_category, etc.)


        # ---------- STEP 3 ----------

        link_agg_data = ((link_id, self._get_link_aggregated_traffic_data(link_id=link_id)) for link_id in trps_per_edge.keys())

        trps_along_sp_preds = {
            trp["trp_id"]: {
                f"{GlobalDefinitions.VOLUME}_preds": self._get_trp_predictions(trp_id=trp["trp_id"],
                                                                               target=GlobalDefinitions.VOLUME,
                                                                               road_category=trp["road_category"])[[GlobalDefinitions.VOLUME, "zoned_dt_iso"]],
                f"{GlobalDefinitions.MEAN_SPEED}_preds": self._get_trp_predictions(trp_id=trp["trp_id"],
                                                                                   target=GlobalDefinitions.MEAN_SPEED,
                                                                                   road_category=trp["road_category"])[[GlobalDefinitions.MEAN_SPEED, "zoned_dt_iso"]],
                **trp,
            }
            for trp in trps_along_sp
        }
        #TODO IF TRP PREDICTIONS ARE IN THE RADIUS OF ANOTHER TRP, JUST REUSE THEM AND DO NOT EXECUTE ML PREDICTIONS AGAIN


        # ---------- STEP 4 ----------

        lons = [trp_data["lon"] for trp_data in trps_along_sp_preds.values()]
        lats = [trp_data["lat"] for trp_data in trps_along_sp_preds.values()]

        def get_ok_structured_data(target: str):
            return dd.from_dict({
                idx: {
                    f"{target}": row[target],
                    "zoned_dt_iso": row["zoned_dt_iso"],
                    "lon": trp_data.get("lon"),
                    "lat": trp_data.get("lat"),
                }
                for trp_data in trps_along_sp_preds.values()
                for idx, row in enumerate(trp_data[f"{target}_preds"].itertuples(index=False), start=0)
            })

        def get_ordinary_kriging(target: str, verbose: bool = False):
            ok_df = get_ok_structured_data(target=GlobalDefinitions.target)
            return OrdinaryKriging(
                x=ok_df["lon"].values,
                y=ok_df["lat"].values,
                z=ok_df[target].values,
                variogram_model="spherical",
                coordinates_type="geographical",
                verbose=verbose,
                enable_plotting=True
            ).execute(
                style="grid",
                xpoints=np.linspace(min(lats)-0.10, max(lats)+0.10, 100),
                ypoints=np.linspace(min(lons)-0.35, max(lons)+0.35, 100)
            )




        #TODO THE GRIDS IN EXECUTE WILL HAVE TO MATCH THE PATH LINE BUFFER RADIUS
        #NOTE EXECUTE OrdinaryKriging FOR ANY OF THE HOURLY FORECASTED DATA SO WE CAN SHOW THE DIFFERENCE IN TRAFFIC BETWEEN HOURS ALONG THE SHORTEST PATH








        #TODO CREATE INTERPOLATION CHARTS FOR OSLO, BERGEN, TRONDHEIM, BODO, TROMSO, STAVANGER, ALTA, ETC. BY EXECUTING FORECASTS FOR EACH TRP IN THOSE CITIES (BY USING MUNICIPALITY ID)










        #TODO DETERMINE CLASSES AFTER ORDINARY KRIGING BY INTERROGATING THE KRIGING RESULTS FOR A CERTAIN AMOUNT OF COORDINATES OF THE WHOLE SHORTEST PATH (OR A CERTAIN AMOUNT OF EACH LINK) AND DETERMINE IF THEY ARE ABOVE OR UNDER AVERAGE TRAFFIC
        #TODO TO CHECK TO WHICH CLASS IT BELONGS TO, USE self._closest()
        for link_id, data in trps_along_sp_preds.items():
            trps_along_sp_preds[link_id][f"link_traffic_{GlobalDefinitions.VOLUME}_classes"] = self._get_increments(n=trps_along_sp_preds[link_id]["link_traffic_averages"][f"weighted_{GlobalDefinitions.VOLUME}_avg"] * 2, k=len(TrafficClasses))
            trps_along_sp_preds[link_id][f"link_traffic_{GlobalDefinitions.MEAN_SPEED}_classes"] = self._get_increments(n=trps_along_sp_preds[link_id]["link_traffic_averages"][f"weighted_{GlobalDefinitions.MEAN_SPEED}_avg"] * 2, k=len(TrafficClasses))











        # Adding removed nodes back into the graph to avoid needing to re-build the whole graph again
        self._network.add_nodes_from(unreachable)

        # Adding removed edges back into the graph to avoid needing to re-build the whole graph again
        if has_only_public_transport_lanes:
            self._network.add_edges_from(has_only_public_transport_lanes_edges)
        if has_toll_stations:
            self._network.add_edges_from(has_toll_stations_edges)
        if has_ferry_routes:
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
        fig, ax = plt.subplots(figsize=(16, 9))
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
                for node in self._network.nodes
            ]
        )
        return fig, f"{self._network['network_id']}.svg"




