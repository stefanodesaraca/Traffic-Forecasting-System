import datetime
import pickle
from itertools import chain
from typing import Any, Generator, Literal, Mapping
from scipy.spatial.distance import minkowski
from shapely import wkt
from pydantic.types import PositiveFloat, PositiveInt
import networkx as nx
import numpy as np
import pandas as pd
import dask.dataframe as dd
import utm
from pykrige.ok import OrdinaryKriging
import matplotlib.pyplot as plt


from definitions import GlobalDefinitions, ProjectTables, ProjectMaterializedViews, TrafficClasses, IconStyles, RoadCategoryTraitLengthWeightMultipliers
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

        self.trps_along_sp: set[dict[str, Any]] | None = None
        self.trps_along_sp_preds: dict[str, Any] | None = {}


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


    def _get_trp_predictions(self, trp_id: str, target: str, road_category: str, lags: list[PositiveInt]) -> Generator[dd.DataFrame, None, None] | dd.DataFrame:
        #return (
        #    MLPredictionPipeline(trp_id=trp_id, target=target, road_category=road_category, model=model, preprocessing_pipeline=MLPreprocessingPipeline(), loader=self._loader, db_broker=self._db_broker).start()
        #    for model in self._db_broker.get_trained_model_objects(target=target, road_category=road_category)
        #)
        models = {m["name"]: pickle.loads(m["pickle_object"]) for m in self._db_broker.get_trained_model_objects(target=target, road_category=road_category)}
        return MLPredictionPipeline(trp_id=trp_id, target=target, road_category=road_category, model=models["HistGradientBoostingRegressor"], preprocessing_pipeline=MLPreprocessingPipeline(), loader=self._loader, db_broker=self._db_broker).start(lags=lags)


    @staticmethod
    def _get_increments(n: int | float, k: PositiveInt) -> list[int | float]:
        step = n // k
        return [step * i for i in range(1, k + 1)]


    @staticmethod
    def _closest(numbers: list[int | float], k: int | float):
        return min(numbers, key=lambda n: abs(n - k))


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


    def _get_shortest_path(self, source: str, destination: str, heuristic: callable, weight: str | type[callable]) -> list[str]:
        return nx.astar_path(G=self._network, source=source, target=destination, heuristic=heuristic, weight=weight)


    def _get_path_edges(self, p: list[str]) -> list[tuple[str, str, Mapping[str, Any]]]:
        return [(u, v, self._network.get_edge_data(u, v)) for u, v in zip(p, p[1:])] #TODO TEST IF THIS WORKS WITH A GENERATOR AS WELL CALLING THE find_route() METHOD TWICE IN THE SAME RUNTIME


    def _get_trps_per_edge(self, edges: list[tuple[str, str, Mapping[str, Any]]], trp_research_buffer_radius: PositiveInt) -> dict[str, list[dict[str, Any]]]:
        trps_per_edge = {}
        while True:
            for e in edges:
                trps_per_edge[e[2]["link_id"]] = self._get_neighbor_trps(e[2]["link_id"], trp_research_buffer_radius)
            if sum(len(v) for v in trps_per_edge) >= min(3, len(edges) // 3):
                return trps_per_edge
            trp_research_buffer_radius += 2000


    @staticmethod
    def _get_ok_structured_data(trps_along_path_preds: dict[str, dict[str, dd.DataFrame]], time_range: list[datetime.datetime], target: str):
        return dd.from_dict({
            idx: {
                f"{target}": row[target],
                "zoned_dt_iso": row["zoned_dt_iso"],
                "lon": trp_data.get("lon"),
                "lat": trp_data.get("lat"),
            }
            for trp_data in trps_along_path_preds.values()
            for idx, row in enumerate(trp_data[f"{target}_preds"].itertuples(index=False), start=0)
            if row["zoned_dt_iso"] in time_range # This way we'll execute ordinary kriging only the strictly necessary number of times
        })


    def _get_ordinary_kriging(self, trps_along_path_preds: dict[str, dict[str, dd.DataFrame]], time_range: list[datetime.datetime], target: str, verbose: bool = False):
        ok_df = self._get_ok_structured_data(trps_along_path_preds=trps_along_path_preds, time_range=time_range, target=GlobalDefinitions.target)
        return OrdinaryKriging(
            x=ok_df["lon"].values,
            y=ok_df["lat"].values,
            z=ok_df[target].values,
            variogram_model="spherical",
            coordinates_type="geographical",
            verbose=verbose,
            enable_plotting=True
        )


    @staticmethod
    def _ok_interpolate(ordinary_kriging_obj: OrdinaryKriging, x_coords: list[float], y_coords: list[float], style: Literal["grid", "points"]) -> Any:
        if not len(x_coords) == len(y_coords):
            raise ValueError("There must be exactly the same number of pairs of coordinates")
        return ordinary_kriging_obj.execute(
            style=style,
            xpoints=x_coords,
            ypoints=y_coords
        )


    @staticmethod
    def _get_advanced_weighting(edge_start: str, edge_end: str, attrs: dict) -> PositiveFloat | None:
        length: PositiveFloat = attrs.get("length")
        road_category: str = attrs.get("road_category")
        min_lanes: PositiveInt = attrs.get("min_lanes")
        max_lanes: PositiveInt = attrs.get("max_lanes")
        highest_speed_limit: PositiveInt = attrs.get("highest_speed_limit")
        lowest_speed_limit: PositiveInt = attrs.get("lowest_speed_limit")

        travel_time_factor = (
            ((length / (((highest_speed_limit / 100) * 85) / 3.6)) * 60) *
            RoadCategoryTraitLengthWeightMultipliers[road_category].value
        ) * 0.35
        # Base travel time adjusted by road category and speed

        lane_factor = (
            # Contribution from minimum and maximum lanes
            min_lanes * 0.10 +
            max_lanes * 0.10
        )
        # Higher flow roads have a smaller multiplier that indicates particularly good conditions to travel as opposed to municipality-roads where multiple factors could influence the travel time, like: road maintenance, heavy tourist vehicles that block the road or that go slower, etc.
        speed_factor = (
            # Contribution of the highest speed limit
            ((highest_speed_limit / 100) * 85) * 0.20 +
            highest_speed_limit * 0.20
        )
        lowest_speed_contribution = lowest_speed_limit * 0.05
        # Contribution of the lowest speed limit

        return (
                travel_time_factor +
                lane_factor +
                speed_factor +
                lowest_speed_contribution
        )

    def find_route(self,
                   source: str,
                   destination: str,
                   time_range: list[datetime.datetime], # Can also be 1 datetime only. Must be zoned
                   use_volume: bool | None = True,
                   use_mean_speed: bool | None = None,
                   has_only_public_transport_lanes: bool | None = None,
                   has_toll_stations: bool | None = None,
                   has_ferry_routes: bool | None = None,
                   trp_research_buffer_radius: PositiveInt = 3500
    ) -> list[tuple[str, str]]:

        if not all([d.strftime(GlobalDefinitions.DT_ISO_TZ_FORMAT) for d in time_range]):
            raise ValueError(f"All time range values must be in {GlobalDefinitions.DT_ISO_TZ_FORMAT} format")

        if not any([use_volume, use_mean_speed]):
            raise ValueError(f"At least one of use_{GlobalDefinitions.VOLUME} or use_{GlobalDefinitions.MEAN_SPEED} must be set to True")

        targets = []
        if use_volume:
            targets.append(GlobalDefinitions.VOLUME)
        if use_mean_speed:
            targets.append(GlobalDefinitions.MEAN_SPEED)

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

        heuristic = lambda u, v: self._get_minkowski_dist(u, v, self._network)
        weight = "length"

        paths = {}


        for p in range(5):

            # ---------- STEP 1 ----------

            sp = self._get_shortest_path(source=source, destination=destination, heuristic=heuristic, weight=weight)
            print(sp)
            sp_edges = self._get_path_edges(sp)
            print(sp_edges)


            # ---------- STEP 2 ----------

            trps_per_edge = self._get_trps_per_edge(edges=sp_edges, trp_research_buffer_radius=trp_research_buffer_radius)
            self.trps_along_sp = set(*chain(trps_per_edge.values())) # A set of dictionary where each dict is a TRP with its metadata (id, road_category, etc.)


            # ---------- STEP 3 ----------

            self.trps_along_sp_preds.update(**{
                trp["trp_id"]: {
                    **{f"{target}_preds": self._get_trp_predictions(trp_id=trp["trp_id"], target=target, road_category=trp["road_category"], lags=[24, 36, 48, 60, 72])[[target, "zoned_dt_iso"]]
                        for target in targets},
                    **trp,
                } for trp in self.trps_along_sp.difference(set(self.trps_along_sp_preds.keys())) # All TRPs that are along the shortest path, but not in the ones for which we already computed the predictions
            })
            # Re-use them for future slightly different shortest path which have the same trps in common with the previous shortest paths
            #TODO IMPLEMENT MULTIPROCESSING AND ADD TO self.trps_along_sp_preds THE PREDICTIONS FOR ALL TRPs FOR WHICH THEY WERE COMPUTED ALL AT ONCE

            # ---------- STEP 4 ----------

            for target in targets:
                line_predictions = pd.DataFrame((
                        (x, y, edge_data["link_id"])
                         for u, v in sp_edges
                         for edge_data, geom in
                         [(self._network.get_edge_data(u, v), wkt.loads(self._network.get_edge_data(u, v)["geom"]))]
                         for x, y in geom.coords
                    ),
                    columns=["lon", "lat", "link_id"]
                ) # Creating the structure of the dataframe where we'll concatenate ordinary kriging results

                ok = self._get_ordinary_kriging(trps_along_path_preds=self.trps_along_sp_preds, time_range=time_range, target=target, verbose=True)

                z_interpolated_vals, kriging_variance= self._ok_interpolate(ordinary_kriging_obj=ok,
                                                                            x_coords=line_predictions["lon"].values,
                                                                            y_coords=line_predictions["lat"].values,
                                                                            style="points")  # Kriging variance is sometimes called "ss" (sigma squared)

                line_predictions[f"{target}_interpolated_value"] = z_interpolated_vals
                line_predictions[f"{target}_variance"] = kriging_variance


            # ---------- STEP 5 ----------

                link_agg_data = dict((link_id, self._get_link_aggregated_traffic_data(link_id=link_id)) for link_id in trps_per_edge.keys())  # Converting link_agg_data to dict for fast lookup
                # Simply using the dict() function on a list of tuples

                line_predictions[f"link_avg_{target}"] = line_predictions["link_id"].map(lambda link_id: link_agg_data[link_id][f"weighted_avg_{target}"])

                # Difference (point prediction - link-level avg)
                line_predictions[f"{target}_diff_from_avg"] = line_predictions[f"{target}_interpolated_value"] - line_predictions[f"link_avg_{target}"]

                # Merge STD computation into the dataframe
                line_predictions = line_predictions.merge((
                    line_predictions
                    .groupby("link_id")[f"{target}_diff_from_avg"]
                    .std()
                    .rename(f"{target}_diff_std")
                ), on="link_id", how="left")

                pct_diff = (line_predictions[f"{target}_interpolated_value"] - line_predictions[f"link_avg_{target}"]) / line_predictions[f"link_avg_{target}"] * 100
                line_predictions[f"{target}_traffic_class"] = np.select(
                    [
                        pct_diff < -25, # LOW
                        (pct_diff >= -25) & (pct_diff < -15), # LOW_AVERAGE
                        (pct_diff >= -15) & (pct_diff <= 15), # AVERAGE
                        (pct_diff > 15) & (pct_diff <= 25), # HIGH_AVERAGE
                        (pct_diff > 25) & (pct_diff <= 50), # HIGH
                        pct_diff > 50 # STOP_AND_GO
                    ],
                    [
                        TrafficClasses.LOW.name,
                        TrafficClasses.LOW_AVERAGE.name,
                        TrafficClasses.AVERAGE.name,
                        TrafficClasses.HIGH_AVERAGE.name,
                        TrafficClasses.HIGH.name,
                        TrafficClasses.STOP_AND_GO.name
                    ],
                    default=TrafficClasses.LOW.name
                ) # Each traffic class represents a percentage difference from the mean, example: if the forecasted value distance from the mean is within 15-25% more than the mean then average_high elif 25-50% more high, elif 50-100% stop and go

                high_traffic_perc = line_predictions[f"{target}_traffic_class"].isin([TrafficClasses.HIGH_AVERAGE.name, TrafficClasses.HIGH.name]).mean() * 100 # Calculating the percentage of the total path which has a traffic level above HIGH_AVERAGE
                # Getting only the fraction of rows where the mask value is True, so it is already a division on the total of rows
                # It's just a shortcut for mask = *row_value* isin(...) -> mask.sum() / len(mask)

                if high_traffic_perc > 50:
                    trp_research_buffer_radius += 2000

                    weight = ...

                    #TODO ADD THE PATH TO paths

                else:
                    paths.update({
                        str(p): {
                            "path": sp,
                            "high_traffic_perc": high_traffic_perc
                        }
                    })
                    break






        #TODO ADD AS A LAYER THE TRPS WHICH WERE USE FOR ORDINARY KRIGING ON A SPECIFIC PATH

        #TODO CREATE INTERPOLATION CHARTS FOR OSLO, BERGEN, TRONDHEIM, BODO, TROMSO, STAVANGER, ALTA, ETC. BY EXECUTING FORECASTS FOR EACH TRP IN THOSE CITIES (BY USING MUNICIPALITY ID)










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




