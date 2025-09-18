from pathlib import Path
import datetime
import pickle
from itertools import chain
from typing import Any, Literal, Mapping
from scipy.spatial.distance import minkowski
from shapely import wkt
from pydantic.types import PositiveFloat, PositiveInt
import networkx as nx
import numpy as np
import pandas as pd
import dask.dataframe as dd
from dask.distributed import Client
import utm
from pykrige.ok import OrdinaryKriging
import folium
from folium.raster_layers import ImageOverlay
import matplotlib.pyplot as plt
import warnings

from definitions import (
    GlobalDefinitions,
    ProjectTables,
    ProjectMaterializedViews,
    TrafficClasses,
    FoliumMapTiles,
    IconStyles,
    RoadCategoryTraitLengthWeightMultipliers,
    MapDefaultConfigs
)
from loaders import BatchStreamLoader
from pipelines import MLPreprocessingPipeline, MLPredictionPipeline
from utils import check_municipality_id, save_plot


warnings.filterwarnings(
    "ignore",
    module="distributed.shuffle"
)



class RoadNetwork:

    def __init__(self,
                 network_id: str,
                 name: str,
                 dask_client: Client,
                 backend: str = "networkx",
                 db_broker: Any | None = None,
                 loader: BatchStreamLoader | None = None,
                 network_binary: bytes | None = None
    ):
        self._dask_client = dask_client
        self._loader: BatchStreamLoader = loader
        self._db_broker: Any = db_broker #Synchronous DBBroker
        self.network_id: str = network_id
        self.name: str = name
        self._backend: str = backend
        self._network: nx.Graph | None = None #TODO CORRECTLY DEFINE ROAD DIRECTIONS, BUT WITH AN UNDIRECTED GRAPH EVERYTHING WORKS CORRECTLY. CHECK IF MULTI DIGRAPH WORKS

        if not network_binary:
            self._network = nx.Graph(**{"network_id": self.network_id, "name": self.name})
        else:
            self._network = pickle.loads(network_binary)  # To load pre-computed graphs

        self.trps_along_sp: set[dict[str, Any]] | None = None
        self.trps_along_sp_preds: dict[str, Any] | None = {}


    @staticmethod
    def _get_minkowski_dist(u: tuple[Any, ...], v: tuple[Any, ...], G: nx.Graph):
        # Parsing WKT geometry into shapely Point
        u_geom = G.nodes[u]["geom"]
        v_geom = G.nodes[v]["geom"]

        u_point = wkt.loads(u_geom) if isinstance(u_geom, str) else u_geom
        v_point = wkt.loads(v_geom) if isinstance(v_geom, str) else v_geom

        # Converting lat/lon to UTM
        u_easting, u_northing, _, _ = utm.from_latlon(u_point.y, u_point.x)
        v_easting, v_northing, _, _ = utm.from_latlon(v_point.y, v_point.x)

        return minkowski([u_easting, u_northing], [v_easting, v_northing], p=2.0) #TODO 1.0 if G.edges[u, v]["road_category"] in GlobalDefinitions.HIGH_SPEED_ROAD_CATEGORIES else


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
                trp.id AS id,
                trp.road_category AS road_category,
                trp.lat AS lat,
                trp.lon AS lon
            FROM neighbors n
            JOIN "{ProjectTables.RoadLink_TrafficRegistrationPoints.value}" rl_t
              ON n.neighbor_link_id = rl_t.link_id
            JOIN "{ProjectTables.TrafficRegistrationPoints.value}" trp
              ON rl_t.trp_id = trp.id;
        """, execute_args=[link_id, buffer_zone_radius])


    def _get_link_aggregated_traffic_data(
            self, link_id: str,
            county_avg_weight: PositiveFloat = 0.25,
            municipality_avg_weight: PositiveFloat = 0.5,
            road_category_avg_weight: PositiveFloat = 0.25
    ) -> dict[str, PositiveFloat | PositiveInt]:
        if round(county_avg_weight + municipality_avg_weight + road_category_avg_weight, 6) != 1.0:
            raise ValueError("Weights sum must be exactly 1")
        self._db_broker.send_sql(f"""
            REFRESH MATERIALIZED VIEW "{ProjectMaterializedViews.TrafficDataByCountyMView.value}";
            REFRESH MATERIALIZED VIEW "{ProjectMaterializedViews.TrafficDataByMunicipalityMView.value}";
            REFRESH MATERIALIZED VIEW "{ProjectMaterializedViews.TrafficDataByRoadCategoryMView.value}";
        """)
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


    def _get_trp_predictions(self, trp_id: str, target: str, road_category: str, lags: list[PositiveInt], model: str) -> dd.DataFrame:
        models = {m["name"]: pickle.loads(m["pickle_object"]) for m in self._db_broker.get_trained_model_objects(target=target, road_category=road_category)} # WARNING: If no models exist for a specific road category this could end up being an empty dictionary
        return MLPredictionPipeline(trp_id=trp_id, target=target, road_category=road_category, model=models[model], preprocessing_pipeline=MLPreprocessingPipeline(), loader=self._loader, db_broker=self._db_broker).start(training_mode=1, lags=lags) #TODO CUSTOMIZE TRAINING MODE


    def load_nodes(self) -> None:
        all(self._network.add_nodes_from((row.to_dict().pop("node_id"), row) for _, row in partition.iterrows()) for partition in self._loader.get_nodes(lat=True, lon=True).partitions)
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

        # NOTE WHEN WE'LL HAVE THE ABILITY TO FILTER DIRECTLY AT THE SOURCE OF THE NODES (WHEN WE'LL HAVE THE MUNICIPALITY AND COUNTY DATA ON THE NODES) WE'LL JUST NOT LOAD THE ONES OUTSIDE THE FILTERS CONDITIONS

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
            trp_research_buffer_radius += 500


    @staticmethod
    def _get_ok_structured_data(y_pred: dict[str, dict[str, dd.DataFrame] | Any], horizon: datetime.datetime, target: str):
        return dd.from_pandas(pd.DataFrame([
            {
                target: row[target],
                "lon": trp_data["lon"],
                "lat": trp_data["lat"],
            }
            for trp_data in y_pred.values()
            for _, row in trp_data[f"{target}_preds"].loc[trp_data[f"{target}_preds"]["zoned_dt_iso"] == horizon].iterrows()
        ]), npartitions=1)


    def _get_ordinary_kriging(self, y_pred: dict[str, dict[str, dd.DataFrame] | Any], horizon: datetime.datetime, target: str, verbose: bool = False):
        ok_df = self._get_ok_structured_data(y_pred=y_pred, horizon=horizon, target=target)
        return OrdinaryKriging(
            x=ok_df["lon"].values,
            y=ok_df["lat"].values,
            z=ok_df[target].values,
            variogram_model="spherical",
            coordinates_type="geographic",
            verbose=verbose,
            #enable_statistics=True,
            enable_plotting=True
        ), plt.gcf()


    @staticmethod
    def _ok_interpolate(ordinary_kriging_obj: OrdinaryKriging, x_coords: list[float | np.floating], y_coords: list[float | np.floating], style: Literal["grid", "points"]) -> Any:
        if not len(x_coords) == len(y_coords):
            raise ValueError("There must be exactly the same number of pairs of coordinates")
        return ordinary_kriging_obj.execute(
            style=style,
            xpoints=x_coords,
            ypoints=y_coords
        )


    @staticmethod
    def _edit_variogram_plot(fig: plt.Figure, target: str) -> plt.Figure:
        fig.suptitle(f"{target} Variogram")
        return fig


    @staticmethod
    def _get_edge_weighting(edge_start: str, edge_end: str, attrs: dict | Mapping[str, Any]) -> PositiveFloat | None:
        length: PositiveFloat = attrs.get("length")
        road_category: str = attrs.get("road_category")
        min_lanes: PositiveInt = attrs.get("min_lanes")
        max_lanes: PositiveInt = attrs.get("max_lanes")
        highest_speed_limit: PositiveInt = attrs.get("highest_speed_limit")
        lowest_speed_limit: PositiveInt = attrs.get("lowest_speed_limit")

        travel_time_factor = (
            ((length / (((highest_speed_limit / 100) * 85) / 3.6)) / 60) *
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

        return abs(
                travel_time_factor +
                lane_factor +
                speed_factor +
                lowest_speed_contribution
        ) #Can't return a potentially negative weight


    def _update_trps_preds(self, targets: list[str], lags: list[int], model: str) -> None:
        self.trps_along_sp_preds.update(**{
           trp["id"]: {
               **{f"{target}_preds": self._get_trp_predictions(trp_id=trp["id"], target=target, road_category=trp["road_category"], lags=lags, model=model)[[target, "zoned_dt_iso"]]
                  for target in targets},
               **trp,
           } for trp in list(trp for trp in self.trps_along_sp if trp["id"] not in self.trps_along_sp_preds.keys()) #Only calculating the predictions for the TRPs which hadn't seen their data predict yet, #TODO CHECK CONTENT: list(filter(lambda x: x not in self.trps_along_sp_preds.keys(), self.trps_along_sp))
           # All TRPs that are along the shortest path, but not in the ones for which we already computed the predictions
        }) # FYI: Using the dict() function on a tuple creates a dictionary
        return None


    def find_route(self,
                   source: str,
                   destination: str,
                   horizon: datetime.datetime, # Can also be 1 datetime only. Must be zoned
                   use_volume: bool | None = True,
                   use_mean_speed: bool | None = None,
                   has_only_public_transport_lanes: bool | None = None,
                   has_toll_stations: bool | None = None,
                   has_ferry_routes: bool | None = None,
                   trp_research_buffer_radius: PositiveInt = 3500,
                   model: str = "HistGradientBoostingRegressor",
                   max_iter: PositiveInt = 5
    ) -> dict[str, dict[str, Any]]:

        if not horizon.strftime(GlobalDefinitions.DT_ISO_TZ_FORMAT):
            raise ValueError(f"The horizon value must be in {GlobalDefinitions.DT_ISO_TZ_FORMAT} format")

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
        paths = {}
        removed_edges = {} #For each path we'll keep the edges removed when searching alternative paths
        lags = GlobalDefinitions.SHORT_TERM_LAGS

        try:
            for p in range(max_iter):

                # ---------- STEP 1 ----------

                sp = self._get_shortest_path(source=source, destination=destination, heuristic=heuristic, weight=self._get_edge_weighting)
                print(sp)
                sp_edges = self._get_path_edges(sp)
                #print(sp_edges)


                # ---------- STEP 2 ----------

                trps_per_edge = self._get_trps_per_edge(edges=sp_edges, trp_research_buffer_radius=trp_research_buffer_radius)
                unique_trps = {tuple(d.items()) for d in chain.from_iterable(trps_per_edge.values())}
                # Some edges could share the same TRP with others so we take each exactly once, so we'll compute the predictions for each one exactly once as well
                self.trps_along_sp = [dict(t) for t in unique_trps] # A set of dictionary where each dict is a TRP with its metadata (id, road_category, etc.)


                # ---------- STEP 3 ----------

                self._update_trps_preds(targets, lags=lags, model=model)
                # Re-using TRPs' predictions for future slightly different shortest path which have the same trps in common with the previous shortest paths


                # ---------- STEP 4 ----------

                total_sp_length = sum(
                    self._network.edges[u, v]["length"]
                    for u, v in zip(sp[:-1], sp[1:])
                )
                print("Route traits length: ", [self._network.edges[u, v]["length"] for u, v in zip(sp[:-1], sp[1:])])
                average_highest_speed_limit = np.mean([
                    self._network.edges[u, v]["highest_speed_limit"]
                    for u, v in zip(sp[:-1], sp[1:])
                ])
                print("Highest speed limit by trait: ", [self._network.edges[u, v]["highest_speed_limit"] for u, v in zip(sp[:-1], sp[1:])])

                paths.update({
                    str(p): {
                        "path": sp,
                        "path_edges": sp_edges, #Has edges latitude and longitudes too
                        "trps_per_edge": trps_per_edge, #Has TRPs latitude and longitudes too
                        "forecasted_travel_time": round(((total_sp_length / (
                                    ((average_highest_speed_limit / 100) * 90) / 3.6)) / 60) + ((len(sp) * 0.25) * 0.30), ndigits=2), # In minutes
                        # The formula indicates the length of the trait (in meters) and divided by 90% of the speed limit (in m/s (dividing it by 3.6)) all multiplied by 60 since we're interested in getting travel time minutes
                        # Considering that on average 25% of the road nodes (especially in urban areas) have traffic lights we'll count each one as 30s of wait time in the worst case scenario (all reds)
                        "trps_along_path": self.trps_along_sp,
                        "trp_research_buffer_radius": trp_research_buffer_radius
                    }
                })

                line_predictions = pd.DataFrame((
                        (x, y, (u, v), edge_data["link_id"])
                        for u, v, edge_data in sp_edges
                        for geom in [wkt.loads(edge_data["geom"])]
                        for x, y in geom.coords
                    ),
                    columns=["lon", "lat", "edge", "link_id"]
                )  # Creating the structure of the dataframe where we'll concatenate ordinary kriging results

                link_agg_data = dict((link_id, self._get_link_aggregated_traffic_data(link_id=link_id)) for link_id in trps_per_edge.keys())  # Converting link_agg_data to dict for fast lookup
                # Simply using the dict() function on a list of tuples


                # ---------- STEP 5 ----------

                for target in targets:

                    ok, variogram_plot = self._get_ordinary_kriging(y_pred=self.trps_along_sp_preds, horizon=horizon, target=target, verbose=True)
                    z_interpolated_vals, kriging_variance = self._ok_interpolate(
                                                                                ordinary_kriging_obj=ok,
                                                                                x_coords=line_predictions["lon"].values,
                                                                                y_coords=line_predictions["lat"].values,
                                                                                style="points"
                                                                            ) # Kriging variance is sometimes called "ss" (sigma squared)

                    variogram_plot = self._edit_variogram_plot(fig=variogram_plot, target=target)

                    line_predictions[f"{target}_interpolated_value"] = z_interpolated_vals
                    line_predictions[f"{target}_variance"] = kriging_variance

                    line_predictions[f"link_avg_{target}"] = line_predictions["link_id"].map(lambda link_id: link_agg_data[link_id][f"weighted_avg_{target}"])

                    # Difference (point prediction - link-level avg)
                    line_predictions[f"{target}_diff_from_avg"] = line_predictions[f"{target}_interpolated_value"] - line_predictions[f"link_avg_{target}"]

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

                    paths[str(p)].update(**{
                        f"{target}_ordinary_kriging_interpolated_values": z_interpolated_vals,
                        f"{target}_ordinary_kriging_variogram_plot": variogram_plot,
                        f"{target}_high_traffic_perc": high_traffic_perc,
                    })


                # ---------- STEP 6 ----------

                if any(paths[str(p)].get(f"{target}_high_traffic_perc", 0) > 50 for target in targets):
                    trp_research_buffer_radius += 2000 #Incrementing the TRP research buffer radius

                    sp_edges_weight = [(u, v, self._get_edge_weighting(u, v, data)) for u, v, data in sp_edges]

                    # Sort by weight (descending) and pick the top-n heaviest nodes to remove them
                    n = max(1, len(sp_edges))  # Maximum number of heaviest edges to remove
                    removed_edges[str(p)] = [
                        (u, v, self._network.edges[u, v].copy())
                        for u, v, w in sorted(sp_edges_weight, key=lambda x: x[2], reverse=True)[:n]
                    ] # Must save the whole edge with the attributes dictionary to add back into the graph afterward (with the attributes dictionary as well)
                    self._network.remove_edges_from([(u, v) for u, v, _ in removed_edges[str(p)]]) # Adopting this method since remove_edges_from accepts (u, v) tuples without attributes, but we need to keep attributes for the future re-insertion in the graph of ALL the nodes removed during the iterations

                else:
                    break

                traffic_priority = {
                    TrafficClasses.LOW.name: 0,
                    TrafficClasses.LOW_AVERAGE.name: 1,
                    TrafficClasses.AVERAGE.name: 2,
                    TrafficClasses.HIGH_AVERAGE.name: 3,
                    TrafficClasses.HIGH.name: 4,
                    TrafficClasses.STOP_AND_GO.name: 5
                }


                class_cols = [f"{target}_traffic_class" for target in targets]
                line_predictions["traffic_class"] = line_predictions[class_cols].apply(
                    lambda row: max(row, key=lambda x: traffic_priority[x]),
                    axis=1
                )

                print("Route predictions: ", line_predictions)
                paths[str(p)]["line_predictions"] = line_predictions


        except nx.exception.NetworkXNoPath:
            pass


        for re in removed_edges.values():
            self._network.add_edges_from(re)

        # Adding removed nodes back into the graph to avoid needing to re-build the whole graph again
        self._network.add_nodes_from(unreachable)

        # Adding removed edges back into the graph to avoid needing to re-build the whole graph again
        if has_only_public_transport_lanes:
            self._network.add_edges_from(has_only_public_transport_lanes_edges)
        if has_toll_stations:
            self._network.add_edges_from(has_toll_stations_edges)
        if has_ferry_routes:
            self._network.add_edges_from(has_ferry_routes_edges)

        return dict(
            sorted(
                paths.items(),
                key=lambda item: item[1]["forecasted_travel_time"],
                reverse=False  # Since we want ascending order having the paths with the lower high traffic percentages first
            )
        )


    @staticmethod
    def _create_map(lat_init: float | np.floating, lon_init: float | np.floating, zoom: PositiveInt = 8, tiles: str = FoliumMapTiles.OPEN_STREET_MAPS.value) -> folium.Map:
        return folium.Map(location=[lat_init, lon_init], tiles=tiles, zoom_start=zoom)


    @staticmethod
    def _add_marker(folium_obj: folium.Map | folium.FeatureGroup, marker_lat: float | np.floating, marker_lon: float | np.floating, tooltip: str | None = None, popup: str | None = None, icon: folium.Icon | None = None, circle: bool = False, fill: bool = False, radius: float | None = None) -> None:
        if isinstance(icon, dict):
            from folium.plugins import BeautifyIcon
            icon = BeautifyIcon(**icon)
        if not circle:
            folium.Marker(location=[marker_lat, marker_lon], tooltip=tooltip, popup=popup, icon=icon).add_to(folium_obj)
        else:
            folium.CircleMarker(location=[marker_lat, marker_lon], radius=radius, tooltip=tooltip, popup=popup, icon=icon, fill=fill).add_to(folium_obj)
        return None

    @staticmethod
    def _add_line(folium_obj: folium.Map | folium.FeatureGroup, locations: list[list[float | np.floating]], color: str, weight: float, smooth_factor: float | None = None, tooltip: str | None = None, popup: str | None = None) -> None:
        folium.PolyLine(locations=locations, color=color, weight=weight, smooth_factor=smooth_factor, tooltip=tooltip, popup=popup).add_to(folium_obj)
    # https://python-visualization.github.io/folium/latest/user_guide/vector_layers/polyline.html


    @staticmethod
    def export_map(map_obj: folium.Map, fp: str | Path) -> None:
        if not fp.endswith(".html"):
            raise ValueError("Wrong exporting format for interactive map. Must be HTML")
        map_obj.save(fp)  # Must end with .html
        return None


    def _get_route_start_end_nodes(self, route: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        return self._network.nodes[route["path"][0]],  self._network.nodes[route["path"][-1]]


    def _get_route_map_layers(self, route: dict[str, Any]) -> list[folium.FeatureGroup]:

        path_start_node, path_end_node = self._get_route_start_end_nodes(route=route)

        # Layers
        steps_layer = folium.FeatureGroup("steps")
        edges_layer = folium.FeatureGroup("edges")
        trps_layer = folium.FeatureGroup("trps")

        # Adding source node
        self._add_marker(folium_obj=steps_layer, marker_lat=path_start_node["lat"], marker_lon=path_start_node["lon"], popup="Start", icon=folium.Icon(icon=IconStyles.SOURCE_NODE_STYLE.value["icon"]))
        # Adding destination node
        self._add_marker(folium_obj=steps_layer, marker_lat=path_end_node["lat"], marker_lon=path_end_node["lon"], popup="Destination", icon=folium.Icon(icon=IconStyles.DESTINATION_NODE_STYLE.value["icon"]))

        for i in range(len(route["line_predictions"]) - 1):
            start = [route["line_predictions"].iloc[i]["lat"], route["line_predictions"].iloc[i]["lon"]]
            end = [route["line_predictions"].iloc[i + 1]["lat"], route["line_predictions"].iloc[i + 1]["lon"]]

            self._add_line(
                folium_obj=edges_layer,
                locations=[start, end],  # two points!
                color=TrafficClasses[route["line_predictions"].iloc[i]["traffic_class"]].value,
                weight=7
            )

        for trp in route["trps_along_path"]:
            self._add_marker(folium_obj=trps_layer, marker_lat=trp["lat"], marker_lon=trp["lon"], popup=trp["id"], icon=folium.Icon(icon=IconStyles.TRP_LINK_STYLE.value["icon"]))

        return [steps_layer, edges_layer, trps_layer]


    def _get_traffic_map_layers(self, trps: list[dict[str, Any]], municipality_geom: wkt, municipality_geom_color: str = "blue", municipality_geom_fill_color: str = "blue", municipality_fill_opacity: PositiveFloat = 0.3) -> list[folium.FeatureGroup]:
        municipality_geom_layer = folium.FeatureGroup("municipality_geom")
        trps_layer = folium.FeatureGroup("trps")

        folium.Polygon(
            locations=municipality_geom,
            color=municipality_geom_color, # The geometry border color
            weight=2, # The geometry border thickness
            fill=True,
            fill_color=municipality_geom_fill_color,
            fill_opacity=municipality_fill_opacity # Making the filling semi-transparent
        ).add_to(municipality_geom_layer)

        for trp in trps:
            self._add_marker(folium_obj=trps_layer, marker_lat=trp["lat"], marker_lon=trp["lon"], popup=trp["id"], icon=IconStyles.TRP_LINK_STYLE.value)

        return [municipality_geom_layer, trps_layer]


    @staticmethod
    def _get_layers_assembly(map_obj: folium.Map, layers: list[folium.FeatureGroup]) -> folium.Map:
        all(layer.add_to(map_obj) for layer in layers)
        return map_obj


    def draw_route(self,
                   route: dict[str, Any],
                   map_loc_init: list[float] | None = None,
                   tiles: str | FoliumMapTiles = FoliumMapTiles.OPEN_STREET_MAPS.value,
                   zoom_init: PositiveInt | None = None) -> folium.Map:

        path_start_node, path_end_node = self._get_route_start_end_nodes(route=route)

        if map_loc_init is None:
            map_loc_init = [path_start_node["lat"], path_start_node["lon"]]

        route_map = self._create_map(lat_init=map_loc_init[0], lon_init=map_loc_init[1], zoom=zoom_init or MapDefaultConfigs.ZOOM.value, tiles=tiles or FoliumMapTiles.OPEN_STREET_MAPS.value)

        for layer in self._get_route_map_layers(route=route):
            layer.add_to(route_map)

        return route_map


    def draw_routes(self,
                    routes: dict[str, dict[str, Any]],
                    map_loc_init: list[float] | None = None,
                    zoom_init: int = MapDefaultConfigs.ZOOM.value,
                    tiles: str = FoliumMapTiles.OPEN_STREET_MAPS. value) -> folium.Map:

        lat_init = None
        lon_init = None

        if map_loc_init is None:
            map_loc_init = [
                (sn["lat"], sn["lon"])
                for sn, _ in [tuple(self._get_route_start_end_nodes(route=r)) for r in routes.values()]
            ]
            lat_init = np.mean([lat for lat, lon in map_loc_init])
            lon_init = np.mean([lon for lat, lon in map_loc_init])

        return self._get_layers_assembly(
            map_obj=self._create_map(lat_init=lat_init, lon_init=lon_init, zoom=zoom_init or MapDefaultConfigs.ZOOM.value, tiles=tiles or FoliumMapTiles.OPEN_STREET_MAPS.value), # The map where to add all layers
            layers=list(chain.from_iterable(self._get_route_map_layers(route=r) for r in routes.values())) # Map layers
        )


    def _get_municipality_id_preds(self, municipality_id: PositiveInt, target: str, model: str) -> dict[str, dict[str, dd.DataFrame]]:

        print("MUNICIPALITY TRPS: ", self._db_broker.get_municipality_trps(municipality_id=municipality_id))

        print("lat: ", [trp["lat"] for trp in self._db_broker.get_municipality_trps(municipality_id=municipality_id)])
        print("lon: ", [trp["lon"] for trp in self._db_broker.get_municipality_trps(municipality_id=municipality_id)])

        return {
            trp["id"]:  {
                f"{target}_preds": self._get_trp_predictions(
                    trp_id=trp["id"],
                    road_category=trp["road_category"],
                    target=target,
                    lags=GlobalDefinitions.SHORT_TERM_LAGS,
                    model=model
                ),
                "lat": trp["lat"],
                "lon": trp["lon"],
            }
            for trp in self._db_broker.get_municipality_trps(municipality_id=municipality_id)
        }


    def _get_municipality_traffic_heatmap(self, municipality_id: PositiveInt, horizon: datetime.datetime, target: str, model: str, zoom_init: PositiveInt | None = None, tiles: str | None = None) -> folium.Map:
        check_municipality_id(municipality_id=municipality_id)

        municipality_geom = self._db_broker.get_municipality_geometry(municipality_id=municipality_id)
        min_lon, min_lat, max_lon, max_lat = municipality_geom.bounds

        gridx = np.linspace(min_lon - 0.35, max_lon + 0.35, 100).tolist()
        gridy = np.linspace(min_lat - 0.10, max_lat + 0.10, 100).tolist()

        ok, variogram_plot = self._get_ordinary_kriging(y_pred=self._get_municipality_id_preds(municipality_id=municipality_id, target=target, model=model), horizon=horizon, target=target)
        z_interpolated_vals, kriging_variance = self._ok_interpolate(
            ordinary_kriging_obj=ok,
            x_coords=gridx,  # Grid x bounds
            y_coords=gridy,  # Grid y bounds
            style="grid"
        )

        print("Kriging variance: ", kriging_variance)

        municipality_geom_center = municipality_geom.centroid

        municipality_traffic_map = self._create_map(lat_init=municipality_geom_center.y,
                                                    lon_init=municipality_geom_center.x,
                                                    zoom=zoom_init or MapDefaultConfigs.ZOOM.value,
                                                    tiles=tiles or FoliumMapTiles.OPEN_STREET_MAPS.value)
        municipality_traffic_map = self._get_layers_assembly(municipality_traffic_map, self._get_traffic_map_layers(self._db_broker.get_municipality_trps(municipality_id=municipality_id), municipality_geom=municipality_geom))

        fig, ax = plt.subplots(figsize=(8, 8))
        grid_interpolation_viz = ax.imshow(
            z_interpolated_vals,
            extent=(min(gridx),
                    max(gridx),
                    min(gridy),
                    max(gridy)),
            origin='lower',
            cmap=TrafficClasses.CMAP.value,  # NOTE BEFORE IT WAS 'magma',
            alpha=1
        )

        cbar = fig.colorbar(grid_interpolation_viz, ax=ax, orientation='vertical')  # Adding a color bar #NOTE TRY WITHOUT , fraction=0.036, pad=0.04
        cbar.set_label(target)

        ax.axis('off')  # Removing axes

        # Creating an image overlay object to add as a layer to the folium city map
        ImageOverlay(
            image=fig,
            bounds=[[min(gridy), min(gridx)], [max(gridy), max(gridx)]],
            # Defining the bounds where the image will be placed
            opacity=0.7,
            interactive=True,
            cross_origin=False,
            zindex=1,
        ).add_to(municipality_traffic_map)

        plt.close(fig)

        return municipality_traffic_map


    def draw_municipality_traffic_volume_heatmap(self, municipality_id: PositiveInt, horizon: datetime.datetime, model: str = "HistGradientBoostingRegressor", zoom_init: PositiveInt | None = None, tiles: str | None = None) -> folium.Map:
        return self._get_municipality_traffic_heatmap(municipality_id=municipality_id, horizon=horizon, target=GlobalDefinitions.VOLUME, model=model, tiles=tiles)


    def draw_municipality_traffic_mean_speed_heatmap(self, municipality_id: PositiveInt, horizon: datetime.datetime, model: str = "HistGradientBoostingRegressor", zoom_init: PositiveInt | None = None, tiles: str | None = None) -> folium.Map:
        return self._get_municipality_traffic_heatmap(municipality_id=municipality_id, horizon=horizon, target=GlobalDefinitions.MEAN_SPEED, model=model, tiles=tiles)


    def degree_centrality(self) -> dict:
        """
        Returns the degree centrality for each node

        Returns:
            A dictionary comprehending the degree centrality of every node of the network
        """
        return nx.degree_centrality(self._network)


    def betweenness_centrality(self) -> dict:
        """
        Returns the betweenness centrality for each node
        """
        return nx.betweenness_centrality(self._network, seed=100)


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




