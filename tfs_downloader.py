from tfs_utils import *
import numpy as np
from gql import gql, Client
from gql.transport.exceptions import TransportServerError
from gql.transport.aiohttp import AIOHTTPTransport
import json
import os
import asyncio
import aiofiles
from warnings import simplefilter
import math
import sys
from tqdm import tqdm, trange
from collections import ChainMap
import pprint
import time


simplefilter("ignore")

ops_folder = "ops"
cwd = os.getcwd()


# --------------------------------- GraphQL Client Start ---------------------------------

#This client is only designed to run synchronously for fast-to-download stuff without needing to complicate simple functions like fetch_traffic_registration_points(), fetch_road_categories(), etc.
def start_client():
    transport = AIOHTTPTransport(url="https://trafikkdata-api.atlas.vegvesen.no/")
    client = Client(transport=transport, fetch_schema_from_transport=True)

    return client

#This client is specifically thought for asynchronous data downloading
async def start_client_async():
    transport = AIOHTTPTransport(url="https://trafikkdata-api.atlas.vegvesen.no/")
    client = Client(transport=transport, fetch_schema_from_transport=True)

    return client


# --------------------------------- GraphQL Queries Section (Data Fetching) ---------------------------------


#The number 3 indicates the Oslo og Viken county, which only includes the Oslo municipality
async def fetch_traffic_registration_points(client: Client):

    tmp_query = gql(
        '''
        {
          trafficRegistrationPoints(
            searchQuery: {roadCategoryIds: [E, R, F, K, P], countyNumbers: [3], isOperational: true, trafficType: VEHICLE, registrationFrequency: CONTINUOUS}
          ) {
            id
            name
            location {
              coordinates {
                latLon {
                  lat
                  lon
                }
              }
              roadReference{
                shortForm
                roadCategory{
                  id
                }
              }
              roadLinkSequence{
                roadLinkSequenceId
                relativePosition
              }
              roadReferenceHistory{
                validFrom
                validTo
              }
              county{
                name
                number
                geographicNumber
                countryPart{
                  id
                  name
                }
              }
              municipality{
                name
                number
              }
            }
            trafficRegistrationType            
            dataTimeSpan {
              firstData
              firstDataWithQualityMetrics
              latestData {
                volumeByDay
                volumeByHour
                volumeAverageDailyByYear
                volumeAverageDailyBySeason
                volumeAverageDailyByMonth
              }
            }
          }
        }
        ''')

    traffic_registration_points = await client.execute_async(tmp_query)

    return traffic_registration_points


def fetch_traffic_volumes_for_trp_id(client: Client, traffic_registration_point: str, time_start: str, time_end: str, last_end_cursor, next_page_query: bool):

    tv_query = {}

    if next_page_query is False:

        tv_query = gql(f"""{{
            trafficData(trafficRegistrationPointId: "{traffic_registration_point}") {{
            trafficRegistrationPoint{{
                  id
                  name
                }}
                volume {{
                    byHour(from: "{time_start}", to: "{time_end}") {{
                        edges {{
                            node {{
                                from
                                to
                                total {{
                                    volumeNumbers {{
                                        volume
                                    }}
                                    coverage {{
                                        percentage
                                    }}
                                }}
                                byLane {{
                                    lane {{
                                        laneNumberAccordingToRoadLink
                                        laneNumberAccordingToMetering
                                    }}
                                    total {{
                                        coverage {{
                                            percentage
                                        }}
                                        volumeNumbers {{
                                            volume
                                        }}
                                    }}
                                }}
                                byDirection {{
                                    heading
                                    total {{
                                        coverage {{
                                            percentage
                                        }}
                                        volumeNumbers {{
                                            volume
                                        }}
                                    }}
                                }}
                            }}
                        }}
                        pageInfo {{
                            hasNextPage
                            endCursor
                        }}
                    }}
                }}
            }}
        }}""")

    elif next_page_query is True:

        tv_query = gql(f"""{{
                    trafficData(trafficRegistrationPointId: "{traffic_registration_point}") {{
                    trafficRegistrationPoint{{
                          id
                          name
                        }}
                        volume {{
                            byHour(from: "{time_start}", to: "{time_end}", after: "{last_end_cursor}") {{
                                edges {{
                                    node {{
                                        from
                                        to
                                        total {{
                                            volumeNumbers {{
                                                volume
                                            }}
                                            coverage {{
                                                percentage
                                            }}
                                        }}
                                        byLane {{
                                            lane {{
                                                laneNumberAccordingToRoadLink
                                                laneNumberAccordingToMetering
                                            }}
                                            total {{
                                                coverage {{
                                                    percentage
                                                }}
                                                volumeNumbers {{
                                                    volume
                                                }}
                                            }}
                                        }}
                                        byDirection {{
                                            heading
                                            total {{
                                                coverage {{
                                                    percentage
                                                }}
                                                volumeNumbers {{
                                                    volume
                                                }}
                                            }}
                                        }}
                                    }}
                                }}
                                pageInfo {{
                                    hasNextPage
                                    endCursor
                                }}
                            }}
                        }}
                    }}
                }}""")


    traffic_volumes = client.execute(tv_query)
    #print(traffic_volumes)

    return traffic_volumes


def fetch_road_categories(client: Client):

    rc_query = gql('''
    {
        roadCategories{
            id
            name
        }
    }
    ''')

    road_categories = client.execute(rc_query)

    return road_categories


def fetch_areas(client: Client):

    a_query = gql('''
    {
      areas {
        countryParts {
          name
          id
          counties {
            name
            number
            geographicNumber
            municipalities {
              name
              number
            }
            countryPart {
              name
              id
            }
          }
        }
      }
    }
    ''')

    areas = client.execute(a_query)

    return areas



# --------------------------------- JSON Writing Section ---------------------------------


async def traffic_registration_points_to_json(ops_name: str):
    """
    The _ops_name parameter is needed to identify the operation where the data needs to be downloaded.
    This implies that the same data can be downloaded multiple times, but downloaded into different operation folders,
    so reducing the risk of data loss or corruption in case of malfunctions.
    """

    client = await start_client_async()
    TRPs = await fetch_traffic_registration_points(client)

    async with aiofiles.open(f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_data/traffic_registration_points.json", "w") as trps_w:
        await trps_w.write(json.dumps(TRPs, indent=4))

    return None


async def traffic_volumes_data_to_json(time_start: str, time_end: str) -> None:
    semaphore = asyncio.Semaphore(5)  # Limit to 5 concurrent tasks
    trps = import_TRPs_data()
    ids = [trp["id"] for trp in trps["trafficRegistrationPoints"]]

    async def download_trp_data(trp_id):
        client = await start_client_async()
        volumes = {}
        pages_counter = 0
        end_cursor = None

        while True:
            try:
                query_result = await asyncio.to_thread(
                    fetch_traffic_volumes_for_trp_id,
                    client, trp_id, time_start, time_end,
                    last_end_cursor=end_cursor,
                    next_page_query=pages_counter > 0
                )

                page_info = query_result["trafficData"]["volume"]["byHour"]["pageInfo"]
                end_cursor = page_info["endCursor"] if page_info["hasNextPage"] else None
                pages_counter += 1

                if pages_counter == 1:
                    volumes = query_result
                else:
                    volumes["trafficData"]["volume"]["byHour"]["edges"].extend(query_result["trafficData"]["volume"]["byHour"]["edges"])

                if end_cursor is None:
                    break

            except TimeoutError:
                continue
            except TransportServerError: #If error code is 503: Service Unavailable
                continue

        return volumes


    async def process_trp(trp_id):
        volumes_data = await download_trp_data(trp_id)

        folder_path = await asyncio.to_thread(read_metainfo_key, ["folder_paths", "data", "traffic_volumes", "subfolders", "raw", "path"])
        filename = f"{trp_id}_volumes_S{time_start[:18].replace(':', '_')}_E{time_end[:18].replace(':', '_')}.json"
        full_path = folder_path + filename

        async with aiofiles.open(full_path, "w") as f:
            await f.write(json.dumps(volumes_data, indent=4))

        await asyncio.to_thread(update_metainfo_async, filename, ["traffic_volumes", "raw_filenames"], mode="append")
        await asyncio.to_thread(update_metainfo_async, full_path, ["traffic_volumes", "raw_filepaths"], mode="append")

    async def limited_task(trp_id):
            async with semaphore:
                return await process_trp(trp_id)

    #Run all downloads in parallel with a maximum of 5 processes at the same time
    await asyncio.gather(*(limited_task(trp_id) for trp_id in ids))

    print("\n\n")

    return None



























