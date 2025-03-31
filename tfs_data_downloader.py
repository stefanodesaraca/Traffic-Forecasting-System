from gql import gql, Client
from gql.transport.aiohttp import AIOHTTPTransport
import json
import os
from warnings import simplefilter
import math
import numpy as np
import sys
from tqdm import tqdm, trange
from collections import ChainMap
import pprint
from concurrent.futures import ThreadPoolExecutor, as_completed

from tfs_utilities import *

simplefilter("ignore")

ops_folder = "ops"
cwd = os.getcwd()


# --------------------------------- GraphQL Client Start ---------------------------------


def start_client():
    transport = AIOHTTPTransport(url="https://trafikkdata-api.atlas.vegvesen.no/")
    client = Client(transport=transport, fetch_schema_from_transport=True)

    return client



# --------------------------------- GraphQL Queries Section (Data Fetching) ---------------------------------


#The number 3 indicates the Oslo og Viken county, which only includes the Oslo municipality
def fetch_traffic_registration_points(client: Client):

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
                roadCategory{
                  id
                }
              }
              roadLinkSequence{
                roadLinkSequenceId
                relativePosition
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

    traffic_registration_points = client.execute(tmp_query)

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


def traffic_registration_points_to_json(ops_name: str):
    """
    The _ops_name parameter is needed to identify the operation where the data needs to be downloaded.
    This implies that the same data can be downloaded multiple times, but downloaded into different operation folders,
    so reducing the risk of data loss or corruption in case of malfunctions.
    """

    client = start_client()

    TMPs = fetch_traffic_registration_points(client)

    with open(f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_data/traffic_registration_points.json", "w") as tmps_w:
        json.dump(TMPs, tmps_w, indent=4)

    return None


def traffic_volumes_data_to_json(ops_name: str, time_start: str, time_end: str):

    client = start_client()

    trps = import_TRPs_data()
    ids = []

    trafficRegistrationPoints = trps["trafficRegistrationPoints"]

    for trp in trafficRegistrationPoints:
        ids.append(trp["id"])

    #print(ids)


    def download_trp_data(trp_id):

        volumes = {}
        query_result = {}

        pages_counter = 0
        end_cursor = ""

        while end_cursor is not None:

            try:
                if pages_counter == 0:
                    query_result = fetch_traffic_volumes_for_trp_id(client=client, traffic_registration_point=trp_id, time_start=time_start, time_end=time_end, last_end_cursor=None, next_page_query=False)

                    end_cursor = query_result["trafficData"]["volume"]["byHour"]["pageInfo"]["endCursor"] if query_result["trafficData"]["volume"]["byHour"]["pageInfo"]["hasNextPage"] is True else None
                    pages_counter += 1
                    #print(end_cursor)

                    volumes = query_result


                elif pages_counter > 0:
                    query_result = fetch_traffic_volumes_for_trp_id(client=client, traffic_registration_point=trp_id, time_start=time_start, time_end=time_end, last_end_cursor=end_cursor, next_page_query=True)

                    end_cursor = query_result["trafficData"]["volume"]["byHour"]["pageInfo"]["endCursor"] if query_result["trafficData"]["volume"]["byHour"]["pageInfo"]["hasNextPage"] is True else None
                    pages_counter += 1
                    #print(end_cursor)


                    volumes["trafficData"]["volume"]["byHour"]["edges"].extend(query_result["trafficData"]["volume"]["byHour"]["edges"])


            except TimeoutError:
                continue
                #This error shouldn't be a problem, but it's important to manage it well.
                #The error gets raised in the fetch_traffic_volumes_for_trp_id() function, so the end cursor won't get updated. Thus, if a TimeoutError gets raised the program will just start downloading data from where it stopped before


        #pprint.pprint(query_result)

        return volumes


    elements = tqdm(ids)

    for i in elements:
        volumes_data = download_trp_data(i)
        write_trp_metadata(i)
        elements.set_postfix({f"writing data for TRP": i})

        # Exporting traffic volumes to a json file, S stands for "Start" and E stands for "End". They represent the time frame in which the data was collected (for a specific traffic measurement point)
        with open(f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_data/traffic_volumes/raw_traffic_volumes/{i}_volumes_S{time_start[:18].replace(':', '_')}_E{time_end[:18].replace(':', '_')}.json", "w") as tv_w:
            json.dump(volumes_data, tv_w, indent=4)

        # print("Data successfully written in memory\n")





    print("\n\n")

    return None



























