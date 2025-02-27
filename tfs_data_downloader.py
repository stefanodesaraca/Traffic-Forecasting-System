from gql import gql, Client
from gql.transport.aiohttp import AIOHTTPTransport
import json
import os
from warnings import simplefilter
import math
import numpy as np
from tqdm import tqdm

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
def fetch_traffic_measurement_points(client: Client):

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

    traffic_measurement_points = client.execute(tmp_query)

    return traffic_measurement_points


def fetch_traffic_volumes_for_tmp_id(client: Client, traffic_measurement_point: str, time_start: str, time_end: str):

    tv_query = gql(f"""{{
        trafficData(trafficRegistrationPointId: "{traffic_measurement_point}") {{
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


def traffic_measurement_points_to_json(ops_name: str):
    """
    The _ops_name parameter is needed to identify the operation where the data needs to be downloaded.
    This implies that the same data can be downloaded multiple times, but downloaded into different operation folders,
    so reducing the risk of data loss or corruption in case of malfunctions.
    """

    client = start_client()

    TMPs = fetch_traffic_measurement_points(client)

    with open(f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_data/traffic_measurement_points.json", "w") as tmps_w:
        json.dump(TMPs, tmps_w, indent=4)

    return None


def traffic_volumes_data_to_json(ops_name: str, time_start: str, time_end: str):

    client = start_client()

    #Read traffic measurement points json file
    with open(f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_data/traffic_measurement_points.json", "r") as tmps_r:
        tmps = json.load(tmps_r)


    ids = []

    trafficRegistrationPoints = tmps["trafficRegistrationPoints"]

    for trp in trafficRegistrationPoints:
        ids.append(trp["id"])

    #print(ids)


    def download_ids_chunk(chunk):
        try:
            id_volumes = {}
            for i in chunk:
                id_volumes.update({i: fetch_traffic_volumes_for_tmp_id(client=client, traffic_measurement_point=i, time_start=time_start, time_end=time_end)})
            return id_volumes
        except TimeoutError:
            print("\033[91mTimeout Error Raised. Safely Exited the Program\033[0m")
            exit()

        #TODO WATCH OUT FOR OUT OF MEMORY ERRORS, IF THE DATA IS TOO BIG FOR THE RAM THIS COULD CAUSE ERRORS


    requestChunkSize = int(math.sqrt(len(ids))) #The chunk size of each request cycle will be equal to the square root of the total number of ids
    requestChunks = np.array_split(ids, requestChunkSize)

    #Checking for duplicates in the ids list
    #print(len(ids), "|", len(set(ids)))

    #print("Requests chunks: ", requestChunks)

    tv = {} #Traffic Volumes

    for ids_chunk in tqdm(requestChunks, total=len(requestChunks)):
        tv.update(download_ids_chunk(ids_chunk)) #The download_ids_chunk returns a dictionary with a set of ids and respective traffic volumes data

    #print(tv)

    time_start = time_start[:18].replace(":", "_") #Keeping only the characters that were inputted by the user
    time_end = time_end[:18].replace(":", "_")

    print("Data collected successfully. Writing...")

    for tmp_id in ids:
        #Exporting traffic volumes to a json file, S stands for "Start" and E stands for "End". They represent the time frame in which the data was collected (for a specific traffic measurement point)
        with open(f"{cwd}/{ops_folder}/{ops_name}/{ops_name}_data/traffic_volumes/raw_traffic_volumes/{tmp_id}_volumes_S{time_start}_E{time_end}.json", "w") as tv_w:
            json.dump(tv[tmp_id], tv_w, indent=4)

    print("Data successfully written in memory\n")

    print("\n\n")

    return None



























