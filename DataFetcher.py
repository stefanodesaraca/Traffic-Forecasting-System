from gql import gql, Client
from gql.transport.aiohttp import AIOHTTPTransport


def start_client():
    transport = AIOHTTPTransport(url="https://trafikkdata-api.atlas.vegvesen.no/")
    client = Client(transport=transport, fetch_schema_from_transport=True)

    return client


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






