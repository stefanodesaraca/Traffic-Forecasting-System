import os
from gql import gql, Client
from gql.transport.aiohttp import AIOHTTPTransport



def collect_traffic_measurement_points():

    transport = AIOHTTPTransport(url="https://trafikkdata-api.atlas.vegvesen.no/")
    client = Client(transport=transport, fetch_schema_from_transport=True)

    tmp_query = gql(
        '''
        {
          trafficRegistrationPoints(
            searchQuery: {roadCategoryIds: [E, R, F, K, P], isOperational: true, trafficType: VEHICLE, registrationFrequency: CONTINUOUS}
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
            direction {
              fromAccordingToRoadLink
              fromAccordingToMetering
              toAccordingToRoadLink
              toAccordingToMetering
            }
            commissions {
              lanes {
                laneNumberAccordingToRoadLink
                laneNumberAccordingToMetering
              }
            }
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


def collect_traffic_volumes_for_tmp_id(traffic_measurement_point: str, time_start: str, time_end: str):

    transport = AIOHTTPTransport(url="https://trafikkdata-api.atlas.vegvesen.no/")
    client = Client(transport=transport, fetch_schema_from_transport=True)

    tv_query = gql('''{
      trafficData(trafficRegistrationPointId: ''' + traffic_measurement_point + ''') {
        volume {
          byHour(from:''' + time_start + ',' + 'to:' + time_end + ''') {
            edges {
              node {
                from
                to
                total {
                  volumeNumbers {
                    volume
                  }
                  coverage {
                    percentage
                  }
                }
                byLane {
                  lane {
                    laneNumberAccordingToRoadLink
                    laneNumberAccordingToMetering
                  }
                  total {
                    coverage {
                      percentage
                    }
                    volumeNumbers {
                      volume
                    }
                  }
                }
                byDirection {
                  heading
                  total {
                    coverage {
                      percentage
                    }
                    volumeNumbers {
                      volume
                    }
                  }
                }
              }
            }
            pageInfo {
              hasNextPage
              endCursor
            }
          }
        }
      }
    }
    ''')

    traffic_volumes = client.execute(tv_query)

    return traffic_volumes





















