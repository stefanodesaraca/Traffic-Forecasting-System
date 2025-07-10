import asyncio
import aiofiles
from warnings import simplefilter
from gql import gql, Client
from gql.transport.exceptions import TransportServerError
from gql.transport.aiohttp import AIOHTTPTransport
from graphql import ExecutionResult
from pydantic import BaseModel

from tfs_utils import GlobalDefinitions
from tfs_base_config import pjh, pmm, tmm

simplefilter("ignore")


class RetryErrors(BaseModel):
    NotFound: int = 404
    RequestTimeout: int = 408
    MisdirectedRequest: int = 421
    UnprocessableContent: int = 422
    TooEarly: int = 425
    TooManyRequests: int = 429
    GatewayTimeout: int = 504


# This client is specifically thought for asynchronous data downloading
async def start_client_async() -> Client:
    return Client(transport=AIOHTTPTransport(url="https://trafikkdata-api.atlas.vegvesen.no/"), fetch_schema_from_transport=True)


async def fetch_areas(client: Client) -> dict | ExecutionResult | None:
    try:
        return await client.execute_async(gql("""
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
                """))
    except TimeoutError:
        return None
    except TransportServerError:  # If error code is 503: Service Unavailable
        return None


async def fetch_road_categories(client: Client) -> dict | ExecutionResult | None:
    try:
        return await client.execute_async(gql("""
        {
            roadCategories{
                id
                name
            }
        }
        """))
    except TimeoutError:
        return None
    except TransportServerError:  # If error code is 503: Service Unavailable
        return None


async def fetch_trps(client: Client, municipality_numbers: list[int] | None = None) -> dict | ExecutionResult | None:
    try:
        return await client.execute_async(gql(
            f"""
            {{
              trafficRegistrationPoints(
                searchQuery: {{roadCategoryIds: [E, R, F, K, P], countyNumbers: [{', '.join([str(n) for n in (municipality_numbers or [3])])}], isOperational: true, trafficType: VEHICLE, registrationFrequency: CONTINUOUS}}
              ) {{
                id
                name
                location {{
                  coordinates {{
                    latLon {{
                      lat
                      lon
                    }}
                  }}
                  roadReference{{
                    shortForm
                    roadCategory{{
                      id
                    }}
                  }}
                  roadLinkSequence{{
                    roadLinkSequenceId
                    relativePosition
                  }}
                  roadReferenceHistory{{
                    validFrom
                    validTo
                  }}
                  county{{
                    name
                    number
                    geographicNumber
                    countryPart{{
                      id
                      name
                    }}
                  }}
                  municipality{{
                    name
                    number
                  }}
                }}
                trafficRegistrationType            
                dataTimeSpan {{
                  firstData
                  firstDataWithQualityMetrics
                  latestData {{
                    volumeByDay
                    volumeByHour
                    volumeAverageDailyByYear
                    volumeAverageDailyBySeason
                    volumeAverageDailyByMonth
                  }}
                }}
              }}
            }}
            """
        )) # The number 3 indicates the Oslo og Viken county, which only includes the Oslo municipality
    except TimeoutError:
        return None
    except TransportServerError:  # If error code is 503: Service Unavailable
        return None


async def fetch_volumes_for_trp_id(client: Client, trp_id: str, time_start: str, time_end: str, last_end_cursor: str, next_page_query: bool) -> dict | ExecutionResult:
    return await client.execute_async(gql(f"""
        {{
            trafficData(trafficRegistrationPointId: "{trp_id}") {{
                trafficRegistrationPoint {{
                    id
                    name
                }}
                volume {{
                    byHour(from: "{time_start}", to: "{time_end}"{f', after: "{last_end_cursor}"' if next_page_query else ''}) {{
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
        }}
        """))


async def volumes_to_json(time_start: str, time_end: str) -> None:
    semaphore = asyncio.Semaphore(5)  # Limit to 5 concurrent tasks

    async def download_trp_data(trp_id):
        client = await start_client_async()
        volumes = {}
        pages_counter = 0
        end_cursor = None

        while True:
            try:
                query_result = await fetch_volumes_for_trp_id(
                    client,
                    trp_id,
                    time_start,
                    time_end,
                    last_end_cursor=end_cursor,
                    next_page_query=pages_counter > 0,
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
                continue #TODO IMPLEMENT EXPONENTIAL BACKOFF FOR A RANGE OF ERRORS (408, 429 AND SO ON...)
            except TransportServerError:  # If error code is 503: Service Unavailable
                continue

        return volumes

    async def process_trp(trp_id: str):

        await download_trp_data(trp_id) #TODO SEND IT TO PIPELINE

        await asyncio.to_thread(tmm.set_trp_metadata, trp_id=trp_id, **{"raw_volumes_file": trp_id + GlobalDefinitions.RAW_VOLUME_FILENAME_ENDING.value + ".json"})  # Writing TRP's empty metadata file #TODO MAKE THIS FUNCTION ASYNC

    async def limited_task(trp_id: str):
        async with semaphore:
            return await process_trp(trp_id)

    # Run all downloads in parallel with a maximum of 5 processes at the same time
    await asyncio.gather(*(limited_task(trp_id) for trp_id in pjh.trps_data.keys()))

    print("\n\n")

    return None
