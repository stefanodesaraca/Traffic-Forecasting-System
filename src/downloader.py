import random
import asyncio
from warnings import simplefilter
from gql import gql, Client
from gql.transport.exceptions import TransportServerError
from gql.transport.aiohttp import AIOHTTPTransport
from graphql import ExecutionResult
from pydantic.types import PositiveInt

from brokers import AIODBBroker
from pipelines import VolumeExtractionPipeline
from utils import GlobalDefinitions

simplefilter("ignore")


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


async def fetch_trp_volumes(client: Client, trp_id: str, time_start: str, time_end: str, last_end_cursor: str, next_page_query: bool) -> dict | ExecutionResult:
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


async def volumes_to_db(gql_client: Client, db_broker_async: AIODBBroker, time_start: str, time_end: str, n_async_jobs: PositiveInt = 5, max_retries: PositiveInt = 10) -> None:
    semaphore = asyncio.Semaphore(n_async_jobs)  # Limit to n_async_jobs async tasks
    pipeline = VolumeExtractionPipeline(db_broker_async=db_broker_async)

    async def download_trp_data(trp_id: str) -> None:
        pages_counter = 0
        retries = 0
        end_cursor = None

        while retries < max_retries:
            try:
                query_result = await fetch_trp_volumes(gql_client,
                                                       trp_id,
                                                       time_start,
                                                       time_end,
                                                       last_end_cursor=end_cursor,
                                                       next_page_query=pages_counter > 0)

                page_info = query_result["trafficData"]["volume"]["byHour"]["pageInfo"]
                end_cursor = page_info["endCursor"] if page_info["hasNextPage"] else None

                await pipeline.ingest(payload=query_result, fields=GlobalDefinitions.VOLUME_INGESTION_FIELDS.value)

                pages_counter += 1
                if end_cursor is None:
                    break

            except (TimeoutError, TransportServerError):
                if retries == max_retries:
                    print("\033[91mFailed to download TRP volumes data\033[0m")
                    break
                await asyncio.sleep(delay=(2 ^ retries) + random.random()) #Exponential backoff
                retries += 1

        return None

    async def limited_task(trp_id: str):
        async with semaphore:
            return await download_trp_data(trp_id)

    # Run all downloads in parallel with a maximum of 5 processes at the same time
    await asyncio.gather(*(limited_task(trp_id) for trp_id in [trp_record["id"] for trp_record in await db_broker_async.get_trp_ids_async()]))


    return None











