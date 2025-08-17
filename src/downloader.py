import random
from typing import Any
from collections import defaultdict
import asyncio
from warnings import simplefilter
from gql import gql, Client
from gql.transport.exceptions import TransportServerError
from gql.transport.aiohttp import AIOHTTPTransport
from graphql import ExecutionResult
from pydantic.types import PositiveInt
from aiohttp.client_exceptions import ClientConnectorError, ClientOSError, ServerDisconnectedError

from pipelines import VolumeExtractionPipeline
from utils import GlobalDefinitions

simplefilter("ignore")


# This client is specifically thought for asynchronous data downloading
async def start_client_async() -> Client:
    return Client(transport=AIOHTTPTransport(url="https://trafikkdata-api.atlas.vegvesen.no/"), fetch_schema_from_transport=True)


async def fetch_areas(gql_client: Client) -> dict | ExecutionResult | None:
    try:
        return await gql_client.execute_async(gql("""
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


async def fetch_road_categories(gql_client: Client) -> dict | ExecutionResult | None:
    try:
        return await gql_client.execute_async(gql("""
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


async def fetch_trps_from_ids(gql_client: Client, trp_ids: list[str]) -> dict | ExecutionResult | None:
    try:
        return await gql_client.execute_async(gql(
            f"""
            {{
              trafficRegistrationPoints(
                trafficRegistrationPointIds: [{", ".join(f'"{x.strip()}"' for x in trp_ids)}]
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


async def fetch_trps(gql_client: Client, municipality_numbers: list[int] | None = None) -> dict | ExecutionResult | None:
    try:
        return await gql_client.execute_async(gql(
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


async def fetch_trp_volumes(gql_client: Client, trp_id: str, time_start: str, time_end: str, last_end_cursor: str, next_page_query: bool) -> dict | ExecutionResult:
    return await gql_client.execute_async(gql(f"""
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


async def volumes_to_db(db_broker_async: Any, time_start: str, time_end: str, n_async_jobs: PositiveInt = 5, max_retries: PositiveInt = 10, batch_size: int = 100000) -> None:
    semaphore = asyncio.Semaphore(n_async_jobs)  # Limit to n_async_jobs async tasks
    pipeline = VolumeExtractionPipeline(db_broker_async=db_broker_async)

    # Shared buffer for batches per TRP
    batch_buffers = defaultdict(dict) # Used to collect batches of data from each TRP to then ingest into the volumes processing pipeline
    batch_lock = asyncio.Lock()

    async def flush_batch(trp_id: str):
        async with batch_lock:
            if batch_buffers[trp_id]:
                await pipeline.ingest(
                    payload=batch_buffers[trp_id],
                    trp_id=trp_id,
                    fields=GlobalDefinitions.VOLUME_INGESTION_FIELDS
                )
                batch_buffers[trp_id].clear()

    async def download_trp_data(trp_id: str) -> None:
        pages_counter = 0
        retries = 0
        end_cursor = None

        while retries < max_retries:
            try:
                query_result = await fetch_trp_volumes(
                    await start_client_async(), #Starting a GraphQL client for each download_trp_data() call since a single client can't handle multiple asynchronous calls. This is because if multiple functions try to access the same client at the same time (while it can only handle one call at a time) this will raise this exception: gql.transport.exceptions.TransportAlreadyConnected: Transport is already connected
                    trp_id,
                    time_start,
                    time_end,
                    last_end_cursor=end_cursor,
                    next_page_query=pages_counter > 0
                )

                page_info = query_result["trafficData"]["volume"]["byHour"]["pageInfo"]
                end_cursor = page_info["endCursor"] if page_info["hasNextPage"] else None

                # Add the query result to the TRP's batch
                async with batch_lock:
                    # If a batch for the TRP exists then just append the collected data to the previously collected ones, otherwise append the whole query result with additional data returned from the API
                    if batch_buffers.get(trp_id, None) is not None:
                        batch_buffers[trp_id]["trafficData"]["volume"]["byHour"]["edges"].extend(query_result["trafficData"]["volume"]["byHour"]["edges"])
                    else:
                        batch_buffers[trp_id] = query_result
                    # Once the number of records in the buffer reaches the batch_size parameter's value
                    if len(batch_buffers[trp_id]) >= batch_size:
                        await flush_batch(trp_id)

                pages_counter += 1
                if end_cursor is None:
                    break

            except (TimeoutError, TransportServerError):
                if retries == max_retries:
                    print("\033[91mFailed to download TRP volumes data\033[0m")
                    break
                await asyncio.sleep(delay=(2 ^ retries) + random.random()) #Exponential backoff
                retries += 1

            except (ClientConnectorError, ClientOSError, ServerDisconnectedError):
                await asyncio.sleep(delay=(2 ^ retries ^ retries) + random.random())  #Big exponential backoff
                retries += 1

        return None

    async def limited_task(trp_id: str):
        async with semaphore:
            return await download_trp_data(trp_id)

    # Run all downloads in parallel with a maximum of 5 processes at the same time
    await asyncio.gather(*(limited_task(trp_id) for trp_id in (trp_record["id"] for trp_record in await db_broker_async.get_trp_ids_async())))

    # Final flush for any remaining items
    for trp_id in list(batch_buffers.keys()):
        await flush_batch(trp_id)

    return None


async def single_trp_volumes_to_db(db_broker_async: Any, trp_id: str, time_start: str, time_end: str, max_retries: PositiveInt = 10, batch_size: int = 100000) -> None:
    pipeline = VolumeExtractionPipeline(db_broker_async=db_broker_async)
    batch_buffer = {}

    async def flush_batch():
        if batch_buffer:
            await pipeline.ingest(
                payload=batch_buffer,
                trp_id=trp_id,
                fields=GlobalDefinitions.VOLUME_INGESTION_FIELDS
            )
            batch_buffer.clear()

    pages_counter = 0
    retries = 0
    end_cursor = None

    while retries < max_retries:
        try:
            gql_client = await start_client_async()
            query_result = await fetch_trp_volumes(
                gql_client,
                trp_id,
                time_start,
                time_end,
                last_end_cursor=end_cursor,
                next_page_query=pages_counter > 0
            )

            page_info = query_result["trafficData"]["volume"]["byHour"]["pageInfo"]
            end_cursor = page_info["endCursor"] if page_info["hasNextPage"] else None

            if batch_buffer:
                batch_buffer["trafficData"]["volume"]["byHour"]["edges"].extend(
                    query_result["trafficData"]["volume"]["byHour"]["edges"]
                )
            else:
                batch_buffer.update(query_result)

            if len(batch_buffer["trafficData"]["volume"]["byHour"]["edges"]) >= batch_size:
                await flush_batch()

            pages_counter += 1
            if end_cursor is None:
                break

        except (TimeoutError, TransportServerError):
            if retries == max_retries:
                print(f"\033[91mFailed to download TRP {trp_id} volumes data\033[0m")
                break
            await asyncio.sleep(delay=(2 ** retries) + random.random())
            retries += 1

        except (ClientConnectorError, ClientOSError, ServerDisconnectedError):
            await asyncio.sleep(delay=(2 ** retries ** retries) + random.random())
            retries += 1

    # Flush any remaining data
    await flush_batch()
