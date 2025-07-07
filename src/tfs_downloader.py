import pprint
import json
import os
import asyncio
import aiofiles
from warnings import simplefilter
from gql import gql, Client
from gql.transport.exceptions import TransportServerError
from gql.transport.aiohttp import AIOHTTPTransport
from graphql import ExecutionResult

from tfs_utils import GlobalDefinitions
from tfs_base_config import pjh, pmm, tmm, trp_toolbox

simplefilter("ignore")


# --------------------------------- GraphQL Client Start ---------------------------------

# This client is specifically thought for asynchronous data downloading
async def start_client_async() -> Client:
    return Client(transport=AIOHTTPTransport(url="https://trafikkdata-api.atlas.vegvesen.no/"), fetch_schema_from_transport=True)


#TODO IN THE FUTURE ASYNCHRONIZE EVERY FUNCTION (CREATE A SPECIFIC FEATURE BRANCH FOR THAT MODIFICATION)

# --------------------------------- GraphQL Queries Section (Data Fetching) ---------------------------------


# The number 3 indicates the Oslo og Viken county, which only includes the Oslo municipality
async def fetch_trps(client: Client, municipality_numbers: list[int] | None = None) -> dict | ExecutionResult:
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
    ))


def fetch_volumes_for_trp_id(client: Client, trp_id: str, time_start: str, time_end: str, last_end_cursor: str, next_page_query: bool) -> dict | ExecutionResult:
    return client.execute(gql(f"""
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


def fetch_road_categories(client: Client) -> dict | ExecutionResult:
    return client.execute(gql("""
    {
        roadCategories{
            id
            name
        }
    }
    """))

#TODO TO IMPLEMENT
def fetch_areas(client: Client) -> dict | ExecutionResult:
    return client.execute(gql("""
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


# --------------------------------- JSON Writing Section ---------------------------------


async def trps_to_json() -> None:
    client = await start_client_async()
    TRPs = await fetch_trps(client)

    async with aiofiles.open(pjh.trps_fp, "w") as trps_w:
        await trps_w.write(json.dumps({data["id"]: data for data in TRPs["trafficRegistrationPoints"]}, indent=4))
    return None


async def volumes_to_json(time_start: str, time_end: str) -> None:
    semaphore = asyncio.Semaphore(5)  # Limit to 5 concurrent tasks

    async def download_trp_data(trp_id):
        client = await start_client_async()
        volumes = {}
        pages_counter = 0
        end_cursor = None

        while True:
            try:
                query_result = await asyncio.to_thread(
                    fetch_volumes_for_trp_id,
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
                continue
            except TransportServerError:  # If error code is 503: Service Unavailable
                continue

        return volumes

    async def process_trp(trp_id):

        # TODO SET PATH USING tmm.set_path(trp_id)

        async with aiofiles.open(await asyncio.to_thread(pmm.get(key="folder_paths.data." + GlobalDefinitions.VOLUME.value + ".subfolders.raw.path") + trp_id + GlobalDefinitions.RAW_VOLUME_FILENAME_ENDING.value + ".json", "w")) as f:
            await f.write(json.dumps(await download_trp_data(trp_id), indent=4))

        tmm.set_trp_metadata(trp_id=trp_id, **{
            "raw_volumes_file": trp_id + GlobalDefinitions.RAW_VOLUME_FILENAME_ENDING.value + ".json"})  # Writing TRP's empty metadata file
        await tmm.set_async(value=(await trp_toolbox.get_global_trp_data_async())[trp_id], key="trp_data", mode="e")

    async def limited_task(trp_id):
        async with semaphore:
            return await process_trp(trp_id)

    # Run all downloads in parallel with a maximum of 5 processes at the same time
    await asyncio.gather(*(limited_task(trp_id) for trp_id in pjh.trps_data.keys()))

    print("\n\n")

    return None
