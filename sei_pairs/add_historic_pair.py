import json
import asyncio
import aiohttp
import time
from asyncio import Semaphore

RATE_LIMIT = 30
SECONDS_IN_MINUTE = 60
SLEEP_TIME = SECONDS_IN_MINUTE / RATE_LIMIT
semaphore = Semaphore(RATE_LIMIT)

# Function to read the JSON file


def read_pairs_file(file_path):
    with open(file_path, "r") as file:
        return json.load(file)

# Function to write the updated JSON file


def write_pairs_file(file_path, data):
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)


async def fetch_ohlcv(pool_address, before_timestamp=None):
    ohlcv_list = []
    async with semaphore:
        await asyncio.sleep(SLEEP_TIME)
        try:
            async with aiohttp.ClientSession() as session:
                url = f"https://api.geckoterminal.com/api/v2/networks/sei-network/pools/{pool_address}/ohlcv/minute?aggregate=1&limit=1000"
                if before_timestamp:
                    url += f"&before_timestamp={before_timestamp}"
                async with session.get(url) as response:
                    data = await response.json()

                    pool_data = (
                        data.get("data", {}).get(
                            "attributes", {}).get("ohlcv_list", [])
                    )
                    if not pool_data:
                        return ohlcv_list

                    for d in pool_data:
                        ohlcv_dict = {
                            "timestamp": d[0],
                            "open": "{:0.18f}".format(d[1]),
                            "high": "{:0.18f}".format(d[2]),
                            "low": "{:0.18f}".format(d[3]),
                            "close": "{:0.18f}".format(d[4]),
                            "volume": "{:0.2f}".format(d[5]),
                        }
                        ohlcv_list.append(ohlcv_dict)
        except Exception as e:
            print(f"Error fetching data for {pool_address}: {e}")

        return ohlcv_list[::-1]


async def add_and_fetch_pair(pairs_file_path, pool_address, created_at):
    pairs_data = read_pairs_file(pairs_file_path)

    # Add new pair data
    pairs_data[pool_address] = {
        "created_at": created_at,
        "pool_address": pool_address,
        "pool_link": f"https://www.seiscan.app/pacific-1/contracts/{pool_address}",
        "links": []
    }

    # Fetch OHLCV data 15 hours ahead of created_at timestamp
    modified_timestamp = created_at + 15 * 3600
    ohlcv_data = await fetch_ohlcv(pool_address, modified_timestamp)

    if ohlcv_data:
        pairs_data[pool_address]["ohlcv"] = ohlcv_data
        print(f"OHLCV data fetched for {pool_address}")

    # Write the updated data to the file
    write_pairs_file(pairs_file_path, pairs_data)
    print("Pair added and updated successfully.")


async def main():
    pairs_file_path = "sei_pairs/historic_pairs.json"

    # Define the pool address and created_at timestamp for the new pair
    new_pool_address = "sei1lu574lgky4st6wy9uhnu5vf7fpsmyusum2rqutx3mzspq49tjtessln84v"
    new_created_at = 1703307600

    await add_and_fetch_pair(pairs_file_path, new_pool_address, new_created_at)

if __name__ == "__main__":
    asyncio.run(main())
