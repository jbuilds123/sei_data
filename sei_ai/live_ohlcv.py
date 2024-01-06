import json
import asyncio
import aiohttp
from asyncio import Semaphore

RATE_LIMIT = 250
SECONDS_IN_MINUTE = 60
SLEEP_TIME = SECONDS_IN_MINUTE / RATE_LIMIT
semaphore = Semaphore(RATE_LIMIT)
PROGRESS_UPDATE_FREQUENCY = 15

# Function to read the JSON file


def read_pairs_file(file_path):
    with open(file_path, "r") as file:
        return json.load(file)

# Function to write the updated JSON file


def write_pairs_file(file_path, data):
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)


async def fetch_ohlcv(pool_address):
    ohlcv_list = []
    async with semaphore:
        await asyncio.sleep(SLEEP_TIME)
        try:
            async with aiohttp.ClientSession() as session:
                url = f"https://api.geckoterminal.com/api/v2/networks/sei-network/pools/{pool_address}/ohlcv/minute?aggregate=1&limit=1000"
                async with session.get(url) as response:
                    data = await response.json()
                    pool_data = data.get("data", {}).get(
                        "attributes", {}).get("ohlcv_list", [])
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


async def main():
    pairs_file_path = "sei_ai/live_pairs.json"
    pairs_data = read_pairs_file(pairs_file_path)

    total_pairs_processed = 0

    for pool_address in pairs_data.keys():
        ohlcv_data = await fetch_ohlcv(pool_address)
        if ohlcv_data:
            pairs_data[pool_address]["ohlcv"] = ohlcv_data
            total_pairs_processed += 1

        # Update progress
        if total_pairs_processed % PROGRESS_UPDATE_FREQUENCY == 0:
            print(f"Progress: {total_pairs_processed} pairs processed")

    write_pairs_file(pairs_file_path, pairs_data)
    print(25 * "=")
    print(f"Total pairs updated with OHLCV data: {total_pairs_processed}")
    print(25 * "=")
    print()

if __name__ == "__main__":
    asyncio.run(main())
