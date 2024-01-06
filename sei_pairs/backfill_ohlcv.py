import json
import asyncio
import aiohttp
import time
from asyncio import Semaphore

RATE_LIMIT = 30
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


async def main():
    pairs_file_path = "sei_pairs/historic_pairs.json"
    pairs_data = read_pairs_file(pairs_file_path)

    total_pairs_processed = 0
    total_pairs_skipped = 0
    current_timestamp = time.time()
    six_hours_in_seconds = 6 * 3600
    sixteen_hours_in_seconds = 16 * 3600

    for pool_address, pair_info in pairs_data.items():
        if "ohlcv" in pair_info and pair_info["ohlcv"]:
            # Skip if OHLCV data already exists
            total_pairs_skipped += 1
            continue

        pair_age = current_timestamp - pair_info["created_at"]
        if pair_age < six_hours_in_seconds:
            total_pairs_skipped += 1
        elif pair_age < sixteen_hours_in_seconds:
            ohlcv_data = await fetch_ohlcv(pool_address)
        else:
            modified_timestamp = pair_info["created_at"] + \
                sixteen_hours_in_seconds
            ohlcv_data = await fetch_ohlcv(pool_address, modified_timestamp)

        if ohlcv_data:
            pairs_data[pool_address]["ohlcv"] = ohlcv_data
            total_pairs_processed += 1

        # Update progress every 15 pairs processed
        if (total_pairs_processed + total_pairs_skipped) % PROGRESS_UPDATE_FREQUENCY == 0:
            print(
                f"Progress: {total_pairs_processed + total_pairs_skipped} pairs processed (Processed: {total_pairs_processed}, Skipped: {total_pairs_skipped})")

    write_pairs_file(pairs_file_path, pairs_data)

    print(25 * "=")
    print(f"Total pairs processed for OHLCV data: {total_pairs_processed}")
    print(f"Total pairs skipped: {total_pairs_skipped}")
    print(25 * "=")
    print()


if __name__ == "__main__":
    asyncio.run(main())
