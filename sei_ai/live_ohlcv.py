import json
import asyncio
import aiohttp


# Function to read the JSON file
def read_pairs_file(file_path):
    with open(file_path, "r") as file:
        return json.load(file)


# Function to write the updated JSON file
def write_pairs_file(file_path, data):
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)


async def fetch_ohlcv(session, pool_address):
    ohlcv_list = []
    try:
        url = f"https://api.geckoterminal.com/api/v2/networks/sei-network/pools/{pool_address}/ohlcv/minute?aggregate=1&limit=1000"
        async with session.get(url) as response:
            data = await response.json()
            pool_data = data.get("data", {}).get("attributes", {}).get("ohlcv_list", [])
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
        print(f"|live_ohlcv| Error fetching data for {pool_address}: {e}")
    return pool_address, ohlcv_list[::-1]


# Function to analyze OHLCV data and update disqualified flag
def analyze_and_update_disqualification(pairs_data, pool_address, ohlcv_data):
    if len(ohlcv_data) >= 4:  # Check if there are more than 4 candles
        first_three_volumes = [float(candle["volume"]) for candle in ohlcv_data[:3]]
        avg_volume = sum(first_three_volumes) / len(first_three_volumes)
        if avg_volume < 25:
            pairs_data[pool_address]["disqualified"] = True
            # print(f"|live_ohlcv| Discqualified pair {pool_address}")


async def main():
    pairs_file_path = "sei_ai/live_pairs.json"
    pairs_data = read_pairs_file(pairs_file_path)

    found_ohlcv = 0
    skipped_pairs = 0
    newly_disqualified_pairs = 0

    async with aiohttp.ClientSession() as session:
        tasks = []
        for address in pairs_data.keys():
            if not pairs_data[address].get("disqualified", False):
                tasks.append(fetch_ohlcv(session, address))
            else:
                skipped_pairs += 1

        results = await asyncio.gather(*tasks)

        for pool_address, ohlcv_data in results:
            if ohlcv_data:
                found_ohlcv += 1
                was_disqualified = pairs_data[pool_address].get("disqualified", False)
                pairs_data[pool_address]["ohlcv"] = ohlcv_data
                analyze_and_update_disqualification(
                    pairs_data, pool_address, ohlcv_data
                )
                if not was_disqualified and pairs_data[pool_address].get(
                    "disqualified", False
                ):
                    newly_disqualified_pairs += 1

    write_pairs_file(pairs_file_path, pairs_data)
    print(
        f"|live_ohlcv| Updated with OHLCV: {found_ohlcv} | Skipped Disq {skipped_pairs} | New Disq {newly_disqualified_pairs}"
    )


if __name__ == "__main__":
    asyncio.run(main())
