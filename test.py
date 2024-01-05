import json
import asyncio
import aiohttp
import time
from asyncio import Semaphore

RATE_LIMIT = 30
SECONDS_IN_MINUTE = 60
SLEEP_TIME = SECONDS_IN_MINUTE / RATE_LIMIT
semaphore = Semaphore(RATE_LIMIT)


async def fetch_ohlcv(pool_address):
    ohlcv_list = []
    async with semaphore:
        await asyncio.sleep(SLEEP_TIME)  # Sleep to respect rate limit
        try:
            async with aiohttp.ClientSession() as session:
                timestamp = 1704348000
                url = f"https://api.geckoterminal.com/api/v2/networks/sei-network/pools/{pool_address}/ohlcv/minute?aggregate=1&before_timestamp={timestamp}&limit=1000"
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

        return ohlcv_list[::-1]  # Reverse the list before returning


test_pair = "sei1amvpjejy0cvkcxdvuwwk80wwuge7e2truj0jrcedlrfqrcua7e6q27q83c"
data = asyncio.run(fetch_ohlcv(test_pair))
print(data)
