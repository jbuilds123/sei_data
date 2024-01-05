import json
import asyncio
import aiohttp
from asyncio import Semaphore

RATE_LIMIT = 30
SECONDS_IN_MINUTE = 60
SLEEP_TIME = SECONDS_IN_MINUTE / RATE_LIMIT


async def fetch_ohlcv(pool_address, semaphore: Semaphore):
    ohlcv_list = []
    async with semaphore:
        await asyncio.sleep(SLEEP_TIME)  # Sleep to respect rate limit
        async with aiohttp.ClientSession() as session:
            url = f"https://api.geckoterminal.com/api/v2/networks/sei-network/pools/{pool_address}/ohlcv/minute?aggregate=5&limit=1000"
            async with session.get(url) as response:
                data = await response.json()

                # Handling the KeyError by checking if keys exist
                pool_data = data.get("data", {}).get(
                    "attributes", {}).get("ohlcv_list", [])

                if not pool_data:
                    print(f"Skipping {pool_address} due to empty pool data.")
                    return ohlcv_list

                for d in pool_data:
                    ohlcv_dict = {
                        'timestamp': d[0],
                        'open': "{:0.18f}".format(d[1]),
                        'high': "{:0.18f}".format(d[2]),
                        'low': "{:0.18f}".format(d[3]),
                        'close': "{:0.18f}".format(d[4]),
                        'volume': "{:0.2f}".format(d[5]),
                    }
                    ohlcv_list.append(ohlcv_dict)

    return ohlcv_list
