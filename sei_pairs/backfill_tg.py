from telethon import TelegramClient
from dotenv import load_dotenv
import os
import asyncio
import re
import json
from datetime import datetime, timedelta, timezone

load_dotenv()

api_id = os.getenv("TELE_API_ID")
api_hash = os.getenv("TELE_API_HASH")
target_channel = "https://t.me/sei_deploys"

# Initialize the Telegram Client
client = TelegramClient("anon", api_id, api_hash)


def load_existing_data(file_path):
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return {}


def save_data_to_file(file_path, data):
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)


async def fetch_and_process_messages(channel):
    # Define regex patterns for the required data
    pool_pattern = r"\[?\*\*Pool\]?\((https://www\.seiscan\.app/pacific-1/contracts/(\w+))\)\*\*\n`(\w+)`"
    token_pattern = r"\[?\*\*([^*]+)\]?\((https://www\.seiscan\.app/pacific-1/contracts/(\w+))\)\*\* - ([\d,]+|⚠️ Unlimited supply)( \(\+[\d]+ decimals\))?"
    deployer_pattern = r"\[?\*\*Deployer\]?\((https://www\.seiscan\.app/pacific-1/accounts/(\w+))\)\*\*"
    deployer_count_pattern = r"\[?\*\*Deployer\]\((https://www\.seiscan\.app/pacific-1/accounts/\w+)\)\*\* \(__(\d+)__\)"
    transaction_pattern = r"\[?\*\*Transaction\]?\((https://www\.seiscan\.app/pacific-1/txs/([0-9A-F]+))\)\*\*"
    link_pattern = r"\[([^\]]+)\]\((https?://[^\)]+)\)"

    async with client:
        messages = await client.get_messages(channel, limit=250)
        processed_messages = []

        for message in messages:
            if message.text:
                data = {}
                pool_match = re.search(pool_pattern, message.text)
                token_match = re.search(token_pattern, message.text)
                deployer_match = re.search(deployer_pattern, message.text)
                transaction_match = re.search(
                    transaction_pattern, message.text)
                links_matches = re.findall(link_pattern, message.text)

                # Add created_at key with the timestamp
                data["created_at"] = int(
                    message.date.replace(tzinfo=timezone.utc).timestamp()
                )

                if pool_match:
                    data["pool_address"] = pool_match.group(3)
                    data["pool_link"] = pool_match.group(1)

                if token_match:
                    data["token_symbol"] = token_match.group(1)
                    data["token_address"] = token_match.group(3)
                    data["token_link"] = token_match.group(2)
                    supply = token_match.group(4)
                    if "Unlimited supply" in supply:
                        data["unlimited_supply"] = True
                        data["total_supply"] = "Unlimited"
                    else:
                        data["unlimited_supply"] = False
                        data["total_supply"] = supply.replace(",", "")

                if deployer_match:
                    data["deployer_address"] = deployer_match.group(2)
                    data["deployer_link"] = deployer_match.group(1)

                deployer_count_match = re.search(
                    deployer_count_pattern, message.text
                )
                if deployer_count_match:
                    deployer_count = int(deployer_count_match.group(2))
                    data["deployer_count"] = deployer_count
                    data["multiple_deploys"] = deployer_count > 0

                if transaction_match:
                    data["transaction_hash"] = transaction_match.group(2)
                    data["transaction_link"] = transaction_match.group(1)

                if links_matches:
                    data["links"] = [
                        {"title": link[0], "url": link[1]}
                        for link in links_matches
                        if "twitter.com" in link[1] or "t.me" in link[1]
                    ]

                processed_messages.append(data)

        return processed_messages


async def main():
    # Define your JSON file path here
    json_file_path = "sei_pairs/historic_pairs.json"
    existing_data = load_existing_data(json_file_path)

    # Initialize counters
    total_pairs_processed = 0
    new_pairs_added = 0
    pairs_skipped = 0

    try:
        latest_messages = await fetch_and_process_messages(target_channel)
        # print("Processed messages:")
        for msg in latest_messages:
            total_pairs_processed += 1
            pool_address = msg.get("pool_address")
            if pool_address:
                if pool_address not in existing_data:
                    # Add new entry if it does not exist
                    existing_data[pool_address] = msg
                    new_pairs_added += 1
                else:
                    pairs_skipped += 1

        save_data_to_file(json_file_path, existing_data)  # Save merged data

        # Print summary
        print(25 * "=")
        print(f"Total pairs from tg fetch: {total_pairs_processed}")
        print(f"New pairs added: {new_pairs_added}")
        print(f"Pairs skipped: {pairs_skipped}")
        print(f"Total pairs in file: {len(existing_data)}")
        print(25 * "=")
        print()

        """
        for msg in latest_messages:
            print(20 * "-")
            print(json.dumps(msg, indent=4))
            print(20 * "-")
        """
    except Exception as e:
        print(f"An error occurred in main: {e}")


if __name__ == "__main__":
    asyncio.run(main())
