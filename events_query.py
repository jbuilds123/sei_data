import subprocess
import json


def events_search():
    """
    command reference:
    delegator reward
    seid query txs --events 'message.sender=cosmos1...&message.action=withdraw_delegator_reward'
    --page 1 --limit 30 --chain-id pacific-1 --node https://sei-rpc.brocha.in/ --output json

    contract creations
    seid query txs --events 'message.sender=cosmos1...&message.action=withdraw_delegator_reward'
    --page 1 --limit 30 --chain-id pacific-1 --node https://sei-rpc.brocha.in/ --output json
    """

    command = [
        "seid",
        "query",
        "txs",
        "--events",
        "wasm.action=create_pair",
        "--height",
        "49722499",
        "--page",
        "1",
        "--limit",
        "3",
        "--chain-id",
        "pacific-1",
        "--node",
        "https://sei-rpc.brocha.in/",
        "--output",
        "json",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.stdout:
        formatted_json = json.loads(result.stdout)
        return formatted_json
    else:
        return None


if __name__ == "__main__":
    output = events_search()
    if output:
        print("Events Result:")
        print(f"Total count: {output['total_count']}")
        print(f"count: {output['count']}")
        print(f"page total: {output['page_total']}")
        print(f"Transactions: {len(output['txs'])}")
        print(f"Hashes:")
        for tx in output["txs"]:
            print(tx["txhash"])
            print(tx["height"])
        print()
        # print(json.dumps(output, indent=2))
        print()
    else:
        print("Events Search is None...")
