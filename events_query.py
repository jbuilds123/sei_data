import subprocess
import json


def events_search():
    '''
    command reference:
    delegator reward
    seid query txs --events 'message.sender=cosmos1...&message.action=withdraw_delegator_reward' 
    --page 1 --limit 30 --chain-id pacific-1 --node https://sei-rpc.brocha.in/ --output json

    contract creations
    seid query txs --events 'message.sender=cosmos1...&message.action=withdraw_delegator_reward' 
    --page 1 --limit 30 --chain-id pacific-1 --node https://sei-rpc.brocha.in/ --output json
    '''

    command = [
        "seid",
        "query",
        "txs",
        "--events",
        "wasm.register_collection=success",
        "--height", "49717431",
        "--page", "1",
        "--limit", "1",
        "--chain-id", "pacific-1",
        "--node", "https://sei-rpc.brocha.in/",
        "--output", "json",
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
        print()
        print(json.dumps(output, indent=2))
        print()
    else:
        print("Events Search is None...")
