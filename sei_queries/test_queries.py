import subprocess
import json


def account_balance(sei_address):
    command = [
        "seid",
        "query",
        "bank",
        "balances",
        sei_address,
        "--chain-id",
        "pacific-1",
        "--node",
        "https://sei-rpc.brocha.in/",
        "--output",
        "json",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.stdout:
        return result.stdout
    else:
        return None


def format_balance(output):
    data = json.loads(output)
    if "balances" in data:
        for balance in data["balances"]:
            if balance["denom"] == "usei":
                # Convert to 6 decimal places
                amount = int(balance["amount"]) / 10**6
                # Round to 3 decimal places
                balance["amount"] = f"{amount:.3f}"
    return data


def events_search():
    '''
    command reference:
    seid query txs --events 'message.sender=cosmos1...&message.action=withdraw_delegator_reward' 
    --page 1 --limit 30 --chain-id pacific-1 --node https://sei-rpc.brocha.in/ --output json
    '''

    command = [
        "seid",
        "query",
        "txs",
        "--events",
        "wasm.action=create_pair",
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
    example_sei_wallet = 'sei1n7p4c4sjxap8nvhfwhgss6xyht2v60fc0423ey'

    query_selector = input(
        "1: Account Balance | 2: Query Txn | 3: Query Events ---> ")
    query_selector = int(query_selector)
    if query_selector == 1:
        output = account_balance(example_sei_wallet)
        if output:
            formatted_output = format_balance(output)
            print(20 * "=")
            print("Account Balance")
            balance_in_sei = formatted_output.get('balances')
            if balance_in_sei and len(balance_in_sei) >= 2:
                balance = balance_in_sei[1]['amount']
                print(f"Balance in SEI: {balance}")
            else:
                print("No balance in SEI found.")
            print(20 * "=")

    if query_selector == 3:
        output = events_search()
        if output:
            print("Events Result:")
            print(json.dumps(output, indent=2))
        else:
            print("Events Search is None...")
