import subprocess
import json


def run_seid_command():
    command = [
        "seid",
        "query",
        "bank",
        "balances",
        "sei1n7p4c4sjxap8nvhfwhgss6xyht2v60fc0423ey",
        "--chain-id",
        "pacific-1",
        "--node",
        "https://sei-rpc.brocha.in/",
        "--output",
        "json",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    return result.stdout


if __name__ == "__main__":
    output = run_seid_command()
    print("Output in json")
    print(json.dumps(json.loads(output), indent=2))
