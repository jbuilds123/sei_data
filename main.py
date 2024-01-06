import schedule
import time
import subprocess


def run_script():
    try:
        # Change the path to the script if necessary
        print("Fetching Live New Sei Pairs...")
        subprocess.run(["python", "sei_ai/live_pairs_fetch.py"], check=True)

        time.sleep(1)

        print("Fetching Live Sei OHLCV...")
        subprocess.run(["python", "sei_ai/live_ohlcv.py"], check=True)

        time.sleep(1)

        print("Live Data Prep...")
        subprocess.run(["python", "sei_ai/live_prep.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


print("Started main.py")
schedule.every(1).minutes.do(run_script)

# Run the schedule in a loop
try:
    while True:
        schedule.run_pending()
        time.sleep(1)
except KeyboardInterrupt:
    print(" ==> Stopping script gracefully")
