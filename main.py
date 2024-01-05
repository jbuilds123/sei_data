import schedule
import time
import subprocess


def run_script():
    try:
        # Change the path to the script if necessary
        print("Fetching New Sei Pairs...")
        subprocess.run(["python", "sei_pairs/tg.py"], check=True)
        time.sleep(1)
        print("Fetching Sei OHLCV...")
        subprocess.run(["python", "sei_pairs/ohlcv.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


# Schedule the task every 30 mins
schedule.every(5).minutes.do(run_script)

# Run the schedule in a loop
while True:
    schedule.run_pending()
    time.sleep(1)
