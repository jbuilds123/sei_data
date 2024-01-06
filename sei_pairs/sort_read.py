import json
import time
import shutil


def count_pairs_with_ohlcv_data(file_path):
    try:
        # Load the JSON data from the file
        with open(file_path, 'r') as file:
            data = json.load(file)
            total_pairs = len(data)
            pairs_with_ohlcv = 0
            oldest_timestamp = None

            # Iterate through pairs to count and find the oldest timestamp
            for pair_info in data.values():
                if 'ohlcv' in pair_info:
                    pairs_with_ohlcv += 1
                    if oldest_timestamp is None or pair_info['ohlcv'][0]['timestamp'] < oldest_timestamp:
                        oldest_timestamp = pair_info['ohlcv'][0]['timestamp']

            # Check if there are pairs with OHLCV data
            if oldest_timestamp is not None:
                current_time = int(time.time())
                hours_difference = (current_time - oldest_timestamp) // 3600
                print(f"Number of pairs in the file: {total_pairs}")
                print(f"Number of pairs with OHLCV data: {pairs_with_ohlcv}")
                print(f"Hours since the oldest pair: {hours_difference} hours")
            else:
                print("No pairs with OHLCV data found.")

            # Sort pairs by 'created_at' timestamp
            sorted_data = dict(
                sorted(data.items(), key=lambda x: x[1]['created_at']))

            # Attempt to write the sorted data back to the file
            try:
                with open(file_path, 'w') as new_file:
                    json.dump(sorted_data, new_file, indent=4)
                print("Pairs sorted by 'created_at' timestamp.")
            except Exception as e:
                print(f"Failed to write sorted data to the file: {str(e)}")
                # Revert the file back to its original state
                shutil.copyfile(file_path + '.bak', file_path)
                print("File reverted to its original state.")

    except FileNotFoundError:
        print("File not found.")
    except json.JSONDecodeError:
        print("Error decoding JSON.")


path = input("Choose your input: 1 historic | 2 live --> ")
path = int(path)

file_path = None
if path == 1:
    file_path = 'sei_pairs/historic_pairs.json'
elif path == 2:
    file_path = 'sei_ai/live_pairs.json'

# Create a backup of the original file before making any changes
shutil.copyfile(file_path, file_path + '.bak')
count_pairs_with_ohlcv_data(file_path)
