import json
import shutil


def erase_ohlcv_data(file_path):
    try:
        # Load the JSON data from the file
        with open(file_path, 'r') as file:
            data = json.load(file)

        # Iterate through pairs and remove OHLCV data
        for pair_info in data.values():
            if 'ohlcv' in pair_info:
                del pair_info['ohlcv']

        # Write the modified data back to the file
        try:
            with open(file_path, 'w') as new_file:
                json.dump(data, new_file, indent=4)
            print("OHLCV data erased successfully.")
        except Exception as e:
            print(f"Failed to write modified data to the file: {str(e)}")
            # Revert the file back to its original state
            shutil.copyfile(file_path + '.bak', file_path)
            print("File reverted to its original state.")

    except FileNotFoundError:
        print("File not found.")
    except json.JSONDecodeError:
        print("Error decoding JSON.")


# Predefined file path for historic pairs
file_path = 'sei_pairs/historic_pairs.json'

# Create a backup of the original file before making any changes
shutil.copyfile(file_path, file_path + '.bak')
erase_ohlcv_data(file_path)
