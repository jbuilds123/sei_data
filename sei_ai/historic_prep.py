import json
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.utils import pad_sequences


def load_pairs(file_path):
    with open(file_path, "r") as file:
        return json.load(file)


def calculate_vwap(df):
    cum_sum = (df["volume"] * (df["high"] +
               df["low"] + df["close"]) / 3).cumsum()
    cum_vol = df["volume"].cumsum()
    with np.errstate(divide="ignore", invalid="ignore"):
        vwap = np.where(
            df["volume"] == 0, 0, np.where(cum_vol != 0, cum_sum / cum_vol, 0)
        )
    return vwap


def calculate_vwap_deviation(df):
    df["vwap_deviation"] = np.where(
        df["volume"] == 0, 0, abs(df["close"] - df["vwap"]) / df["vwap"]
    )
    return df


def calculate_candle_volatility(df):
    df["volatility"] = np.where(
        df["volume"] == 0, 0, (df["high"] - df["low"]) / df["open"]
    )
    return df


def calculate_winner(data, end_price, gain_threshold=15000, max_gain=2500000):
    first_open = data["open"].iloc[0]
    if first_open == 0:
        return 0
    gain_percentage = ((end_price - first_open) / first_open) * 100
    if gain_percentage > max_gain or gain_percentage < gain_threshold:
        return 0
    return 1


def calculate_gain_percentage(data, end_price):
    first_open = data["open"].iloc[0] * 1.10
    if first_open == 0:
        return 0
    gain_percentage = ((end_price - first_open) / first_open) * 100
    return gain_percentage


def replace_nan_with_mean(sequences):
    # Convert to a numpy array for easier manipulation
    sequences_np = np.array(sequences)
    # Calculate the mean for each feature/column, ignoring NaN values
    col_means = np.nanmean(sequences_np, axis=(0, 1))
    # Find indices where NaN values are present
    inds = np.where(np.isnan(sequences_np))
    # Replace NaNs with the mean of the respective column
    sequences_np[inds] = np.take(col_means, inds[2])
    return sequences_np


def has_excessive_gain_between_candles(ohlcv_list, max_gain=12000):
    for i in range(len(ohlcv_list) - 1):
        open_price = float(ohlcv_list[i]["open"])
        next_close_price = float(ohlcv_list[i + 1]["close"])
        if open_price > 0:
            gain = (next_close_price - open_price) / open_price * 100
            if gain > max_gain:
                return True
    return False


def main():
    pairs = load_pairs("sei_pairs/historic_pairs.json")
    scaler = StandardScaler()

    sequence_data_list = []
    labels = []
    pair_addresses = []
    gain_percentages = []
    end_prices = []
    original_close_prices_sequences = []
    sequence_lengths = []
    low_data_losers = 0
    skipped_pairs_due_to_volume = 0
    pairs_with_ohlcv = 0

    # print total pairs raw
    print("-------------------------------")
    print(f"Total Pairs Processed: {len(pairs)}")

    for pair, pair_data in pairs.items():
        if "ohlcv" in pair_data and pair_data["ohlcv"]:
            pairs_with_ohlcv += 1
            df = pd.DataFrame(pair_data["ohlcv"])

            # Convert string columns to numeric
            df["open"] = pd.to_numeric(df["open"], errors='coerce')
            df["high"] = pd.to_numeric(df["high"], errors='coerce')
            df["low"] = pd.to_numeric(df["low"], errors='coerce')
            df["close"] = pd.to_numeric(df["close"], errors='coerce')
            df["volume"] = pd.to_numeric(df["volume"], errors='coerce')

            # Calculate average volume for the first 1-3 candles
            avg_volume = df.head(3)["volume"].mean()

            # Skip pairs with low average volume
            if avg_volume < 25:
                skipped_pairs_due_to_volume += 1
                continue

            # feature calculations
            df["vwap"] = calculate_vwap(df)
            df = calculate_vwap_deviation(df)
            df = calculate_candle_volatility(df)

            end_price = pair_data.get("end_price", df["close"].iloc[-1])
            gain_percentage = calculate_gain_percentage(df, end_price)
            is_winner = None

            # Check for excessive gain between candles
            if has_excessive_gain_between_candles(df.to_dict('records')):
                is_winner = 0  # Mark as non-winner due to excessive gain
            else:
                # Check for low data losers
                if len(df) <= 10:
                    is_winner = 0
                    low_data_losers += 1
                else:
                    is_winner = calculate_winner(df, end_price)

            # Store original close prices for each sequence length before normalization
            for i in range(1, min(len(df), 8)):  # Start from 1 candle to 8
                original_close_prices = df["close"].head(i).values
                original_close_prices_sequences.append(
                    original_close_prices.tolist())

            normalized_data = scaler.fit_transform(
                df[["close", "volume", "vwap", "vwap_deviation", "volatility"]]
            )
            df[["close", "volume", "vwap", "vwap_deviation",
                "volatility"]] = normalized_data

            for i in range(1, min(len(df), 8)):
                sequence = df[["close", "volume", "vwap",
                               "vwap_deviation", "volatility"]].head(i).values
                sequence_data_list.append(sequence)
                gain_percentages.append(gain_percentage)
                end_prices.append(end_price)
                pair_addresses.append(pair)
                sequence_lengths.append(i)
                labels.append(is_winner)

    max_sequence_length = 7

    padded_sequences = pad_sequences(
        sequence_data_list, maxlen=max_sequence_length, padding="post", dtype="float32")

    # Check for NaN values and replace them if found
    if np.isnan(padded_sequences).any():
        padded_sequences = replace_nan_with_mean(padded_sequences)

    labels = np.array(labels)
    pair_addresses = np.array(pair_addresses)
    gain_percentages = np.array(gain_percentages)
    end_prices = np.array(end_prices, dtype="float32")
    sequence_lengths = np.array(sequence_lengths, dtype=int)

    # For example, print the first few processed data points
    '''
    for i in range(min(20, len(padded_sequences))):
        print(
            f"Pair: {pair_addresses[i]}, Gain: {gain_percentages[i]:.2f}%, Label: {labels[i]}")
    '''

    # Save the prepared data for training
    with open("sei_ai/training_data.pkl", "wb") as f:
        pickle.dump(
            (
                padded_sequences,
                labels,
                pair_addresses,
                gain_percentages,
                end_prices,
                original_close_prices_sequences,
                sequence_lengths,
            ),
            f,
        )

    # Calculate the percentage of pairs that are winners
    # Since labels are 0 or 1, mean will give the proportion of 1s
    winner_percentage = np.mean(labels) * 100

    # Print the winner percentage
    print(f"Pairs with OHLCV data: {pairs_with_ohlcv}")
    print(f"Percentage of pairs that are winners: {winner_percentage:.2f}%")
    print(f"Low Avg Vol: {skipped_pairs_due_to_volume}")
    print(f"Low Data Losers: {low_data_losers}")
    print("-------------------------------")


if __name__ == "__main__":
    main()
