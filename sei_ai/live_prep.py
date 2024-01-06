import json
import pickle
import subprocess
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.utils import pad_sequences


def load_pairs(file_path):
    with open(file_path, "r") as file:
        return json.load(file)


def calculate_vwap(df):
    cum_sum = (df["volume"] * (df["high"] + df["low"] + df["close"]) / 3).cumsum()
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
    pairs = load_pairs("sei_ai/live_pairs.json")
    scaler = StandardScaler()
    max_sequence_length = 7
    sequence_data_list = []
    labels = []

    # mapping data
    pair_addresses = []
    gain_percentages = []
    end_prices = []
    original_close_prices_sequences = []
    sequence_lengths = []

    # counters for printing
    low_data_losers = 0
    skipped_pairs_due_to_volume = 0
    skipped_excessive_gain = 0
    pairs_with_ohlcv = 0

    for pair, pair_data in pairs.items():
        if (
            "ohlcv" in pair_data
            and pair_data["ohlcv"]
            and not pair_data["disqualified"]
        ):
            df = pd.DataFrame(pair_data["ohlcv"])

            # Convert string columns to numeric
            df["open"] = pd.to_numeric(df["open"], errors="coerce")
            df["high"] = pd.to_numeric(df["high"], errors="coerce")
            df["low"] = pd.to_numeric(df["low"], errors="coerce")
            df["close"] = pd.to_numeric(df["close"], errors="coerce")
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

            # Check if the number of candles is between 3 and 8
            if 3 <= len(df) <= 8:
                # Check for excessive gain between candles
                if has_excessive_gain_between_candles(df.to_dict("records")):
                    skipped_excessive_gain += 1
                    # print(f"|live_prep| Excess gains detected for {pair}")
                    continue

                # Calculate average volume for the first 1-3 candles
                avg_volume = df.head(3)["volume"].mean()

                # Skip pairs with low average volume
                if avg_volume < 25:
                    skipped_pairs_due_to_volume += 1
                    continue

                pairs_with_ohlcv += 1

                # feature calculations
                df["vwap"] = calculate_vwap(df)
                df = calculate_vwap_deviation(df)
                df = calculate_candle_volatility(df)

                end_price = pair_data.get("end_price", df["close"].iloc[-1])
                gain_percentage = calculate_gain_percentage(df, end_price)
                is_winner = 0

                # Store original close prices for each sequence length before normalization
                for i in range(1, min(len(df), 8)):  # Start from 1 candle to 8
                    original_close_prices = df["close"].head(i).values
                    original_close_prices_sequences.append(
                        original_close_prices.tolist()
                    )

                normalized_data = scaler.fit_transform(
                    df[["close", "volume", "vwap", "vwap_deviation", "volatility"]]
                )
                df[
                    ["close", "volume", "vwap", "vwap_deviation", "volatility"]
                ] = normalized_data

                for i in range(1, min(len(df), 8)):
                    sequence = (
                        df[["close", "volume", "vwap", "vwap_deviation", "volatility"]]
                        .head(i)
                        .values
                    )
                    sequence_data_list.append(sequence)
                    gain_percentages.append(gain_percentage)
                    end_prices.append(end_price)
                    pair_addresses.append(pair)
                    sequence_lengths.append(i)
                    labels.append(is_winner)

            else:
                continue  # Skip processing candle range incorrect

    if pairs_with_ohlcv > 0:
        padded_sequences = pad_sequences(
            sequence_data_list,
            maxlen=max_sequence_length,
            padding="post",
            dtype="float32",
        )

        # Check for NaN values and replace them if found
        if np.isnan(padded_sequences).any():
            padded_sequences = replace_nan_with_mean(padded_sequences)

        labels = np.array(labels)
        pair_addresses = np.array(pair_addresses)
        gain_percentages = np.array(gain_percentages)
        end_prices = np.array(end_prices, dtype="float32")
        original_close_prices_sequences = [
            np.array(prices, dtype="float32")
            for prices in original_close_prices_sequences
        ]
        sequence_lengths = np.array(sequence_lengths, dtype=int)

        # Print the variables before saving
        print("-------------------------------")
        print("|live_prep| Padded Sequences:", padded_sequences)
        print("|live_prep| Labels:", labels)
        print("|live_prep| Pair Addresses:", pair_addresses)
        print("|live_prep| Gain Percentages:", gain_percentages)
        print("|live_prep| End Prices:", end_prices)
        print(
            "|live_prep| Original Close Prices Sequences:",
            original_close_prices_sequences,
        )
        print("|live_prep| Sequence Lengths:", sequence_lengths)

        # Check if any of the arrays are non-empty
        if any(
            [
                len(padded_sequences) > 0,
                len(labels) > 0,
                len(pair_addresses) > 0,
                len(gain_percentages) > 0,
                len(end_prices) > 0,
                len(original_close_prices_sequences) > 0,
                len(sequence_lengths) > 0,
            ]
        ):
            # Save the prepared data for training
            with open("sei_ai/live_data.pkl", "wb") as f:
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
            if len(labels) > 0:
                # Calculate the percentage of pairs that are winners
                winner_percentage = np.mean(labels) * 100
                print(
                    f"|live_prep| Percentage of pairs that are winners: {winner_percentage:.2f}%"
                )
            else:
                print("|live_prep| No data for winner percentage calculation.")

            # Print the winner percentage
            print(f"|live_prep| Pairs with OHLCV data: {pairs_with_ohlcv}")
            print(f"|live_prep| Low Avg Vol: {skipped_pairs_due_to_volume}")
            print(f"|live_prep| Low Data Losers: {low_data_losers}")
            print("-------------------------------")

            # Absolute path to the live_model.py script
            live_model_script = "sei_ai/live_model.py"

            # Run the live ml model script
            print("|live_prep| Running live_model.py")
            subprocess.run(["python", live_model_script])
            print("|live_prep| Done running live_model.py")

    else:
        print("-------------------------------")
        print("|live_prep| No live pairs matching conditions...")
        print(
            f"|live_prep| Low Avg Vol: {skipped_pairs_due_to_volume} | Excessive Gain: {skipped_excessive_gain}"
        )
        print("-------------------------------")


if __name__ == "__main__":
    main()
