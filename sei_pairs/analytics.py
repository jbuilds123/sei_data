import json


def load_pairs(file_path):
    with open(file_path, "r") as file:
        return json.load(file)


def calculate_volatility(ohlcv):
    # Volatility is now calculated as the percentage difference between high and low prices relative to the open price
    open_price = float(ohlcv["open"])
    if open_price > 0:
        return ((float(ohlcv["high"]) - float(ohlcv["low"])) / open_price) * 100
    else:
        return 0.0


def has_low_volume_first_5_candles(ohlcv_list, min_volume=10):
    # Check if any of the first 5 candles has volume less than or equal to min_volume
    return any(float(candle["volume"]) <= min_volume for candle in ohlcv_list[:5])


def has_excessive_gain_between_candles(ohlcv_list, max_gain=12000):
    for i in range(len(ohlcv_list) - 1):
        open_price = float(ohlcv_list[i]["open"])
        next_close_price = float(ohlcv_list[i + 1]["close"])
        if open_price > 0:
            gain = (next_close_price - open_price) / open_price * 100
            if gain > max_gain:
                return True
    return False


def find_similar_pairs(all_pairs, top_pairs, summary_stats, threshold=0.1):
    """
    Find pairs not in top_pairs that have close to average volume and volatility
    for the first three candles.
    threshold: Percentage difference allowed from the average.
    """
    similar_pairs = {}
    top_pair_addresses = set(pair for pair, _ in top_pairs)

    for pair, data in all_pairs.items():
        if pair in top_pair_addresses or "ohlcv" not in data or len(data["ohlcv"]) < 3:
            continue

        is_similar = True
        for i in range(3):
            candle_stats = summary_stats[f"candle_{i+1}"]
            ohlcv = data["ohlcv"][i]
            volume = float(ohlcv["volume"])
            volatility = calculate_volatility(ohlcv)

            # Check if volume and volatility are within the threshold of the average
            if not (
                candle_stats["average_volume"] * (1 - threshold)
                <= volume
                <= candle_stats["average_volume"] * (1 + threshold)
                and candle_stats["average_volatility"] * (1 - threshold)
                <= volatility
                <= candle_stats["average_volatility"] * (1 + threshold)
            ):
                is_similar = False
                break

        if is_similar:
            first_open = float(data["ohlcv"][0]["open"])
            highest_close = max(float(candle["close"])
                                for candle in data["ohlcv"])
            gain = (highest_close - first_open) / first_open * 100
            similar_pairs[pair] = {"gain": gain, "ohlcv": data["ohlcv"]}

    return similar_pairs


def calculate_summary_statistics(top_pairs):
    volume_stats = [[], [], []]  # For first 3 candles
    volatility_stats = [[], [], []]  # For first 3 candles

    for _, data in top_pairs:
        # Check if there are at least 3 candles
        if len(data["ohlcv"]) < 3:
            continue

        for i in range(3):
            ohlcv = data["ohlcv"][i]
            volume = float(ohlcv["volume"])
            volatility = calculate_volatility(ohlcv)
            volume_stats[i].append(volume)
            volatility_stats[i].append(volatility)

    # Calculating statistics
    summary = {}
    for i in range(3):
        summary[f"candle_{i+1}"] = {
            "average_volume": sum(volume_stats[i]) / len(volume_stats[i]),
            "highest_volume": max(volume_stats[i]),
            "lowest_volume": min(volume_stats[i]),
            "average_volatility": sum(volatility_stats[i]) / len(volatility_stats[i]),
            "highest_volatility": max(volatility_stats[i]),
            "lowest_volatility": min(volatility_stats[i]),
        }
    return summary


def main():
    pairs = load_pairs("sei_pairs/historic_pairs.json")
    gain_pairs = {}

    for pair, data in pairs.items():
        if "ohlcv" in data and data["ohlcv"]:
            if has_low_volume_first_5_candles(data["ohlcv"]) or has_excessive_gain_between_candles(data["ohlcv"]):
                continue  # Disqualify pairs with low volume or excessive gain

            first_open = 1.1 * float(data["ohlcv"][0]["open"])
            highest_close = max(float(candle["close"])
                                for candle in data["ohlcv"])
            gain = (highest_close - first_open) / first_open * 100

            # Disqualify pairs with gains larger than 5,000,000%
            if gain <= 4000000:
                gain_pairs[pair] = {
                    "gain": gain,
                    "ohlcv": data["ohlcv"],
                    "first_open": first_open,
                    "highest_close": highest_close,
                }

    top_10_pairs = sorted(gain_pairs.items(), key=lambda x: x[1]["gain"], reverse=True)[
        :10
    ]

    for rank, (pair, data) in enumerate(top_10_pairs, start=1):
        print(f"Pair {rank}: {pair}")
        print(f"Percentage Gain: {data['gain']:.2f}%")
        print(f"First Open Price: {data['first_open']:.18f}")
        print(f"Highest Close Price: {data['highest_close']:.18f}")
        print("First 3 Candles:")
        for ohlcv in data["ohlcv"][:3]:
            print(f"  Timestamp: {ohlcv['timestamp']}")
            print(f"  Open: {ohlcv['open']}")
            print(f"  High: {ohlcv['high']}")
            print(f"  Low: {ohlcv['low']}")
            print(f"  Close: {ohlcv['close']}")
            print(f"  Volume: {ohlcv['volume']}")
            print(f"  Volatility: {calculate_volatility(ohlcv):.2f}%")
            print()
        print("-" * 40)

    # Calculating and printing summary statistics
    summary_stats = calculate_summary_statistics(top_10_pairs)
    print("Summary Statistics for Top 3 Pairs:")
    for i in range(3):
        stats = summary_stats[f"candle_{i+1}"]
        print(f"Candle {i+1}:")
        print(f"  Average Volume: {stats['average_volume']:.2f}")
        print(f"  Highest Volume: {stats['highest_volume']:.2f}")
        print(f"  Lowest Volume: {stats['lowest_volume']:.2f}")
        print(f"  Average Volatility: {stats['average_volatility']:.2f}")
        print(f"  Highest Volatility: {stats['highest_volatility']:.2f}")
        print(f"  Lowest Volatility: {stats['lowest_volatility']:.2f}")
        print()

    # Finding similar pairs based on average volume and volatility
    similar_pairs = find_similar_pairs(pairs, top_10_pairs, summary_stats)

    print("\nPairs Similar to Top 10 based on Volume and Volatility:")
    if similar_pairs:
        for pair, data in similar_pairs.items():
            print(f"Pair: {pair}")
            print(f"Percentage Gain: {data['gain']:.2f}%")
            for i, ohlcv in enumerate(data["ohlcv"][:3]):
                print(f"  Candle {i+1}:")
                print(f"    Volume: {ohlcv['volume']}")
                print(f"    Volatility: {calculate_volatility(ohlcv):.18f}")
            print("-" * 40)
    else:
        print("No similar pairs found based on the given criteria.")


if __name__ == "__main__":
    main()
