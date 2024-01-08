import numpy as np
import pickle
import json
import logging
import os
import requests
import time
import asyncio
import traceback
import pytz
from datetime import datetime, timezone
import tensorflow as tf
from datetime import datetime
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    matthews_corrcoef,
    brier_score_loss,
    log_loss,
)
from imblearn.over_sampling import SMOTE
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers.legacy import Adam
from db.session import SessionLocal, engine
from db.base import Base
from db.models import PaperTrades
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv

load_dotenv()

ds_url = os.getenv("pt_wh")
ping_role = os.getenv("sei_ping_role")

# Create tables
Base.metadata.create_all(bind=engine)

live_pair_addresses = []

# Configure logging
log_directory = "logs"
log_filename = "live_pair_logs.log"
if not os.path.exists(log_directory):
    os.makedirs(log_directory)


# Custom formatter class
class ESTFormatter(logging.Formatter):
    est = pytz.timezone("US/Eastern")

    def formatTime(self, record, datefmt=None):
        ct = datetime.fromtimestamp(record.created)
        localized_ct = self.est.localize(ct)
        if datefmt:
            return localized_ct.strftime(datefmt)
        else:
            return localized_ct.strftime("%Y-%m-%d %I:%M:%S %p %Z")


# Setup logger with custom formatter
logger = logging.getLogger()
logger.setLevel(logging.INFO)
fh = logging.FileHandler(os.path.join(log_directory, log_filename))
formatter = ESTFormatter("%(asctime)s %(message)s", "%Y-%m-%d %I:%M:%S %p %Z")
fh.setFormatter(formatter)
logger.addHandler(fh)


def send_discord_webhook(pair_address, paper_trade):
    if not ds_url:
        print(
            "Discord webhook URL is not set. Please check your environment variables."
        )
        return

    if pair_address is not None and paper_trade is not None:
        # Construct the message to be sent, including the role ping
        message_content = (
            f"New SEI AI Paper Trade 📊 <@&{ping_role}>" if ping_role else ""
        )
        message = {
            "content": message_content,
            "embeds": [
                {
                    "title": f"***${paper_trade['network']} Network***",
                    "description": f"**Pair Address:** [{pair_address}](https://www.geckoterminal.com/sei-network/pools/{pair_address})\n\n**Entry Price:** *{paper_trade['entry_price']}*\n\n**Entry Candle:** *{paper_trade['entry_candle']}*",
                    "color": 16711680,
                    "footer": {
                        "text": "This is just a simulation test trade that is counting as a papertrade in a fully live environment. I suggest to watch and judge the AI, not follow just yet. Exercise patience while it learns the market!",
                    },
                }
            ],
        }

    try:
        response = requests.post(ds_url, json=message)
        # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
        response.raise_for_status()
        print("Webhook sent successfully.")
    except requests.exceptions.HTTPError as errh:
        print(f"HTTP Error: {errh}")
    except requests.exceptions.ConnectionError as errc:
        print(f"Error Connecting: {errc}")
    except requests.exceptions.Timeout as errt:
        print(f"Timeout Error: {errt}")
    except requests.exceptions.RequestException as err:
        print(f"Error: {err}")


def create_paper_trade_entry(pair_address, trade_details):
    session = SessionLocal()
    trade_details_converted = convert_numpy(trade_details)
    try:
        # Attempt to create a new trade entry
        new_trade = PaperTrades(
            pair_address=pair_address,
            created_at=int(time.time()),
            trade_details=trade_details_converted,
        )
        session.add(new_trade)
        session.commit()
        print(f"Paper trade entry created for pair {pair_address}")

    except SQLAlchemyError as e:
        session.rollback()
        if "duplicate key value violates unique constraint" in str(e):
            print(f"Duplicate entry for pair {pair_address} not created.")
        else:
            print(f"Error creating paper trade entry: {e}")
    finally:
        session.close()


def convert_numpy(obj):
    if isinstance(obj, np.generic):
        return obj.item()  # Replace np.asscalar(obj) with obj.item()
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    else:
        return obj


def load_data(training_file, live_file):
    with open(training_file, "rb") as f:
        training_data = pickle.load(f)

    try:
        with open(live_file, "rb") as f:
            live_data = pickle.load(f)

        # add pairs to live pairs list
        live_pair_addresses.extend(live_data[2])

        # begin total pairs debug

        # Remove duplicates and debug live pair addresses
        unique_live_pair_addresses = set(live_pair_addresses)
        print(f"Total Live Pairs: {len(unique_live_pair_addresses)}")

        # Indices of uniform data
        # Adjust these indices based on your data structure
        uniform_data_indices = [0, 1, 2, 3, 4, 6]

        # Concatenating uniform data
        concatenated_uniform_data = [
            np.concatenate([training_data[i], live_data[i]])
            for i in uniform_data_indices
        ]

        # Combine all data
        training_data = tuple(concatenated_uniform_data)

    except FileNotFoundError:
        print(
            f"No live data file found at {live_file}, proceeding with only training data."
        )

    return training_data


def save_model(model, base_filename="model_v", directory="sei_ai/tuned_models"):
    if not os.path.exists(directory):
        os.makedirs(directory)

    version = 1
    while os.path.exists(f"{directory}/{base_filename}{version}.keras"):
        version += 1

    model_path = f"{directory}/{base_filename}{version}.keras"
    model.save(model_path)
    print(f"Model saved as {model_path}")
    return model_path


def load_close_data(training_file, live_file):
    with open(training_file, "rb") as f:
        training_data = pickle.load(f)

    try:
        with open(live_file, "rb") as f:
            live_data = pickle.load(f)

        # Handling non-uniform data
        # Assuming original_close_prices_sequences is at index 5
        training_close_prices = training_data[5]
        live_close_prices = live_data[5]

        # Explicitly keep as a list of arrays
        combined_close_prices = list(
            training_close_prices) + list(live_close_prices)

    except FileNotFoundError:
        print(
            f"No live data file found at {live_file}, proceeding with only training data."
        )

    return combined_close_prices


def read_executed_trades(file_path):
    try:
        with open(file_path, "r") as file:
            return set(json.load(file))
    except (FileNotFoundError, json.JSONDecodeError):
        return set()


def save_executed_trades(file_path, executed_trades):
    with open(file_path, "w") as file:
        json.dump(list(executed_trades), file)


def apply_smote(X_train, y_train, sequence_shape):
    # Impute NaN values
    imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
    X_train_imputed = imputer.fit_transform(
        X_train.reshape(X_train.shape[0], -1))

    # Apply SMOTE
    smote = SMOTE()
    X_train_sm, y_train_sm = smote.fit_resample(X_train_imputed, y_train)

    # Reshape back to original shape
    X_train_sm = X_train_sm.reshape(
        X_train_sm.shape[0], sequence_shape[1], sequence_shape[2]
    )

    return X_train_sm, y_train_sm


def compute_class_weights(y_train_sm):
    # Calculate the weight for class 0
    weight_for_0 = (1 / np.sum(y_train_sm == 0)) * (len(y_train_sm) / 2.0)
    # Calculate the weight for class 1
    weight_for_1 = (1 / np.sum(y_train_sm == 1)) * (len(y_train_sm) / 2.0)

    class_weights_dict = {0: weight_for_0, 1: weight_for_1}
    return class_weights_dict


def create_lstm_model(input_shape):
    model = Sequential(
        [
            LSTM(128, return_sequences=True, input_shape=input_shape),
            BatchNormalization(),
            Dropout(0.5),
            LSTM(128),
            BatchNormalization(),
            Dropout(0.5),
            Dense(64, activation="relu"),
            Dropout(0.3),
            Dense(1, activation="sigmoid"),
        ]
    )
    return model


def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -tf.reduce_sum(
            alpha * tf.pow(1.0 - pt_1, gamma) * tf.math.log(pt_1)
        ) - tf.reduce_sum((1 - alpha) * tf.pow(pt_0, gamma) * tf.math.log(1.0 - pt_0))

    return focal_loss_fixed


def compile_and_train_model(
    model, X_train_sm, y_train_sm, class_weights_dict, save_model_flag=True
):
    optimizer = Adam(learning_rate=0.001)  # Use the legacy Adam optimizer
    model.compile(optimizer=optimizer, loss=focal_loss(), metrics=["accuracy"])

    early_stopping = EarlyStopping(monitor="val_loss", patience=10)
    model_checkpoint = ModelCheckpoint(
        "best_model.keras", save_best_only=True, monitor="val_loss", mode="min"
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.1, patience=5, min_lr=0.0001
    )

    history = model.fit(
        X_train_sm,
        y_train_sm,
        epochs=100,
        batch_size=64,
        validation_split=0.2,
        class_weight=class_weights_dict,
        callbacks=[early_stopping, model_checkpoint, reduce_lr],
    )

    if save_model_flag:
        model_path = save_model(model)
    else:
        model.load_weights("best_model.keras")
    return model


def evaluate_model(model, X_test, y_test, threshold=0.70):
    y_pred = model.predict(X_test).flatten()

    # Convert probabilities to class labels
    y_pred_class = (y_pred > threshold).astype(int)
    # threshold was 0.75

    # Calculate and print the total number of predictions above and below the threshold
    total_above_threshold = np.sum(y_pred_class)
    total_below_threshold = len(y_pred_class) - total_above_threshold
    print(
        f"Total Predictions Above Threshold: {total_above_threshold}, Below Threshold: {total_below_threshold}"
    )

    test_loss = log_loss(y_test, y_pred)  # Log Loss
    test_auc = roc_auc_score(y_test, y_pred)  # AUC-ROC Score
    precision = precision_score(y_test, y_pred_class)
    recall = recall_score(y_test, y_pred_class)
    f1 = f1_score(y_test, y_pred_class)
    mcc = matthews_corrcoef(y_test, y_pred_class)
    brier = brier_score_loss(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred_class)

    """
    print(f"Log Loss: {test_loss:.2f}")
    print(f"AUC-ROC Score: {test_auc:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"Matthews Correlation Coefficient: {mcc:.2f}")
    print(f"Brier Score: {brier:.2f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    """

    return y_pred


def load_pretrained_model(model_version, input_shape):
    model_path = f"sei_ai/tuned_models/model_v{model_version}.keras"

    if os.path.exists(model_path):
        model = load_model(
            model_path, custom_objects={"focal_loss_fixed": focal_loss()}
        )
    else:
        print("No pre-trained model found. Creating a new model.")
        model = create_lstm_model(input_shape)
    return model


def calculate_performance_metrics(
    predictions,
    pairs_test,
    y_test,
    sequence_lengths,
    pair_addresses,
    original_close_prices_sequences,
    threshold,
    trade_size,
):
    global live_pair_addresses
    executed_trades_file = "sei_ai/paper_trades.json"
    executed_trades = read_executed_trades(executed_trades_file)

    total_spent = 0
    total_dollar_gain_loss = 0
    total_buy_in_fees = 0
    total_sell_fees = 0
    wins = 0
    buy_in_fee = 0.07
    sell_fee = 0.07
    tp, fp, tn, fn = 0, 0, 0, 0
    live_trades = 0
    false_negatives_pairs = []
    trade_simulations = []
    first_predictions = set()

    pair_addresses_list = pair_addresses.tolist()

    for i, (pair_address, prob, actual_label, seq_length) in enumerate(
        zip(pairs_test, predictions, y_test, sequence_lengths)
    ):
        pred_label = int(prob > threshold)
        if pred_label == 1 and actual_label == 1:
            tp += 1
        elif pred_label == 1 and actual_label == 0:
            fp += 1
        elif pred_label == 0 and actual_label == 1:
            fn += 1
        elif pred_label == 0 and actual_label == 0:
            tn += 1

        # Ensure only one trade per pair
        if pair_address in first_predictions:
            continue

        # Check if the sequence length is greater than 2 and the probability is above the threshold
        if seq_length > 2 and prob > threshold:
            first_predictions.add(pair_address)
            pair_index = pair_addresses_list.index(pair_address)
            fixed_index = seq_length - 1
            entry_price = (
                original_close_prices_sequences[pair_index +
                                                fixed_index][-1] * 1.2
            )

            # Processing for live pairs
            if (
                pair_address in live_pair_addresses
                and pair_address not in executed_trades
            ):
                live_trades += 1
                paper_trade = {
                    "pair_address": pair_address,
                    "network": "sei",
                    "entry_price": entry_price,
                    "entry_candle": seq_length,
                }
                # print("Live Paper Trade", paper_trade)
                send_discord_webhook(pair_address, paper_trade)
                executed_trades.add(pair_address)
                save_executed_trades(executed_trades_file, executed_trades)
            # Processing for simulated trades
            else:
                adjusted_exit_price = end_prices[pair_index] * 0.9
                gain_loss_percent = (
                    (adjusted_exit_price - entry_price) / entry_price
                ) * 100
                total_buy_in_fees += buy_in_fee

                if gain_loss_percent > 5000000:
                    dollar_gain_loss = -(trade_size + buy_in_fee)
                    gain_loss_percent = -100
                else:
                    potential_dollar_gain_loss = (
                        gain_loss_percent / 100) * trade_size
                    net_dollar_gain_loss = potential_dollar_gain_loss - buy_in_fee
                    if net_dollar_gain_loss >= trade_size + buy_in_fee:
                        dollar_gain_loss = net_dollar_gain_loss - sell_fee
                        total_sell_fees += sell_fee
                    else:
                        dollar_gain_loss = -(trade_size + buy_in_fee)
                        gain_loss_percent = -100

                trade_simulations.append(
                    (
                        pair_address,
                        entry_price,
                        adjusted_exit_price,
                        gain_loss_percent,
                        dollar_gain_loss,
                        seq_length,
                    )
                )
                total_spent += trade_size + buy_in_fee
                total_dollar_gain_loss += dollar_gain_loss
                if dollar_gain_loss > 0:
                    wins += 1

    # Calculate other performance metrics
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    win_rate = (
        (wins / len(trade_simulations)) *
        100 if len(trade_simulations) > 0 else 0
    )
    overall_growth_loss_percent = (
        (total_dollar_gain_loss / total_spent) * 100 if total_spent > 0 else 0
    )

    # total trades made
    trades_made = len(trade_simulations)

    # Sort trade_simulations by dollar_gain in descending order
    sorted_trade_simulations = sorted(
        trade_simulations, key=lambda x: x[4], reverse=True
    )

    for trade in sorted_trade_simulations:
        (
            pair_address,
            entry_price,
            adjusted_exit_price,
            gain_loss_percent,
            dollar_gain_loss,
            entry_candle,
        ) = trade

        live_status = pair_address in live_pair_addresses

        # Convert NumPy array or scalar to native Python type
        if isinstance(entry_price, np.ndarray):
            entry_price = entry_price.tolist()  # Convert array to list
        elif np.isscalar(entry_price):
            entry_price = float(entry_price)  # Convert scalar to float

        trade_details = {
            "pair": pair_address,
            "entry_candle": entry_candle,
            "entry_price": entry_price,
            "adjusted_exit": adjusted_exit_price,
            "percent_gain": round(gain_loss_percent, 2),
            "dollar_gain": round(dollar_gain_loss, 2),
            "is_live_pair": live_status,
        }
        # print(trade_details)

    print(25 * "-")
    print("===Prediction Stats===")
    print(f"True Positives: {tp} (Rate: {tpr:.2%})")
    print(f"False Positives: {fp} (Rate: {fpr:.2%})")
    print(f"True Negatives: {tn} (Rate: {tnr:.2%})")
    print(f"False Negatives: {fn} (Rate: {fnr:.2%})")
    print("\nFalse Negative Pairs:")
    for pair in false_negatives_pairs:
        print(f"  -- {pair}")
    print()
    print("===Trade Stats===")
    print(f"Live Trades: {live_trades}")
    """
    print(f"{trades_made} Trades | ${trade_size} Buy-Ins")
    print(f"Total Spent: ${total_spent:.2f}")
    print(f"Buy-in Fees: ${total_buy_in_fees:.2f}")
    print(f"Sell Fees: ${total_sell_fees:.2f}")
    print(f"PNL Dollar: ${total_dollar_gain_loss:.2f}")
    print(f"PNL Percent: {overall_growth_loss_percent:.2f}%")
    print(f"Win Rate: {win_rate:.2f}%")
    """
    print()


# Model settings
probabiliy_threshold = 0.70  # was 0.55
run_on_full_data = True
train_new_model = False
model_version = 1

# Position size
trade_size = 5

# Main execution
training_file = "sei_ai/training_data.pkl"
live_file = "sei_ai/live_data.pkl"

(
    padded_sequences,
    labels,
    pair_addresses,
    gain_percentages,
    end_prices,
    sequence_lengths,
) = load_data(training_file, live_file)


combined_close_prices = load_close_data(training_file, live_file)

# Since we are not splitting data, directly use the full dataset
X_for_predictions = padded_sequences
y_for_predictions = labels
pairs_for_predictions = pair_addresses


# Use the model_version when calling load_pretrained_model
model = load_pretrained_model(
    model_version, (X_for_predictions.shape[1], X_for_predictions.shape[2])
)

# Evaluate model
predictions = evaluate_model(model, X_for_predictions, y_for_predictions)


# debug
# Define a function to colorize log messages
def colorize_log_message(message, probability):
    if probability > 0.70:
        return f"🎈🎈🎈🎈🎈🎈🎈🎈🎈🎈{message}🎈🎈🎈🎈🎈🎈🎈🎈🎈🎈"
    else:
        return message


# Store the last processed pair address
last_pair_address = None

# Debugging: Print probabilities for live pairs
for i, pair_address in enumerate(pair_addresses):
    if pair_address in live_pair_addresses:
        log_message = (
            f"Pair Address: {pair_address}, Probability = {predictions[i]:.4f}"
        )
        # Colorize log message based on probability
        colored_log_message = colorize_log_message(log_message, predictions[i])
        logging.info(colored_log_message)

        # Check if the pair address has changed
        if last_pair_address and last_pair_address != pair_address:
            # Add a divider or line break
            logging.info("")

        # Update the last processed pair address
        last_pair_address = pair_address

# Add a final divider after the last set of messages
if last_pair_address:
    logging.info("--------------------------------------------------")


# Calculate performance metrics
calculate_performance_metrics(
    predictions,
    pairs_for_predictions,
    y_for_predictions,
    sequence_lengths,
    pair_addresses,
    combined_close_prices,
    probabiliy_threshold,
    trade_size,
)
