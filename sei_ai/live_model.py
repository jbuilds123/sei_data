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
ping_role = os.getenv("ping_role")

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
    est = pytz.timezone('US/Eastern')

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
formatter = ESTFormatter('%(asctime)s %(message)s', "%Y-%m-%d %I:%M:%S %p %Z")
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
            f"New Paper Trade Detected<@&{ping_role}>" if ping_role else ""
        )
        message = {
            "content": message_content,
            "embeds": [
                {
                    "title": f"Paper Trade Notification for ${paper_trade['network']}",
                    "description": f"Pair Address: {pair_address}\nEntry Price: {paper_trade['entry_price']}\nEntry Candle: {paper_trade['entry_candle']}\n Actual Candle Entry: {paper_trade['current_candle']}",
                    "color": 16711680,
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

        '''
        # Debug: Print sample of training data
        print("Sample of Training Data:")
        for i in range(5):  # Adjust number of samples as needed
            # Assuming 0 index is sequences
            print(f"Training Data Sample {i}: {training_data[0][i]}")
            # Assuming 1 index is labels
            print(f"Label: {training_data[1][i]}")

        # Debug: Print sample of live data
        print("Sample of Live Data:")
        for i in range(5):  # Adjust number of samples as needed
            # Assuming 0 index is sequences
            print(f"Live Data Sample {i}: {live_data[0][i]}")
            print(f"Label: {live_data[1][i]}")  # Assuming 1 index is labels
        '''
        # end debug

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


def evaluate_model(model, X_test, y_test, threshold=0.55):
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

    # end debug

    test_loss = log_loss(y_test, y_pred)  # Log Loss
    test_auc = roc_auc_score(y_test, y_pred)  # AUC-ROC Score
    precision = precision_score(y_test, y_pred_class)
    recall = recall_score(y_test, y_pred_class)
    f1 = f1_score(y_test, y_pred_class)
    mcc = matthews_corrcoef(y_test, y_pred_class)
    brier = brier_score_loss(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred_class)

    '''
    print(f"Log Loss: {test_loss:.2f}")
    print(f"AUC-ROC Score: {test_auc:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"Matthews Correlation Coefficient: {mcc:.2f}")
    print(f"Brier Score: {brier:.2f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    '''

    return y_pred


def load_pretrained_model(model_version, input_shape):
    model_path = f"sei_ai/tuned_models/model_v{model_version}.keras"

    if os.path.exists(model_path):
        print(f"Loading pre-trained model from {model_path}")
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
    pair_max_prob, pair_labels, pair_entry_candle, trade_simulations = {}, {}, {}, []
    total_spent = 0
    total_dollar_gain_loss = 0
    total_buy_in_fees = 0
    total_sell_fees = 0
    wins = 0
    buy_in_fee = 0.07
    sell_fee = 0.07
    trades_made = 0
    false_negatives_pairs = []

    # debug
    # Initialize counter for pairs with predictions
    total_pairs_set = set(pair_addresses)
    print("Total Pairs: ", len(total_pairs_set))

    pair_addresses_list = pair_addresses.tolist()

    for pair_address, prob, actual_label, seq_length in zip(
        pairs_test, predictions, y_test, sequence_lengths
    ):
        if pair_address not in pair_max_prob or prob > pair_max_prob[pair_address]:
            pair_max_prob[pair_address] = prob
            pair_labels[pair_address] = actual_label
            pair_entry_candle[pair_address] = seq_length

    final_predictions = {
        pair: (int(prob > threshold), label)
        for pair, prob, label in zip(
            pair_max_prob.keys(), pair_max_prob.values(), pair_labels.values()
        )
    }

    tp, fp, tn, fn = 0, 0, 0, 0
    total_pos_pred, total_neg_pred = 0, 0
    live_trades = 0

    for pair_address, (prediction, actual_label) in final_predictions.items():
        entry_candle = pair_entry_candle[pair_address]

    if pair_address in live_pair_addresses:
        # For live pairs, check if the probability is above threshold for any candle between 2 and 7
        try:
            if 2 <= entry_candle <= 7 and pair_max_prob[pair_address] > threshold:
                logging.info(
                    f"Live trade found for candle {entry_candle}, pair {pair_address}")
                live_trades += 1
                entry_price = (
                    original_close_prices_sequences[pair_index +
                                                    fixed_index][-1] * 1.2
                )
                total_buy_in_fees += buy_in_fee

                # For live pairs, use the actual length of the sequence
                pair_index = pair_addresses_list.index(pair_address)
                current_candle_number = len(
                    original_close_prices_sequences[pair_index])

                # live trade details and execution
                paper_trade = {
                    "network": "sei",
                    "entry_price": entry_price,
                    "entry_candle": entry_candle,
                    "current_candle": current_candle_number,
                }
                create_paper_trade_entry(pair_address, paper_trade)
                send_discord_webhook(pair_address, paper_trade)
                logging.info("Discord wh and paper trade entry made")
        except Exception as e:
            logging.error(
                f"Error in processing live trade for {pair_address}: {str(e)}")
            logging.error(traceback.format_exc())  # Log detailed traceback
    else:
        # Only proceed if the entry candle is within the desired range (3rd to 7th candle)
        if 3 <= entry_candle <= 7:
            if prediction == 1:
                total_pos_pred += 1
                pair_index = pair_addresses_list.index(pair_address)
                fixed_index = entry_candle - 1

                entry_price = (
                    original_close_prices_sequences[pair_index +
                                                    fixed_index][-1] * 1.2
                )

                adjusted_exit_price = end_prices[pair_index] * 0.9
                gain_loss_percent = (
                    (adjusted_exit_price - entry_price) / entry_price
                ) * 100

                total_buy_in_fees += buy_in_fee

                # Check if gain percentage is over 5 million
                if gain_loss_percent > 5000000:
                    dollar_gain_loss = -(trade_size + buy_in_fee)
                    gain_loss_percent = -100  # -100% gain
                else:
                    potential_dollar_gain_loss = (
                        gain_loss_percent / 100) * trade_size
                    net_dollar_gain_loss = potential_dollar_gain_loss - buy_in_fee

                    if net_dollar_gain_loss >= trade_size + buy_in_fee:
                        dollar_gain_loss = net_dollar_gain_loss - sell_fee
                        total_sell_fees += sell_fee
                    else:
                        dollar_gain_loss = -(trade_size + buy_in_fee)
                        gain_loss_percent = -100  # Marking as a 100% loss

                trade_simulations.append(
                    (
                        pair_address,
                        entry_price,
                        adjusted_exit_price,
                        gain_loss_percent,
                        dollar_gain_loss,
                        entry_candle,
                    )
                )
                total_spent += trade_size + buy_in_fee
                total_dollar_gain_loss += dollar_gain_loss
                if dollar_gain_loss > 0:
                    wins += 1
                if actual_label == 1:
                    tp += 1
                else:
                    fp += 1
            else:
                total_neg_pred += 1
                if actual_label == 1:
                    fn += 1
                else:
                    tn += 1

        # Check for false negatives
        if prediction == 0 and actual_label == 1:
            false_negatives_pairs.append(pair_address)

    trade_simulations.sort(key=lambda x: x[3], reverse=True)

    tpr = tp / total_pos_pred if total_pos_pred > 0 else 0
    fpr = fp / total_pos_pred if total_pos_pred > 0 else 0
    tnr = tn / total_neg_pred if total_neg_pred > 0 else 0
    fnr = fn / total_neg_pred if total_neg_pred > 0 else 0

    total_predictions = total_pos_pred + total_neg_pred
    win_rate = (wins / total_pos_pred) * 100 if total_pos_pred > 0 else 0
    overall_growth_loss_percent = (
        total_dollar_gain_loss / total_spent * 100 if total_spent > 0 else 0
    )

    # total trades made
    trades_made = len(trade_simulations)

    for trade in trade_simulations:
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

        if live_status:
            # For live pairs, use the actual length of the sequence
            pair_index = pair_addresses_list.index(pair_address)
            current_candle_number = len(
                original_close_prices_sequences[pair_index])

            # Check if the entry candle is the current candle
            if current_candle_number == entry_candle:
                live_trades += 1
                paper_trade = {
                    "network": "sei",
                    "entry_price": entry_price,
                    "entry_candle": entry_candle,
                }
                create_paper_trade_entry(pair_address, paper_trade)
                send_discord_webhook(pair_address, paper_trade)
                # Exclude this pair from future processing
                # exclude_pair(pair_address)
            else:
                live_trades += 1
                paper_trade = {
                    "network": "sei",
                    "entry_price": entry_price,
                    "entry_candle": entry_candle,
                }
                create_paper_trade_entry(pair_address, paper_trade)
                pair_address = None
                paper_trade = None
                send_discord_webhook(pair_address, paper_trade)

    print(25 * "-")
    '''
    print("===Prediction Stats===")
    print(
        f"Total Predictions Made (including rejections): {total_predictions}")
    print(f"Positive Predictions: {total_pos_pred}")
    print(f"True Positives: {tp} (Rate: {tpr:.2%})")
    print(f"False Positives: {fp} (Rate: {fpr:.2%})")
    print(f"True Negatives: {tn} (Rate: {tnr:.2%})")
    print(f"False Negatives: {fn} (Rate: {fnr:.2%})")
    print("\nFalse Negative Pairs:")
    for pair in false_negatives_pairs:
        print(f"  -- {pair}")
    print()
    '''
    print("===Trade Stats===")
    print(f"Live Trades: {live_trades}")
    '''
    print(f"{trades_made} Trades | ${trade_size} Buy-Ins")
    print(f"Total Spent: ${total_spent:.2f}")
    print(f"Buy-in Fees: ${total_buy_in_fees:.2f}")
    print(f"Sell Fees: ${total_sell_fees:.2f}")
    print(f"PNL Dollar: ${total_dollar_gain_loss:.2f}")
    print(f"PNL Percent: {overall_growth_loss_percent:.2f}%")
    print(f"Win Rate: {win_rate:.2f}%")
    print()
    '''


# Model settings
probabiliy_threshold = 0.55  # was 0.75
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
predictions = evaluate_model(
    model, X_for_predictions, y_for_predictions
)

# debug
# Debugging: Print probabilities for live pairs
print("\nProbabilities for Live Pairs:")
for i, pair_address in enumerate(pair_addresses):
    if pair_address in live_pair_addresses:
        logging.info(
            f"Pair Address: {pair_address}, Probability = {predictions[i]:.4f}")
# debug

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
