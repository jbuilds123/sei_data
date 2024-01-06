import numpy as np
import pickle
import json
import os
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
from keras.models import load_model
import tensorflow as tf


def load_data(file_name):
    with open(file_name, "rb") as f:
        return pickle.load(f)


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


def split_data(sequences, labels, addresses, test_size, random_state):
    return train_test_split(
        sequences, labels, addresses, test_size=test_size, random_state=random_state
    )


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


def evaluate_model(model, X_test, y_test, threshold=0.75):
    y_pred = model.predict(X_test).flatten()
    # Convert probabilities to class labels
    y_pred_class = (y_pred > threshold).astype(int)
    # threshold was 0.50

    test_loss = log_loss(y_test, y_pred)  # Log Loss
    test_auc = roc_auc_score(y_test, y_pred)  # AUC-ROC Score
    precision = precision_score(y_test, y_pred_class)
    recall = recall_score(y_test, y_pred_class)
    f1 = f1_score(y_test, y_pred_class)
    mcc = matthews_corrcoef(y_test, y_pred_class)
    brier = brier_score_loss(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred_class)

    print(f"Log Loss: {test_loss:.2f}")
    print(f"AUC-ROC Score: {test_auc:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"Matthews Correlation Coefficient: {mcc:.2f}")
    print(f"Brier Score: {brier:.2f}")
    print("Confusion Matrix:")
    print(conf_matrix)

    return y_pred


def calculate_performance_metrics(
    predictions,
    pairs_test,
    y_test,
    sequence_lengths,
    pair_addresses,
    end_prices,
    original_close_prices_sequences,
    threshold,
    trade_size,
):
    pair_max_prob, pair_labels, pair_entry_candle, trade_simulations = {}, {}, {}, []
    total_spent = 0
    total_dollar_gain_loss = 0
    total_buy_in_fees = 0
    total_sell_fees = 0
    wins = 0
    buy_in_fee = 0.07
    sell_fee = 0.07
    trades_made = 0

    pair_addresses_list = pair_addresses.tolist()
    false_negatives_pairs = []

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

    for pair_address, (prediction, actual_label) in final_predictions.items():
        entry_candle = pair_entry_candle[pair_address]

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

                total_buy_in_fees += buy_in_fee  # Update buy-in fees here

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
                        # Only increment total_spent by trade_size and buy_in_fee, sell_fee added later
                        total_spent += trade_size + buy_in_fee
                    else:
                        dollar_gain_loss = -(trade_size + buy_in_fee)
                        gain_loss_percent = -100
                        total_spent += trade_size + buy_in_fee

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
        trade_details = {
            "pair": pair_address,
            "entry_candle": entry_candle,
            "entry_price": entry_price,
            "adjusted_exit": adjusted_exit_price,
            "percent_gain": round(gain_loss_percent, 2),
            "dollar_gain": round(dollar_gain_loss, 2),
        }
        print(trade_details)

    print(50 * "-")
    print("===Prediction Stats===")
    print(
        f"Total Predictions Made (including rejections): {total_predictions}")
    print(f"Positive Predictions: {total_pos_pred}")
    print(f"True Positives: {tp} (Rate: {tpr:.2%})")
    print(f"False Positives: {fp} (Rate: {fpr:.2%})")
    print(f"True Negatives: {tn} (Rate: {tnr:.2%})")
    print(f"False Negatives: {fn} (Rate: {fnr:.2%})")
    if fn > 0:
        print("\nFalse Negative Pairs:")
        for pair in false_negatives_pairs:
            print(f"  -- {pair}")
    print()
    print("===Trade Stats===")
    print(f"{trades_made} Trades | ${trade_size} Buy-Ins")
    # Correctly calculate the total spent, adding sell fees if applicable
    total_spent = total_spent + total_sell_fees
    print(f"Total Spent: ${total_spent:.2f}")
    print(f"Buy-in Fees: ${total_buy_in_fees:.2f}")
    print(f"Sell Fees: ${total_sell_fees:.2f}")
    print(f"PNL Dollar: ${total_dollar_gain_loss:.2f}")
    print(f"PNL Percent: {overall_growth_loss_percent:.2f}%")
    print(f"Win Rate: {win_rate:.2f}%")
    print()


# handy usages
probabiliy_threshold = 0.65  # was 0.75
run_on_full_data = True
train_new_model = False
model_version = 2

# Position size
trade_size = 5

# Main execution
(
    padded_sequences,
    labels,
    pair_addresses,
    gain_percentages,
    end_prices,
    original_close_prices_sequences,
    sequence_lengths,
) = load_data("sei_ai/training_data.pkl")


# Data preprocessing
if run_on_full_data:
    X_for_predictions = padded_sequences
    y_for_predictions = labels
    pairs_for_predictions = pair_addresses
else:
    X_train, X_test, y_train, y_test, pairs_train, pairs_test = split_data(
        padded_sequences, labels, pair_addresses, 0.2, 42
    )
    X_for_predictions = X_test
    y_for_predictions = y_test
    pairs_for_predictions = pairs_test

# Model training
if train_new_model:
    X_train_sm, y_train_sm = apply_smote(
        X_train, y_train, padded_sequences.shape)
    class_weights_dict = compute_class_weights(y_train_sm)
    model = create_lstm_model((X_train.shape[1], X_train.shape[2]))
    model = compile_and_train_model(
        model, X_train_sm, y_train_sm, class_weights_dict, save_model_flag=True
    )
else:
    model_path = f"sei_ai/tuned_models/model_v{model_version}.keras"
    # Include custom loss function when loading the model
    model = load_model(model_path, custom_objects={
                       "focal_loss_fixed": focal_loss()})

# Evaluate model
predictions = evaluate_model(model, X_for_predictions, y_for_predictions)

# Calculate performance metrics
calculate_performance_metrics(
    predictions,
    pairs_for_predictions,
    y_for_predictions,
    sequence_lengths,
    pair_addresses,
    end_prices,
    original_close_prices_sequences,
    probabiliy_threshold,
    trade_size,
)
