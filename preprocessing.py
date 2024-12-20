import json
import pandas as pd
from sklearn.utils import resample

class Preprocessing:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        with open(self.file_path, "r") as file:
            data = json.load(file)
        return data
    
    def prepare_dataset(self, data) -> pd.DataFrame:
        rows = []
        for entry in data:
            transmitter_frequency = entry["transmitter"]["frequency"]
            label = entry["label"]
            receiver_features = []
            
            # Flatten all receiver data into a single list
            for receiver in entry["receivers"]:
                receiver_features.extend(receiver["position"])
                receiver_features.extend(receiver["velocity"])

            row = [transmitter_frequency] + receiver_features + [label]
            rows.append(row)
        
        num_receivers = max(len(entry["receivers"]) for entry in data)
        columns = ["trans_freq"]
        for i in range(1, num_receivers + 1):
            columns += [
                f"receiver_{i}_pos_x",
                f"receiver_{i}_pos_y",
                f"receiver_{i}_vel_x",
                f"receiver_{i}_vel_y",
            ]
        columns.append("label")

        df = pd.DataFrame(rows, columns=columns)
        df = df.fillna(0)

        return df

    def balance_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        # Separate majority and minority classes
        df_majority = df[df['label'] == 1]
        df_minority = df[df['label'] == 0]

        # Undersample the majority class
        df_majority_undersampled = resample(
            df_majority,
            replace=False,     # Sample without replacement
            n_samples=len(df_minority),  # Match minority class size
            random_state=42    # For reproducibility
        )

        # Combine the undersampled majority class with the minority class
        balanced_df = pd.concat([df_minority, df_majority_undersampled])

        return balanced_df