import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import torch
from transformers import TrainingArguments, Trainer, BertTokenizer, BertForSequenceClassification
import os 
from sklearn.preprocessing import LabelEncoder


class ToxicCommentClassifier:
    def __init__(self, model_name='bert-base-uncased', num_labels=3, device='mps'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(device)
        self.device = device

    def tokenize_data(self, data, max_length=512):
        return self.tokenizer(data, padding=True, truncation=True, max_length=max_length)

    def create_dataset(self, X, y):
        encodings = self.tokenize_data(X)
        return Dataset(encodings, y)

    def train_model(self, train_dataset, val_dataset, output_dir="output", epochs=5, batch_size=32):
        args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics
        )

        trainer.train()
        trainer.evaluate()
        return trainer

    @staticmethod
    def compute_metrics(p):
        pred, labels = p
        pred = np.argmax(pred, axis=1)

        accuracy = accuracy_score(y_true=labels, y_pred=pred)
        recall = recall_score(y_true=labels, y_pred=pred)
        precision = precision_score(y_true=labels, y_pred=pred)
        f1 = f1_score(y_true=labels, y_pred=pred)

        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

    def predict(self, text):
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(self.device)
        outputs = self.model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predictions = predictions.cpu().detach().numpy()
        return predictions

    def save_model(self, path):
        self.model.save_pretrained(path)

    def load_model(self, path):
        self.model = BertForSequenceClassification.from_pretrained(path).to(self.device)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

def load_data(file_path, sample_size=1000):
    data = pd.read_csv(file_path, error_bad_lines=False, engine="python")
    data = data[['comment_text', 'toxic']]
    return data.sample(n=sample_size)

def split_data(data, test_size=0.2, stratify=True):
    X = list(data["description"])
    y = list(data["label"])
    return train_test_split(X, y, test_size=test_size, stratify=y if stratify else None)

def merge_csv_files(directory_path, num_files=10):
    # Get the list of CSV files in the directory
    csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]
    
    # Sort the files to ensure consistency
    csv_files.sort()
    
    # Limit to the first `num_files` files
    csv_files = csv_files[:num_files]
    
    # Initialize an empty DataFrame
    merged_df = pd.DataFrame()

    # Loop through the files and append them
    for file in csv_files:
        file_path = os.path.join(directory_path, file)
        df = pd.read_csv(file_path)
        df = df[['description', 'label']]

        
        # Append to the merged DataFrame
        merged_df = pd.concat([merged_df, df], ignore_index=True)
    
    # Save the merged DataFrame to a new CSV file
    return merged_df

def convert_labels_to_numeric(df, column_name):
    # Initialize the label encoder
    label_encoder = LabelEncoder()

    # Fit and transform the labels
    df[column_name] = label_encoder.fit_transform(df[column_name])

    # Optionally, return the mapping of the original labels to the new numeric values
    label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    
    return df, label_mapping

# Example Usage
if __name__ == "__main__":
    # Load and prepare data
    data_dir_name = '/Users/krishnayadav/Documents/aiTradingBot/bert_finetune/'
    merged_df = merge_csv_files(data_dir_name)
    merged_df, label_mapping = convert_labels_to_numeric(merged_df, 'label')
    data = merged_df
    # file_path = "/content/drive/MyDrive/Youtube Tutorials/datasets/toxic_commnets.csv"
    # data = load_data(file_path)
    
    X_train, X_val, y_train, y_val = split_data(data)

    # Initialize classifier
    classifier = ToxicCommentClassifier()

    # Create datasets
    train_dataset = classifier.create_dataset(X_train, y_train)
    val_dataset = classifier.create_dataset(X_val, y_val)

    # Train model
    trainer = classifier.train_model(train_dataset, val_dataset)

    # Save and reload model
    # classifier.save_model('CustomModel')
    # classifier.load_model('CustomModel')

    # # Predict sample texts
    # predictions = classifier.predict("go to hell")
    # print(predictions)

