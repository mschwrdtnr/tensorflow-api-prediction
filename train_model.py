import sys
import os
import pandas as pd
import numpy as np

def train_model():
    # Environment variables
    # MODEL_DIR = os.environ["MODEL_DIR"]
    # MODEL_FILE = os.environ["MODEL_FILE"]
    # MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)

    train_path= "data/train.csv"
    model_path = "ml_model"

    train_data = pd.read_csv(train_path)

if __name__ == '__main__':
    train_model()