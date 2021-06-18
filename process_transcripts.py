# Imports
import numpy as np
import pandas as pd

# Process and Return Dataset Function
def get_train_dataset(FILE_PATH):
    # Import Dataset
    df = pd.read_excel(FILE_PATH)

    # Training Dataset
    data = []

    # Create Data
    for i in range(len(df)):
        # Get Row
        row = df.iloc[i]
        # Get Advertiser and Ad Copy
        advertiser = row['Advertiser'].lower()
        copy = row['Ad_copy']
        # Find Sentences with Advertiser
        sentences = copy.split('.')
        for sentence in sentences:
            # Format sentence
            if len(sentence) < 2: continue
            if not sentence[0].isalpha(): sentence = sentence[1:]
            # Create Training Examlple
            idx = sentence.lower().find(advertiser)
            if idx != -1:
                X = (sentence)
                Y = {'entities': [(idx, idx + len(advertiser), 'BRAND')]}
                data.append((X, Y))
    # Return Dataset
    return data