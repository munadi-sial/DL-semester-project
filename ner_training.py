import random
import spacy
from spacy.util import minibatch, compounding
from spacy.training.example import Example
from process_transcripts import get_train_dataset

# Hyperparameters
epochs = 3
dropout = 0.3

# Get complete dataset from file
FILE_PATH = 'transcripts.xlsx'
DATASET = get_train_dataset(FILE_PATH)

# Get training and test examples from shuffled dataset
random.shuffle(DATASET)
TRAIN_DATA = DATASET[0 : int(0.8 * len(DATASET))]
TEST_DATA = DATASET[int(0.8 * len(DATASET)) : len(DATASET)]

# Load SpaCy
nlp = spacy.load('en_core_web_sm')

# Add labels to "ner"
ner = nlp.get_pipe("ner") # get the pipeline component to customize the NER for ad recognition
for _, annotations in TRAIN_DATA:
  for ent in annotations.get("entities"):
    ner.add_label(ent[2])

# Disable unneeded pipeline components
unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']

# Training the model from TRAIN_DATA
with nlp.disable_pipes(*unaffected_pipes):
    for epoch in range(epochs):
        random.shuffle(TRAIN_DATA)
        losses = {}
        #batches = minibatch(TRAIN_DATA, size=compounding(1.0, 4.0, 1.001))
        batches = spacy.util.minibatch(TRAIN_DATA, size=2)

        for batch in batches:
            for text, annotations in batch:
                doc = nlp.make_doc(text) # create example
                example = Example.from_dict(doc, annotations)
                nlp.update([example], losses=losses, drop=dropout) # update the model
                print("Losses: ", losses, ", Epoch: ", epoch)

# Testing the model from TEST_DATA
for text, _ in TEST_DATA:
    doc = nlp(text)
    for ent in doc.ents:
        print(ent.text, ent.start_char, ent.end_char, ent.label_)