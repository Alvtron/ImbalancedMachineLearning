import sys
import os.path
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight

class Poker:
    size = 311875200
    size_ordered = 2598960
    predictor_labels = ['S1','C1','S2','C2','S3','C3','S4','C4','S5','C5','CLASS']
    feature_labels = ['S1','C1','S2','C2','S3','C3','S4','C4','S5','C5']
    class_labels = [0,1,2,3,4,5,6,7,8,9]
    class_descriptions = [
        '0: Nothing in hand; not a recognized poker hand',
        '1: One pair; one pair of equal ranks within five cards',
        '2: Two pairs; two pairs of equal ranks within five cards',
        '3: Three of a kind; three equal ranks within five cards',
        '4: Straight; five cards, sequentially ranked with no gaps',
        '5: Flush; five cards with the same suit',
        '6: Full house; pair + different rank three of a kind',
        '7: Four of a kind; four equal ranks within five cards',
        '8: Straight flush; straight + flush',
        '9: Royal flush; {Ace, King, Queen, Jack, Ten} + flush']
    class_probabilities = {
        0:0.50117739,
        1:0.42256903,
        2:0.04753902,
        3:0.02112845,
        4:0.00392464,
        5:0.0019654,
        6:0.00144058,
        7:0.0002401,
        8:0.00001385,
        9:0.00000154}

    @staticmethod
    def save_new_dataset_from_unordered(file_path, sample_size, random_state = 42):
        chunk_size = Poker.size/10
        iterations = 0
        print(f"Reading '../Library/dataset/poker.unordered.csv' in {chunk_size} chunks and saving a {sample_size} stratified sample to '{file_path}'...")
        print("")
        for chunk in pd.read_csv('../Library/dataset/poker.unordered.csv', delimiter = ',', header = None, chunksize = chunk_size):
            chunk.columns = Poker.predictor_labels
            # Define X and y
            X = chunk.drop('CLASS', axis=1)
            y = chunk['CLASS']
            # Stratified sample
            X_throw, X_keep, y_throw, y_keep = train_test_split(X, y, stratify = y, random_state = random_state, test_size = sample_size)
            chunk = pd.concat([X_keep, y_keep], axis=1, sort=None)
            # Write to file
            if(iterations is 0):
                chunk.to_csv(file_path, mode = 'w', index = False, header = False)
            else:
                chunk.to_csv(file_path, mode = 'a', index = False, header = False)
            # Update console
            iterations = iterations + 1
            sys.stdout.write(f'\r{iterations}/10 chunks read.')

    @staticmethod
    def print_data_set_probability(dataset):
        distribution = dataset.CLASS.value_counts(normalize=True)
        distribution.columns = ['CLASS', 'Probability']
        print(distribution)

    def __init__(self, test_size, validation_size = None, random_state = 42):
        print('Importing data set...')
        dataset_file = '../Library/dataset/poker.unordered_01.csv'
        if (not os.path.isfile(dataset_file)):
            Poker.save_new_dataset_from_unordered(dataset_file, 0.01)

        dataset = pd.read_csv(dataset_file, header = None, sep = ',', skip_blank_lines = True)
        dataset.columns = self.predictor_labels
        Poker.print_data_set_probability(dataset)

        self.X = dataset.drop('CLASS', axis=1)
        self.y = dataset['CLASS']

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, stratify = self.y, random_state = random_state, test_size = test_size)
        if (validation_size is not None):
            X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, stratify = y_train, random_state = random_state, test_size = validation_size)

        print("Data set class distribution:")
        if (validation_size is None):
            class_distribution = pd.concat([self.y.value_counts(), y_train.value_counts(), y_test.value_counts()], axis = 1, sort = False)
            class_distribution.columns = ['dataset', 'train', 'test']
        else:
            class_distribution = pd.concat([self.y.value_counts(), y_train.value_counts(), y_validate.value_counts(), y_test.value_counts()], axis = 1, sort = False)
            class_distribution.columns = ['dataset', 'train', 'validate', 'test']
        
        print(class_distribution)

        print('Creating sample weights...')
        sample_weights_per_class = compute_class_weight(class_weight = 'balanced', classes = Poker.class_labels, y = self.y)
        train_sample_weights = []
        for class_value in y_train:
            train_sample_weights.append(sample_weights_per_class[class_value])

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.train_sample_weights = train_sample_weights
        if (validation_size is not None):
            self.y_validate = y_validate
            self.X_validate = X_validate