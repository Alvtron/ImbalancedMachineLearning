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
        print("")

    @staticmethod
    def print_data_set_probability(dataset):
        distribution = dataset.CLASS.value_counts(normalize=True)
        distribution.columns = ['CLASS', 'Probability']
        print(distribution)

    def __init__(self, data_distribution = [0.2, 0.1, 0.7], sample_size = 0.05, random_state = 42):
        if (sample_size < 0 or sample_size > 1):
            raise ValueError("Sample size is outside the allowed range (0.0 to 1.0).")
        if (len(data_distribution) > 3 or len(data_distribution) < 2):
            raise ValueError("Data distribution must consist of 2 to 3 values.")
        if (sum(data_distribution) != 1.0):
            raise ValueError(f"Data distribution must add up to 1.0. Sum was {sum(data_distribution)}.")

        dataset_file = '../Library/dataset/poker.unordered_%0.4f.csv' % sample_size
        print(f"Importing data set '{dataset_file}'...")
        if (not os.path.isfile(dataset_file)):
            print(f"Could not find file '{dataset_file}'")
            print(f'Attempting to create it with sample size {sample_size}...')
            Poker.save_new_dataset_from_unordered(dataset_file, sample_size)

        dataset = pd.read_csv(dataset_file, header = None, sep = ',', skip_blank_lines = True)
        dataset.columns = self.predictor_labels
        
        train_distr = data_distribution[0]

        if (len(data_distribution) == 2):
            test_distr = data_distribution[1]
        if (len(data_distribution) == 3):
            test_distr = data_distribution[1] + data_distribution[2]
            valid_distr = data_distribution[1] / test_distr
        
        if (len(data_distribution) == 3):
            print('Splitting data into train, validate and test set...')
        else:
            print('Splitting data into train and test set...')

        X = dataset.drop('CLASS', axis=1)
        y = dataset['CLASS']

        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, random_state = random_state, test_size = test_distr)
        if (len(data_distribution) == 3):
            X_test, X_validate, y_test, y_validate = train_test_split(X_test, y_test, stratify = y_test, random_state = random_state, test_size = valid_distr)

        # Print data distribution
        print("Calculating class distribution:")
        if (len(data_distribution) == 2):
            print(f'Class distribution: {data_distribution[0] * 100}% train / {data_distribution[1] * 100}% test')
        if (len(data_distribution) == 3):
            print(f'Class distribution: {data_distribution[0] * 100}% train / {data_distribution[1] * 100}% validate / {data_distribution[2] * 100}% test')

        if (len(data_distribution) == 2):
            class_distribution = pd.concat([y.value_counts(), y_train.value_counts(), y_test.value_counts()], axis = 1, sort = False)
            class_distribution.columns = ['dataset', 'train', 'test']
        if (len(data_distribution) == 3):
            class_distribution = pd.concat([y.value_counts(), y_train.value_counts(), y_validate.value_counts(), y_test.value_counts()], axis = 1, sort = False)
            class_distribution.columns = ['dataset', 'train', 'validate', 'test']
        
        print(class_distribution)

        print('Creating sample weights...')
        sample_weights_per_class = compute_class_weight(class_weight = 'balanced', classes = Poker.class_labels, y = y)
        train_sample_weights = []
        for class_value in y_train:
            train_sample_weights.append(sample_weights_per_class[class_value])
       


        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.train_sample_weights = train_sample_weights
        if (len(data_distribution) == 3):
            self.y_validate = y_validate
            self.X_validate = X_validate