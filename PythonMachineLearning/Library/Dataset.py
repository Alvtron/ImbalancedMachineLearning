import sys
import os.path
import pandas as pd
import time
import random
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.utils.class_weight import compute_class_weight

class Poker:
    population_size = 311875200
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
    def save_new_stratified_sample_from_dataset(file_path, sample_size, random_state = 42):
        chunk_size = Poker.population_size/10
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
    def save_new_distribution_sample_from_dataset(file_path, sample_distribution, random_state = 42):
        chunk_size = Poker.population_size/10
        iterations = 0
        print(f"Reading '../Library/dataset/poker.unordered.csv' in {chunk_size} chunks and saving a distribution sample to '{file_path}'...")
        print("")
        for chunk in pd.read_csv('../Library/dataset/poker.unordered.csv', delimiter = ',', header = None, chunksize = chunk_size):
            chunk.columns = Poker.predictor_labels
            for key, value in sample_distribution.items():
                sample = chunk[chunk['CLASS'] == key].sample(frac = value)
                # Write to file
                if(iterations is 0):
                    sample.to_csv(file_path, mode = 'w', index = False, header = False)
                else:
                    sample.to_csv(file_path, mode = 'a', index = False, header = False)
            # Update console
            iterations = iterations + 1
            sys.stdout.write(f'\r{iterations}/10 chunks read.')
        print("")

    @staticmethod
    def print_data_set_probability(dataset):
        distribution = dataset.CLASS.value_counts(normalize=True)
        distribution.columns = ['CLASS', 'Probability']
        print(distribution)

    @staticmethod
    def print_data_set_statistics(dataset):
        print("First 10 values:")
        print(dataset.head(10))

        print("Descriptive statistics:")
        data_types = pd.DataFrame(dataset.dtypes)
        data_types = data_types.T.rename(index={0: 'dtypes'})
        skewness = pd.DataFrame(dataset.skew())
        skewness = skewness.T.rename(index={0: 'skewness'})
        descriptive_statistics = dataset.describe()
        print(data_types.append([descriptive_statistics, skewness]))

        print("Correlations:") 
        print(dataset.corr(method='pearson'))

    def import_dataset(sample_size = 0.05):
        if (sample_size < 0 or sample_size > 1):
            raise ValueError("Sample size is outside the allowed range (0.0 to 1.0).")

        dataset_file = '../Library/dataset/poker.unordered_%0.4f.csv' % sample_size

        print(f"Importing data set '{dataset_file}'...")
        if (not os.path.isfile(dataset_file)):
            print(f"Could not find file '{dataset_file}'")
            print(f'Attempting to create it with sample size {sample_size}...')
            Poker.save_new_stratified_sample_from_dataset(dataset_file, sample_size)

        dataset = pd.read_csv(dataset_file, header = None, sep = ',', skip_blank_lines = True)
        number_of_samples = len(dataset.index)

        if (number_of_samples == 0 or number_of_samples < Poker.population_size * sample_size * 0.99 or number_of_samples > Poker.population_size * sample_size * 1.01):
            print(f"Wrong dataset size for sample size {sample_size}. Registered size was {number_of_samples} (should be {Poker.population_size * sample_size}).")
            print(f'Attempting to generate new sample with sample size {sample_size}...')
            Poker.save_new_stratified_sample_from_dataset(dataset_file, sample_size)

        dataset = pd.read_csv(dataset_file, header = None, sep = ',', skip_blank_lines = True)
        number_of_samples = len(dataset.index)

        if (number_of_samples == 0 or number_of_samples < Poker.population_size * sample_size * 0.99 or number_of_samples > Poker.population_size * sample_size * 1.01):
            raise Exception(f"Wrong dataset size for sample size {sample_size}. Registered size was {number_of_samples} (should be {Poker.population_size * sample_size}).")

        dataset.columns = Poker.predictor_labels
        return dataset

    def __init__(self, data_distribution = [0.2, 0.1, 0.7], sample_size = 0.05, sampling_strategy = None, verbose = False, random_state = 42):
        if (len(data_distribution) > 3 or len(data_distribution) < 2):
            raise ValueError("Data distribution must consist of 2 to 3 values.")
        if (sum(data_distribution) != 1.0):
            raise ValueError(f"Data distribution must add up to 1.0. Sum was {sum(data_distribution)}.")

        # Import data set
        dataset = Poker.import_dataset(sample_size)

        # Print info about data
        if (verbose):
            Poker.print_data_set_statistics(dataset)

        # Calculate sample distribution
        train_distr = data_distribution[0]

        if (len(data_distribution) == 2):
            test_distr = data_distribution[1]
        if (len(data_distribution) == 3):
            test_distr = data_distribution[1] + data_distribution[2]
            valid_distr = data_distribution[1] / test_distr
        
        if (verbose):
            if (len(data_distribution) == 3):
                print('Splitting data into train, validate and test set...')
            else:
                print('Splitting data into train and test set...')

        # Split data set 
        X = dataset.drop('CLASS', axis=1)
        y = dataset['CLASS']

        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, random_state = random_state, test_size = test_distr)
        if (len(data_distribution) == 3):
            X_test, X_validate, y_test, y_validate = train_test_split(X_test, y_test, stratify = y_test, random_state = random_state, test_size = valid_distr)

        # Resample training data
        if (sampling_strategy is not None):
            if (verbose):
                print('Before OverSampling, number of samples: {}'.format(X_train.shape[0]))
            if(sampling_strategy == "under_sampling"):
                sm = RandomUnderSampler(random_state = random_state, sampling_strategy = 'auto')
            elif(sampling_strategy == "over_sampling"):
                sm = RandomOverSampler(random_state = random_state, sampling_strategy = 'auto')
            else:
                raise Exception("Wrong sampling strategy. Must be 'under_sampling' or 'over_sampling'")
            X_train_res, y_train_res = sm.fit_sample(X = X_train.values, y = y_train.values)
            if (verbose):
                print('After OverSampling, number of samples: {}'.format(X_train_res.shape[0]))
            X_train = pd.DataFrame(data = X_train_res, columns = Poker.feature_labels)
            y_train = pd.Series(data = y_train_res.flatten())

        # Print data distribution
        if (verbose):
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

        # Sample weights
        if (verbose):
            print('Creating sample weights...')

        sample_weights_per_class = compute_class_weight(class_weight = 'balanced', classes = Poker.class_labels, y = y)
        train_sample_weights = []
        for class_value in y_train:
            train_sample_weights.append(sample_weights_per_class[class_value])

        # Initialize
        self.sample_size = sample_size
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.train_sample_weights = train_sample_weights
        self.training_size = data_distribution[0]
        if (len(data_distribution) == 2):
            self.testing_size = data_distribution[1]
        if (len(data_distribution) == 3):
            self.y_validate = y_validate
            self.X_validate = X_validate
            self.validation_size = data_distribution[1]
            self.testing_size = data_distribution[2]