import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight

class Poker(object):
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
        '9: Royal flush; {Ace, King, Queen, Jack, Ten} + flush'
        ]

    def __init__(self, train_size, validation_size = None, random_state = 42):
        print('Importing data set...')
        df_train = pd.read_csv('../Library/dataset/training.txt', header=None, sep=',')
        df_test = pd.read_csv('../Library/dataset/testing.txt', header=None, sep=',')

        dataset = pd.concat([df_train, df_test])
        dataset.columns = self.predictor_labels

        # Inspecting data set and removing duplicates
        print(f"Data set shape: {dataset.shape}")
        dataset_unique = dataset.drop_duplicates()
        print(f"{dataset.shape[0] - dataset_unique.shape[0]} duplicates removed.")
        dataset = dataset_unique
        # Shuffling data set
        print("Shuffling dataset...")
        dataset = shuffle(dataset, random_state = random_state)

        self.X = dataset.drop('CLASS', axis=1)
        self.y = dataset['CLASS']

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, stratify = self.y, random_state = random_state, test_size = 1 - train_size)
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
        sample_weights_per_class = compute_class_weight(class_weight = 'balanced', classes = self.class_labels, y = self.y)
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