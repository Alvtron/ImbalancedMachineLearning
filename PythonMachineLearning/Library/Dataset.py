import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight

def create_poker_dataset():
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

    print('Importing data set...')
    df_train = pd.read_csv('../Library/dataset/training.txt', header=None, sep=',')
    df_test = pd.read_csv('../Library/dataset/testing.txt', header=None, sep=',')

    dataset = pd.concat([df_train, df_test])
    dataset.columns = predictor_labels

    # Inspecting data set and removing duplicates
    print(f"Data set shape: {dataset.shape}")
    dataset_unique = dataset.drop_duplicates()
    print(f"{dataset.shape[0] - dataset_unique.shape[0]} duplicates removed.")
    dataset = dataset_unique
    # Shuffling data set
    print("Shuffling dataset...")
    dataset = shuffle(dataset, random_state = 42)

    # Creating train set, validation set and test set... 
    print('Creating train set, validation set and test set...')
    y = dataset['CLASS']
    X = dataset.drop('CLASS', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, random_state = 42, test_size = 0.2)
    X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, stratify = y_train, random_state = 22, test_size = 0.2)

    print("Data set class distribution:")
    class_distribution = pd.concat([y.value_counts(), y_train.value_counts(), y_validate.value_counts(), y_test.value_counts()], axis=1, sort=False)
    class_distribution.columns = ['dataset', 'train', 'validate', 'test']
    print(class_distribution)

    print('Creating sample weights...')
    sample_weights_per_class = compute_class_weight(class_weight = 'balanced', classes = class_labels, y = y)
    train_sample_weights = []
    for class_value in y_train:
        train_sample_weights.append(sample_weights_per_class[class_value])

    return predictor_labels, feature_labels, class_labels, class_descriptions, X_train, X_validate, X_test, y_train, y_validate, y_test, train_sample_weights