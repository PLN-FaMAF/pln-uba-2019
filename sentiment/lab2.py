import os

from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split


def load_datasets_unlabeled_test():
    dataset = load_files('review_polarity_competition/reviews_sentoken', shuffle=False)
    docs_train, docs_dev, y_train, y_dev = train_test_split(
        dataset.data, dataset.target, test_size=0.10, random_state=42)
    dirname = "review_polarity_competition/test_reviews_sentoken"
    test = []
    # I do this to keep the files in numeric order
    for fname in range(len(os.listdir(dirname))):
        fname = str(fname) + ".txt"
        with open(os.path.join(dirname, fname)) as fd:
            test.append(fd.read())
    train = docs_train, y_train
    dev = docs_dev, y_dev
    return train, dev, test


def save_results(fname, labels):
    with open(fname, 'w') as f:
        f.write("Id,Category\n")
        for i,l in enumerate(labels):
            f.write(str(i) + ".txt," + str(l) + "\n")
