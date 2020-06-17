import pandas as pd
import numpy as np
from pathlib import Path
from scipy.sparse import csr_matrix, hstack, load_npz, save_npz

# USER_CNT = 943
# FILM_CNT = 1682


class TDataLoader:

    DATA_SUFFIX = Path("./data")

    def __init__(self, sep="\t"):
        self.sep = sep
        self.raw_train_data_filepath = TDataLoader.DATA_SUFFIX / "train.txt"
        self.raw_test_data_filepath = TDataLoader.DATA_SUFFIX / "test.txt"
        self.prediction_filepath = TDataLoader.DATA_SUFFIX / "predictions.txt"
        self.X_train_filepath = TDataLoader.DATA_SUFFIX / "X_train.npz"
        self.X_test_filepath = TDataLoader.DATA_SUFFIX / "X_test.npz"
        self.y_train_filepath = TDataLoader.DATA_SUFFIX / 'y_train.txt'
        self.y_test_filepath = TDataLoader.DATA_SUFFIX / 'y_test.txt'
        self.users_cnt = 0
        self.films_cnt = 0
        self.user_groups = None
        self.film_groups = None

    def save(self, predictions):
        with self.prediction_filepath.open('w') as out:
            out.write('Id,Score\n')
            for num in range(predictions.size):
                out.write('{},{:.5}\n'.format(num + 1, predictions[num, 0]))

    def load(self, load_preprocessed=False):
        if load_preprocessed:
            X_train = load_npz(self.X_train_filepath)
            X_test = load_npz(self.X_test_filepath)
            y_train = np.loadtxt(self.y_train_filepath).reshape(-1, 1)
            y_test = np.loadtxt(self.y_test_filepath).reshape(-1, 1)
            return X_train, X_test, y_train, y_test

        train_df = pd.read_csv(self.raw_train_data_filepath, sep=self.sep, header=None,
                               names=["user_id", "film_id", "rating"], dtype=int)
        data_columns = ["user_id", "film_id"]
        train_df[data_columns] -= 1

        test_df = pd.read_csv(self.raw_test_data_filepath, sep=self.sep, header=None,
                              names=['user_id', 'film_id'], dtype=int)
        test_df[data_columns] -= 1

        df = pd.concat([train_df[data_columns], test_df])
        self.users_cnt, self.films_cnt = df.max(axis=0) + 1
        self.user_groups = df.groupby('user_id')
        self.film_groups = df.groupby('film_id')

        X_train, y_train = self.process_data(train_df, 'train')
        X_test, y_test = self.process_data(test_df, 'test')
        return X_train, X_test, y_train, y_test

    def process_data(self, df: pd.DataFrame, data_type: str):
        X, y = self.get_base_matrix(df)
        X_other_films = self.get_add_data('user_id', X)
        X_other_users = self.get_add_data('film_id', X)
        X = hstack((X, X_other_films, X_other_users))
        save_npz('data/X_' + data_type + '.npz', X)
        np.savetxt(self.DATA_SUFFIX / ('y_' + data_type + '.txt'), y)
        return X, y

    def get_base_matrix(self, df: pd.DataFrame):
        item_cnt = df.shape[0]

        user_idx = df["user_id"].values
        film_idx = df["film_id"].values
        if 'rating' in df.columns:
            y = df['rating'].values.astype(np.int).reshape(-1, 1)
        else:
            y = np.full((item_cnt, 1), 2.5)

        # Users
        sample_idx = np.arange(item_cnt)
        feature_idx = user_idx
        data_values = np.ones(item_cnt, dtype=int)

        # Films
        sample_idx = np.hstack((sample_idx, np.arange(item_cnt)))
        feature_idx = np.hstack((feature_idx, film_idx + self.users_cnt))
        data_values = np.hstack((data_values, np.ones(item_cnt, dtype=int)))

        # Base matrix
        shape = (item_cnt, self.users_cnt + self.films_cnt)
        return csr_matrix((data_values, (sample_idx, feature_idx)), shape, dtype=np.float), y

    @staticmethod
    def get_data(x):
        return [1 / x, ] * x

    def get_add_data(self, group_label, X):
        if group_label == 'user_id':
            item_cnt = self.users_cnt
            length = self.films_cnt
            offset = 0
            other_label = 'film_id'
            groups = self.user_groups
        else:
            item_cnt = self.films_cnt
            length = self.users_cnt
            offset = self.users_cnt
            other_label = 'user_id'
            groups = self.film_groups

        x, y, data = [], [], []
        group_keys = groups.groups.keys()
        for item_id in range(item_cnt):
            if item_id not in group_keys:
                continue
            item_idx = groups.get_group(item_id)[other_label].to_numpy()
            sample_idx = np.argwhere(X.getcol(offset + item_id).toarray().reshape(-1))
            x += np.tile(item_idx, sample_idx.size).tolist()
            y += np.repeat(sample_idx, item_idx.size).tolist()
            data += self.get_data(item_idx.size) * sample_idx.size
        shape = (X.shape[0], length)
        return csr_matrix((data, (y, x)), shape, dtype=np.float)


def main():
    dataloader = TDataLoader()
    X_train, X_test, y_train, y_test = dataloader.load(load_preprocessed=False)


if __name__ == "__main__":
    main()