import math
import numpy as np
import pandas as pd
from pathlib import Path


class TParametrInitializers:

    DATA_SUFFIX = Path("./data")

    def __init__(self, latent_dimension=5):
        self.sep = "\t"
        self.latent_dimension = latent_dimension
        self.train_dataset_filepath = TParametrInitializers.DATA_SUFFIX / "train.txt"
        self.test_dataset_filepath = TParametrInitializers.DATA_SUFFIX / "test.txt"

        train_df = pd.read_csv(self.train_dataset_filepath, sep=self.sep, header=None,
                               names=["user_id", "film_id", "rating"], dtype=int)
        test_df = pd.read_csv(self.test_dataset_filepath, sep=self.sep, header=None,
                              names=["user_id", "film_id"], dtype=int)

        df = pd.concat([train_df[["user_id", "film_id"]], test_df])
        self.users_cnt, self.films_cnt = df.max(axis=0)
        self.df = train_df

    def xavier(self):

        w0 = self.df['rating'].mean()

        mean_user_rating = np.zeros(self.users_cnt)
        user_groups = self.df[['user_id', 'rating']].groupby('user_id').mean()
        mean_user_rating[user_groups.index - 1] = user_groups.values.ravel()

        mean_film_rating = np.zeros(self.films_cnt)
        film_groups = self.df[['film_id', 'rating']].groupby('film_id').mean()
        mean_film_rating[film_groups.index - 1] = film_groups.values.ravel()

        w = np.hstack([mean_user_rating, mean_film_rating, mean_user_rating,  mean_film_rating])

        threshold = math.sqrt(6) / math.sqrt(w.size)
        V = np.random.uniform(-threshold, threshold, self.latent_dimension * w.size)
        V = V.reshape(w.size, self.latent_dimension)
        return w0 / 5, w / 5, V

    def default(self):
        w0 = 0.0
        w = np.zeros(self.users_cnt * 2 + self.films_cnt * 2, dtype=np.float64)
        V = np.random.normal(loc=0, scale=0.01, size=(w.size, self.latent_dimension))
        return w0, w, V


def main():
    initializer = TParametrInitializers()
    w0, w, V = initializer.xavier()


if __name__ == '__main__':
    main()
