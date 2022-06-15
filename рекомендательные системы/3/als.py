import cupy as cp
import cupyx.scipy.sparse
import numpy as np

from . import utils


class als_model:
    def __init__(self, scale=0.01):
        self.scale = scale

    def initalize_features(self, num_users, num_items):
        self.user_features = self.create_embeddings(num_users)
        self.item_features = self.create_embeddings(num_items)

    def create_embeddings(self, length, embeddings=2):
        return cp.random.rand(embeddings, length) * 2 - 1

    def train(self, data, user_count, item_count, error_goal):
        train_data = data[~data["valid"]]
        valid_data = data[data["valid"]]
        shape = (user_count, item_count)

        (
            train_row,
            train_column,
            train_data,
            train_sparse,
            train_mask,
        ) = self.get_dataset(train_data, shape)

        (
            valid_row,
            valid_column,
            valid_data,
            valid_sparse,
            valid_mask,
        ) = self.get_dataset(valid_data, shape)

        error = error_goal + 1
        while error > error_goal:
            self.user_features = self.als(train_sparse, train_mask, self.item_features)
            self.item_features = self.als(
                train_sparse.T, train_mask.T, self.user_features
            )
            user_item_indexes = {
                "user_index": valid_row,
                "item_index": valid_column,
            }
            predictions = self.predict(user_item_indexes, as_np=False)
            error = utils.rmse(predictions, valid_data)
            print("RMSE", error)

    def get_dataset(self, data, shape):
        row = cp.asarray(data["user_index"])
        column = cp.asarray(data["item_index"])
        data = cp.asarray(data["overall"])

        sparse_data = cupyx.scipy.sparse.coo_matrix((data, (row, column)), shape=shape)
        mask = cp.asarray([1 for _ in range(len(data))], dtype=np.float32)
        sparse_mask = cupyx.scipy.sparse.coo_matrix((mask, (row, column)), shape=shape)
        return row, column, data, sparse_data, sparse_mask

    def als(self, values, mask, features):
        numerator = values.dot(features.T)
        squared_features = (features ** 2).sum(axis=0)[:, None]
        denominator = self.scale + mask.dot(squared_features)
        return (numerator / denominator).T

    def predict(self, user_item_indexes, as_np=True):
        # This function has been modified to match TensorFlow's predict function input.
        user_indexes = cp.asarray(user_item_indexes["user_index"])
        item_indexes = cp.asarray(user_item_indexes["item_index"])
        user_features = self.user_features[:, user_indexes]
        item_features = self.item_features[:, item_indexes]
        predictions = (user_features * item_features).sum(axis=0)
        if as_np:
            return predictions.get()
        return predictions

    def get_top_n(self, n, user_index, item_indexes=None):
        item_features = self.item_features
        if item_indexes:
            item_features = self.item_features[:, item_indexes]
        predictions = self.user_features[:, user_index].dot(item_features)
        top_indexes = cp.argpartition(predictions, -n)[-n:]
        if item_indexes:
            top_indexes = cp.asarray(item_indexes)[top_indexes]
        return top_indexes.tolist()

    def save(self):
        cp.save("als_user_features.npy", self.user_features)
        cp.save("als_item_features.npy", self.item_features)

    def load(self):
        self.user_features = cp.load("als_user_features.npy")
        self.item_features = cp.load("als_item_features.npy")
