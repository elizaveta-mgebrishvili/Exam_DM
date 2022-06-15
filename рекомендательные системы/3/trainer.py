import time
import pandas as pd
import tensorflow as tf
from . import als, dataset, wide_and_deep, utils
import cupy as cp
import cudf
from tensorflow.keras.layers import Input
import numpy as np


def get_als_model(data):
    # If using an ALS model, replace the FIXMEs below.
    user_count = data["user_index"].max() + 1
    item_count = data["item_index"].max() + 1
    rmse_goal = 1.25
    scale = 0.01
    als_model = als.als_model(scale=scale)
    als_model.initalize_features(user_count, item_count)
    als_model.train(data, user_count, item_count, rmse_goal)
    return als_model


def get_wide_and_deep_model(data):
    # If using a Wide and Deep model, replace the FIXME's below.
    # TODO: update the below input tensor to match the incomming data.
    # A few example Inputs have been created to help get started.
    input_tensor = {
        "user_index": Input(name="user_index", shape=(1,), dtype="int32"),
        "item_index": Input(name="item_index", shape=(1,), dtype="int32"),
        "price_filled": Input(name="price_filled", shape=(1,), dtype="float32"),
        "salesRank_Electronics": Input(name="salesRank_Electronics", shape=(1,), dtype="float32"),
        "brand_index": Input(name="brand_index", shape=(1,), dtype="int32"),
        "category_0_2_index": Input(name="category_0_2_index", shape=(1,), dtype="int32"),
        "category_1_2_index": Input(name="category_1_2_index", shape=(1,), dtype="int32"),
    }

    # TODO: define the columns below. A few have been placed to help get started.
    one_hot_columns = ["user_index", "item_index", "brand_index"]
    numerical_columns = ["price_filled", "salesRank_Electronics"]
    multi_hot_columns = ["category_0_2_index", "category_1_2_index"]
    crossed_columns = [("user_index", "brand_index")]
    return wide_and_deep.get_wide_and_deep(
        data,
        one_hot_columns,
        numerical_columns,
        multi_hot_columns,
        crossed_columns,
        input_tensor,
    )


def get_candidate_generator():
    ratings = pd.read_csv("data/task_3_ratings.csv")
    # TODO: Choose either get_als_model or get_wide_and_deep_model and pass in ratings.
    model = get_als_model(ratings)
    return model


def get_candidate_scoring_model():
    metadata = cudf.read_csv("data/task_3_metadata.csv")
    # Add data transformation here
    utils.apply_gauss_rank(metadata, "salesRank_Electronics")

    lab_2_data = cudf.read_csv("data/task_2_wide_and_deep.csv")
    lab_2_data = lab_2_data.drop("salesRank_Electronics", axis=1)
    lab_2_data = lab_2_data.merge(
        metadata[["salesRank_Electronics"]],
        how="left",
        left_on="item_index",
        right_index=True,
    )

    # TODO: Choose either get_als_model or get_wide_and_deep_model and pass in lab_2_data.
    return get_wide_and_deep_model(lab_2_data)


if __name__ == "__main__":
    # These are tests for the above code.
    # We use a copy of these for assessment, so please don't change.
    cg_time = time.time()
    cg_model = get_candidate_generator()
    elapsed = time.time() - cg_time
    print("Candidate Generation Training Time", elapsed)
    assert elapsed < 15

    ratings = pd.read_csv("data/task_3_ratings.csv")
    validation_data = ratings[ratings["valid"]]
    user_item_indexes = {
        "user_index": np.asarray(validation_data["user_index"]),
        "item_index": np.asarray(validation_data["item_index"]),
    }
    cg_time = time.time()
    cg_predictions = cg_model.predict(user_item_indexes, as_np=False)
    elapsed = time.time() - cg_time
    print("Candidate Generation Prediction Time", elapsed)
    assert elapsed < 0.005
    cg_rmse = utils.rmse(cg_predictions, cp.asarray(validation_data["overall"]))
    print("Candidate Generation RMSE", cg_rmse)
    assert cg_rmse < 1.3
    cg_model.save()

    ratings = pd.read_csv("data/task_2_wide_and_deep.csv")
    validation_data = ratings[ratings["valid"]]
    cs_time = time.time()
    cs_model = get_candidate_scoring_model()
    elapsed = time.time() - cs_time
    print("Candidate Scorer Training Time", elapsed)
    assert elapsed < 600

    cs_time = time.time()
    cs_predictions = cs_model.predict(
        {
            "user_index": np.asarray(validation_data["user_index"]),
            "item_index": np.asarray(validation_data["item_index"]),
            "brand_index": np.asarray(validation_data["brand_index"]),
            "price_filled": np.asarray(validation_data["price_filled"])[:, None],
            "salesRank_Electronics": np.asarray(
                validation_data["salesRank_Electronics"]
            )[:, None],
            "category_0_2_index": np.asarray(validation_data["category_0_2_index"]),
            "category_1_2_index": np.asarray(validation_data["category_1_2_index"]),
        }
    ).flatten()
    elapsed = time.time() - cs_time
    print("Candidate Scorer Prediction Time", elapsed)
    assert elapsed < 30
    cs_rmse = utils.rmse(cs_predictions, np.asarray(validation_data["overall"]))
    print("Candidate Scorer RMSE", cs_rmse)
    assert cs_rmse < 1.175
    cs_model.save("candidate_scorer", overwrite=True)

    print("Tests pass!")
