import argparse
import time

import numpy as np
import pandas as pd
import cudf
import cupy as cp
import sys
import tensorflow as tf
import tritonhttpclient

from . import als, wide_and_deep, utils


CANDIDATE_SCORER = "candidate_scorer"
BATCH_SIZE = 16

try:
    triton_client = tritonhttpclient.InferenceServerClient(
        url="triton:8000", verbose=True
    )
    print("client created.")
except Exception as e:
    print("channel creation failed: " + str(e))

# TODO: Load Candidate Generator. Choose one of the algorithms below to uncomment.
# Uncomment to load ALS model as Candidate Generator
candidate_generator = als.als_model()
candidate_generator.load()

# Uncomment to load wide_and_deep_model as Candidate Generator
#candidate_generator = tf.keras.models.load_model('FIXME_MODEL_PATH')

# Load metadata and transform
metadata = cudf.read_csv("data/task_3_metadata.csv")
utils.apply_gauss_rank(metadata, "salesRank_Electronics")
utils.log_norm(metadata, "price_filled")

def user_prediction(user_index, score_n=4):
    # Get Candidates
    candidates = candidate_generator.get_top_n(BATCH_SIZE, user_index)
    candidate_data = metadata.iloc[candidates]
    candidates = np.array(candidates, dtype=np.int64)
    
    # TODO: Add columns specific to your Triton model.
    # A few examples are placed to get started.
    columns = [
        ('user_index', "INT32"),
        ('item_index', "INT32"),
        ('brand_index', "INT32"),
        ('price_filled', "FP32"),
        ('salesRank_Electronics', "FP32"),
        ('category_0_2_index', "INT32"),
        ('category_1_2_index', "INT32"),
    ]

    dtypes = {
        "INT32": np.int32,
        "FP32": np.float32
    }

    # TODO: Fix the below for loops to format the data for Triton
    inputs = []
    for column in columns:
        name = column[0]
        dtype = column[1]
        inputs.append(tritonhttpclient.InferInput(name, [BATCH_SIZE, 1], dtype))
        
    inputs[0].set_data_from_numpy(np.array([[user_index]] * BATCH_SIZE, dtype=np.int32))
    inputs[1].set_data_from_numpy(candidates[:, None].astype(np.int32))
    
    for i in range(2, len(columns)):
        name = columns[i][0]
        dtype = dtypes[columns[i][1]]
        data = np.expand_dims(candidate_data[name].to_array().astype(dtype), axis=-1)
        inputs[i].set_data_from_numpy(data)
    
    outputs = [
        tritonhttpclient.InferRequestedOutput("tf.__operators__.add", binary_data=False)
    ]
    results = triton_client.infer(CANDIDATE_SCORER, inputs, outputs=outputs)
    results = np.array(results.get_response()["outputs"][0]["data"])

    # Rank results
    top_indexes = np.argpartition(results, -score_n)[-score_n:]
    return candidates[top_indexes], results[top_indexes]

if __name__ == "__main__":
    # These are tests for the above code.
    # We use a copy of these for assessment, so please don't change.
    ranking_time = time.time()
    indexes, predictions = user_prediction(131676)  # Prediction for user 131676
    elapsed = time.time() - ranking_time
    print("Ranking Time", elapsed)
    assert elapsed < 2
    print("Top Indexes", indexes)
    print("Top Predictions", predictions)
    
    avg_prediction = predictions.mean()
    print("Average Rating", avg_prediction)
    assert avg_prediction > 4.2
    assert len(indexes) == 4
    
    print("Tests pass!")

