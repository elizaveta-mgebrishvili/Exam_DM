import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, concatenate, Dense, DenseFeatures
from nvtabular.loader.tensorflow import KerasSequenceValidater

from . import dataset


def get_dense(
    data,
    embedding_inputs={},
    concatenate_layers=[],
    emb_size=32,
    hidden_layers=[256, 128],
):
    # Make embeddings
    embeddings = []
    for col in embedding_inputs:
        max_index = int(data[col].max() + 1)
        embedding_input = embedding_inputs[col]
        embedding_layer = Embedding(max_index, emb_size, name="emb_" + col)
        embeddings.append(tf.squeeze(embedding_layer(embedding_input), axis=1))

    # The current layer at the end of our function chain
    end_layer = concatenate(concatenate_layers + embeddings, axis=1)
    for i, units in enumerate(hidden_layers):
        end_layer = Dense(units, activation="relu", name="dnn_{}".format(i))(end_layer)
    return Dense(1, activation=None, name="pred")(end_layer)


def get_deep(
    data,
    input_tensor,
    one_hot_columns,
    numerical_columns,
    multi_hot_columns,
    embedding_size=32,
):
    # Get One-hot embeddings
    embedding_inputs = {key: input_tensor[key] for key in one_hot_columns}

    # Get numerical layers to concatenate with embeddings
    numerical_layers = [input_tensor[col] for col in numerical_columns]

    # Get multihot embeddings
    multihot_emb_avg = None
    if multi_hot_columns:
        doc_max = max(data[multi_hot_columns].to_pandas().max())
        multihot_emb = Embedding(int(doc_max + 1), embedding_size)
        multihot_emb_ids = [multihot_emb(input_tensor[col]) for col in multi_hot_columns]
        multihot_emb_stack = tf.keras.backend.stack(multihot_emb_ids, axis=2)
        multihot_emb_avg = tf.squeeze(tf.keras.backend.mean(multihot_emb_stack, axis=2), axis=1)

    concatenate_layers = numerical_layers + [multihot_emb_avg]
    output_tensor_deep = get_dense(
        data, embedding_inputs=embedding_inputs, concatenate_layers=concatenate_layers
    )
    return tf.keras.Model(input_tensor, output_tensor_deep), output_tensor_deep


def get_wide(
    data, input_tensor, one_hot_columns, crossed_columns, embedding_size=128,
):
    cat_identity = {}
    for col in one_hot_columns:
        number_of_buckets = int(data[col].max() + 1)
        cat_identity[col] = tf.feature_column.categorical_column_with_identity(
            col, number_of_buckets)

    cat_indicator = {
        col: tf.feature_column.indicator_column(cat_identity[col])
        for col in cat_identity
    }

    cat_cross = {}
    for col1, col2 in crossed_columns:
        key = col1 + "_" + col2
        crossed_column = tf.feature_column.crossed_column(
            [cat_identity[col1], cat_identity[col2]], embedding_size
        )
        cat_cross[key] = tf.feature_column.indicator_column(crossed_column)

    cat_indicator.update(cat_cross)
    dense_features = cat_indicator.values()
    dense_features = DenseFeatures(dense_features, name="deep_inputs")(input_tensor)
    return Dense(1, activation=None, name="output_wide")(dense_features)


def get_wide_and_deep(
    data,
    one_hot_columns,
    numerical_columns,
    multi_hot_columns,
    crossed_columns,
    input_tensor,
):
    cat_coloumns = one_hot_columns + multi_hot_columns
    deep_model, output_tensor_deep = get_deep(
        data, input_tensor, one_hot_columns, numerical_columns, multi_hot_columns
    )

    output_tensor_wide = get_wide(data, input_tensor, one_hot_columns, crossed_columns)
    output_tensor = output_tensor_deep + output_tensor_wide

    train_ds, valid_ds = dataset.get_test_and_train(data, cat_cols=cat_coloumns, cont_cols=numerical_columns)
    model = tf.keras.Model(input_tensor, output_tensor)

    metrics = [tf.keras.metrics.RootMeanSquaredError(name="rmse")]
    model.compile(optimizer="adam", loss="mse", metrics=metrics)
    model.fit(train_ds, callbacks=[KerasSequenceValidater(valid_ds)], epochs=2)
    return model


def get_top_n(model, n, user_index, item_indexes, metadata):
    batch_size = len(item_indexes)
    item_indexes_np = np.asarray(item_indexes)
    predictions = model.predict({
        "user_index": np.asarray([user_index]*batch_size),
        "item_index": item_indexes_np,
        "brand_index": np.asarray(metadata["brand_index"]),
        "price_filled": np.asarray(metadata["price_filled"])[:, None],
        "salesRank_Electronics": np.asarray(metadata["salesRank_Electronics"])[:, None],
        "category_0_2_index": np.asarray(metadata["category_0_2_index"]),
        "category_1_2_index": np.asarray(metadata["category_1_2_index"])
    })
    top_indexes = np.argpartition(predictions, -n)[-n:]
    return item_indexes_np[top_indexes]
