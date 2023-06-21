import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from tensorflow import keras
from tensorflow.keras import layers


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res


def build_model(
        input_shape,
        head_size,
        num_heads,
        ff_dim,
        num_transformer_blocks,
        mlp_units,
        dropout=0,
        mlp_dropout=0,
        n_classes=2
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)


def fit_model(y_train, x_train):

    input_shape = x_train.shape[1:]

    model = build_model(
        input_shape,
        head_size=256,
        num_heads=4,
        ff_dim=4,
        num_transformer_blocks=4,
        mlp_units=[128],
        mlp_dropout=0.4,
        dropout=0.25,
    )

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        metrics=["sparse_categorical_accuracy"],
    )

    callbacks = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]

    model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        epochs=200,
        batch_size=64,
        callbacks=callbacks,
    )
    return model


def prepare_train_data(df, y, predictors, standardize=False):
    # sub set to points available as training
    y_train = df[y][~df[y].isna()]

    # check if training data only contains 2 different values
    if len(np.unique(y_train)) != 2:
        raise ValueError('Target variable y can only have 2 values')

    # subset dataframe columns ot only those which are relevant for the classifier
    x_df = df[predictors].copy() if predictors else df.copy()
    # impute nan,s infinte etc with mean value within each column
    imp = SimpleImputer(strategy="mean")
    x_df = imp.fit_transform(x_df)

    # standardize
    scaler = StandardScaler().fit(x_df) if standardize else None
    x_df = scaler.transform(x_df) if scaler else x_df

    # subset predictor variables to points where training data is available
    x_train = x_df[~df[y].isna()]
    x_train = np.expand_dims(x_train, axis=2)

    y_train[y_train == 0.] = 0
    y_train[y_train == 1.] = 1
    y_train = y_train.values

    return y_train, x_train, scaler


def prepare_data2(df, predictors, scaler):

    # subset dataframe columns ot only those which are relevant for the classifier
    x_df = df[predictors].copy() if predictors else df.copy()
    # impute nan,s infinte etc with mean value within each column
    imp = SimpleImputer(strategy="mean")
    x_df = imp.fit_transform(x_df)

    x_df = scaler.transform(x_df) if scaler else x_df
    x_train = np.expand_dims(x_df, axis=2)

    return x_train