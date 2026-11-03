from __future__ import annotations

import inspect

import tensorflow as tf
from tensorflow.keras.applications import (
    EfficientNetB0,
    EfficientNetB1,
    EfficientNetB2,
    EfficientNetB3,
    EfficientNetB4,
)
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import Adam

from .config import (
    BATCH_SIZE,
    DROP_CONNECT_RATE,
    DROPOUT_RATE,
    EPOCHS,
    IMAGE_SIZE,
    LEARNING_RATE,
    LOGS_DIR,
    LR_FACTOR,
    LR_MIN_DELTA,
    LR_PATIENCE,
    MODELS_DIR,
    N_CLASSES,
    VALIDATION_SPLIT,
)

BACKBONE_FN = {
    "B0": EfficientNetB0,
    "B1": EfficientNetB1,
    "B2": EfficientNetB2,
    "B3": EfficientNetB3,
    "B4": EfficientNetB4,
}


def freeze_batch_norm(model: tf.keras.Model) -> None:
    """Paper §5.1: keep BN frozen during fine-tuning."""
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False


def build_efficientnet(
    variant: str = "B2",
    image_size: int = IMAGE_SIZE,
    drop_connect_rate: float = DROP_CONNECT_RATE,
    freeze_bn: bool = False,
    weights: str | None = "imagenet",
) -> tf.keras.Model:
    """ImageNet EfficientNet + GAP + Dropout(0.5) + Dense(3, softmax)."""
    key = variant.upper().replace("EFFICIENTNET", "")
    if key not in BACKBONE_FN:
        raise ValueError(f"Unsupported EfficientNet variant: {variant}")

    constructor = BACKBONE_FN[key]
    base_kwargs = {
        "weights": weights,
        "include_top": False,
        "input_shape": (image_size, image_size, 3),
    }
    if "drop_connect_rate" in inspect.signature(constructor).parameters:
        base_kwargs["drop_connect_rate"] = drop_connect_rate
    base = constructor(**base_kwargs)
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.Dropout(rate=DROPOUT_RATE)(x)
    outputs = tf.keras.layers.Dense(N_CLASSES, activation="softmax")(x)
    model = tf.keras.Model(inputs=base.input, outputs=outputs, name=f"EfficientNet{key}_tumor")
    if freeze_bn:
        freeze_batch_norm(model)
    return model


def compile_model(model: tf.keras.Model, learning_rate: float = LEARNING_RATE) -> tf.keras.Model:
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=learning_rate),
        metrics=["accuracy"],
    )
    return model


def make_callbacks(variant: str) -> list:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = MODELS_DIR / f"efficientnet{variant.lower()}.h5"
    return [
        TensorBoard(log_dir=str(LOGS_DIR / variant.lower())),
        ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            save_best_only=True,
            mode="auto",
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_accuracy",
            factor=LR_FACTOR,
            patience=LR_PATIENCE,
            min_delta=LR_MIN_DELTA,
            mode="auto",
            verbose=1,
        ),
    ]


def train_model(
    model: tf.keras.Model,
    x_train,
    y_train,
    variant: str,
    epochs: int = EPOCHS,
    initial_epoch: int = 0,
    batch_size: int = BATCH_SIZE,
    validation_split: float = VALIDATION_SPLIT,
):
    callbacks = make_callbacks(variant)
    history = model.fit(
        x_train,
        y_train,
        validation_split=validation_split,
        initial_epoch=initial_epoch,
        epochs=epochs,
        verbose=1,
        batch_size=batch_size,
        callbacks=callbacks,
    )
    return history, MODELS_DIR / f"efficientnet{variant.lower()}.h5"
