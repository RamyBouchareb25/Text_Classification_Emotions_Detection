# Create a sequential model
model = tf.keras.models.Sequential([
    # Embedding layer with a vocabulary size of 10,000, embedding dimension of 16, and input length of 50
    tf.keras.layers.Embedding(10000, 16, input_length=50),

    # Bidirectional LSTM layer with 20 units, returning sequences (for stacking another LSTM layer)
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20, return_sequences=True)),

    # Bidirectional LSTM layer with 20 units (no return_sequences as it's the last LSTM layer)
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20)),

    # Dense output layer with 6 units (for 6 emotion classes) and softmax activation
    tf.keras.layers.Dense(6, activation='softmax')
])


# Compile the model using sparse categorical crossentropy as the loss function,
# Adam optimizer, and accuracy as the metric to monitor during training
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)


# EarlyStopping callback to stop training when the validation accuracy stops improving
early_stop = EarlyStopping(
    monitor='val_accuracy',  # Monitor validation accuracy for early stopping
    mode='min',              # 'min' mode means training will stop when the quantity monitored has stopped decreasing
    verbose=1,               # Verbosity level (1: update messages, 0: silent)
    patience=10               # Number of epochs with no improvement after which training will be stopped
)


# Convert padded sequences and labels for training data to NumPy arrays
padded_train = np.array(padded_train)
train_labels = np.array(train_labels)

# Convert padded sequences and labels for validation data to NumPy arrays
padded_val = np.array(padded_val)
val_labels = np.array(val_labels)



# Train the model on the padded training data and corresponding labels
# Validate on the padded validation data with validation labels
# Use the EarlyStopping callback to stop training if validation accuracy doesn't improve for 10 consecutive epochs
history = model.fit(
    padded_train,             # Padded training sequences
    train_labels,             # Training labels
    validation_data=(padded_val, val_labels),  # Validation data and labels
    epochs=10,                # Number of epochs
    callbacks=[early_stop]    # EarlyStopping callback
)