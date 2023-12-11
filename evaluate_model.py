# Convert padded sequences and labels for test data to NumPy arrays
padded_test = np.array(padded_test)
test_labels = np.array(test_labels)

# Evaluate the trained model on the padded test data and corresponding labels
test_loss, test_accuracy = model.evaluate(padded_test, test_labels)

# Print the test accuracy
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')


# Function to visualize training and validation history
def show_history(history):
    # Get the number of epochs trained
    epochs_trained = len(history.history['loss'])

    # Set up the figure with two subplots (Accuracy and Loss)
    plt.figure(figsize=(8, 4))

    # Plot Training and Validation Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(range(0, epochs_trained), history.history.get('accuracy'), label='Training')
    plt.plot(range(0, epochs_trained), history.history.get('val_accuracy'), label='Validation')
    plt.ylim([0., 1.])
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot Training and Validation Loss
    plt.subplot(1, 2, 2)
    plt.plot(range(0, epochs_trained), history.history.get('loss'), label='Training')
    plt.plot(range(0, epochs_trained), history.history.get('val_loss'), label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Show the plot
    plt.show()
    
show_history(history)

preds = model.predict(np.expand_dims(padded_test[146],axis=0))[0]
index = np.argmax(preds)
print(index_to_class[index])