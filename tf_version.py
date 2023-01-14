import tensorflow as tf

class CNN:
    def __init__(self, input_shape, num_filters, filter_size, pool_size, num_classes):
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Conv2D(num_filters, (filter_size, filter_size), input_shape=input_shape, activation='relu'))
        self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(pool_size, pool_size)))
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def fit(self, X, Y, batch_size, epochs):
        self.model.fit(X, Y, batch_size=batch_size, epochs=epochs)

    def predict(self, X):
        return self.model.predict(X)

# Usage example:

# Create a CNN with input shape (32, 32, 3), 8 filters, filter size of 5, pool size of 2, and 10 classes
cnn = CNN((32, 32, 3), 8, 5, 2, 10)

# Generate random input and output data
X = np.random.randn(100, 32, 32, 3)
Y = np.random.randn(100, 10)

# Fit the model to the data
cnn.fit(X, Y, batch_size=32, epochs=10)

# Make predictions
output = cnn.predict(X)