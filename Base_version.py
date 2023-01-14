import numpy as np

class CNN:
    def __init__(self, input_shape, num_filters, filter_size, pool_size, num_classes):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.pool_size = pool_size
        self.num_classes = num_classes
        self.W1 = np.random.randn(filter_size, filter_size, input_shape[2], num_filters) / np.sqrt(filter_size)
        self.b1 = np.zeros((num_filters,))
        self.W2 = np.random.randn(int(num_filters * input_shape[0] / pool_size), int(num_filters * input_shape[1] / pool_size), num_filters) / np.sqrt(num_filters * input_shape[0] / pool_size)
        self.b2 = np.zeros((num_classes,))

    def forward(self, X):
        self.X = X
        self.conv_out = np.zeros((X.shape[0], int((X.shape[1]-self.filter_size+1)/self.pool_size), int((X.shape[2]-self.filter_size+1)/self.pool_size), self.num_filters))
        for i in range(self.num_filters):
            self.conv_out[:,:,:,i] = np.maximum(0, (np.sum(self.X[:,:,:,np.newaxis] * self.W1[np.newaxis,:,:,i], axis=(1, 2)) + self.b1[i]) / self.pool_size)
        self.fc_out = np.dot(self.conv_out.reshape(X.shape[0], -1), self.W2) + self.b2
        return self.fc_out

    def backward(self, dout, lr):
        dfc_out = dout
        dconv_out = np.dot(dfc_out, self.W2.T).reshape(*self.conv_out.shape)
        dconv_out[self.conv_out <= 0] = 0
        self.W2 -= lr * np.dot(self.conv_out.reshape(self.X.shape[0], -1).T, dfc_out)
        self.b2 -= lr * np.sum(dfc_out, axis=0)
        dX = np.zeros_like(self.X)
        for i in range(self.num_filters):
            dX += np.repeat(np.repeat(dconv_out[:,:,:,i], self.pool_size, axis=1), self.pool_size, axis=2) * self.W1[np.newaxis,:,:,i]
        self.W1 -= lr * np.sum(self.X[:,:,:,np.newaxis] * dX[:,:,:,np.newaxis], axis=0)
        self.b1 -= lr * np.sum(dconv_out, axis=(0, 1, 2))

# Usage example:

# Create a CNN with input shape (32, 32, 3), 8 filters, filter size of 5, pool size of 2, and 10 classes
cnn = CNN((32, 32, 3), 8, 5, 2, 10)

# Generate random input and output data
X = np.random.randn(100, 32, 32, 3)
Y = np.random.randn(100, 10)

# Forward pass
output = cnn.forward(X)

# Compute loss
loss = np.sum((output - Y) ** 2)

# Backward pass
cnn.backward(output - Y, lr=0.01)