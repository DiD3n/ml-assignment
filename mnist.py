import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train.reshape((60000, 28, 28, 1))
x_test = x_test.reshape((10000, 28, 28, 1))

# check if model already exists
model_cnn = None
try:
    raise ValueError('Force model creation')
    model_cnn = tf.keras.models.load_model('model_cnn.keras')
except:
    model_cnn = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model_cnn.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

    history = model_cnn.fit(x_train, y_train, epochs=5)

    # only in this context we have historic data
    import matplotlib.pyplot as plt

    plt.plot(history.history['loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Over Time')
    plt.show()

test_loss, test_acc = model_cnn.evaluate(x_test, y_test)

print(f"Test accuracy: {test_acc}")


model_cnn.save('model_cnn.keras')

model_cnn.summary()
