import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)
x_train = x_train.reshape(
    x_train.shape[0], x_train.shape[1], x_train.shape[2], 1
)
x_test = x_test.reshape(
    x_test.shape[0], x_test.shape[1], x_test.shape[2], 1
)

# Create a convolutional neural network
model = tf.keras.models.Sequential([

    # Convolutional layer. Learn 32 filters using a 3x3 kernel
    tf.keras.layers.Conv2D(
        32, (3, 3), activation="relu", input_shape=(28, 28, 1)
    ),

    # Max-pooling layer, using 2x2 pool size
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    # Flatten units
    tf.keras.layers.Flatten(),

    # Add a hidden layer with dropout
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5),

    # Add an output layer with output units for all 10 digits
    tf.keras.layers.Dense(10, activation="softmax")
])

# Train neural network
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
model.fit(x_train, y_train, epochs=10)

# Evaluate neural network performance
model.evaluate(x_test,  y_test, verbose=2)

# Actividad 1 
# MOSTARR EL DATASET 5 primeros ejemplos del dataset

plt.figure(figsize=(10, 2))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(x_test[i].reshape(28, 28), cmap="gray")
    plt.title(f"Label: {y_test[i].argmax()}")
    plt.axis('off')
plt.show()

# Actividad 2
# Predecir un ejemplo del Testing y mostrar sus porcentajes de clasificacion
# Mostrar la imagen

index = 244
example = x_test[index]
prediction = model.predict(np.array([example]))
predicted_class = prediction.argmax()

# Mostrar la imagen con la predicción y la etiqueta real
plt.imshow(example.reshape(28, 28), cmap="gray")
plt.title(f"Predicted: {predicted_class}, Actual: {y_test[index].argmax()}")
plt.axis('off')
plt.show()

# Mostrar las probabilidades de clasificación con los primeros 4 decimales
probabilities = prediction[0]
for i, prob in enumerate(probabilities):
    print(f"Probabilidad de {i}: {prob:.4f}")

formatted_prediction = [f"{p:.6f}" for p in prediction[0]]
print("Class probabilities:", formatted_prediction)