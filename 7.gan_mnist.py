# pip install tensorflow keras numpy

from numpy import expand_dims, ones, zeros
from numpy.random import rand, randint
from keras.datasets.mnist import load_data
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, LeakyReLU
import tensorflow as tf

# Define the standalone discriminator model
def define_discriminator(in_shape=(28,28,1)):
    model = Sequential([
        tf.keras.Input(shape=in_shape),
        Conv2D(64, (3,3), strides=(2, 2), padding='same'),
        LeakyReLU(negative_slope=0.2),
        Dropout(0.4),
        Conv2D(64, (3,3), strides=(2, 2), padding='same'),
        LeakyReLU(negative_slope=0.2),
        Dropout(0.4),
        Flatten(),
        Dense(1, activation='sigmoid')
    ])
    # Compile model
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

# Load and prepare MNIST training images
def load_real_samples():
    # Load MNIST dataset
    (trainX, _), (_, _) = load_data()
    # Expand to 3D, e.g. add channels dimension
    X = expand_dims(trainX, axis=-1)
    # Convert from unsigned ints to floats
    X = X.astype('float32')
    # Scale from [0,255] to [0,1]
    X = X / 255.0
    return X

# Select real samples
def generate_real_samples(dataset, n_samples):
    # Choose random instances
    ix = randint(0, dataset.shape[0], n_samples)
    # Retrieve selected images
    X = dataset[ix]
    # Generate 'real' class labels (1)
    y = ones((n_samples, 1))
    return X, y

# Generate n fake samples with class labels
def generate_fake_samples(n_samples):
    # Generate uniform random numbers in [0,1]
    X = rand(28 * 28 * n_samples)
    # Reshape into a batch of grayscale images
    X = X.reshape((n_samples, 28, 28, 1))
    # Generate 'fake' class labels (0)
    y = zeros((n_samples, 1))
    return X, y

# Train the discriminator model
@tf.function
def train_step(model, X_real, y_real, X_fake, y_fake):
    with tf.GradientTape() as tape:
        y_pred_real = model(X_real, training=True)
        y_pred_fake = model(X_fake, training=True)
        real_loss = model.compute_loss(X_real, y_real, y_pred_real)
        fake_loss = model.compute_loss(X_fake, y_fake, y_pred_fake)
        total_loss = real_loss + fake_loss
    gradients = tape.gradient(total_loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return real_loss, fake_loss

def train_discriminator(model, dataset, n_iter=100, n_batch=256):
    half_batch = int(n_batch / 2)

    # Manually enumerate epochs
    for i in range(n_iter):
        # Get randomly selected 'real' samples
        X_real, y_real = generate_real_samples(dataset, half_batch)
        # Generate 'fake' examples
        X_fake, y_fake = generate_fake_samples(half_batch)
        # Update discriminator
        real_loss, fake_loss = train_step(model, X_real, y_real, X_fake, y_fake)
        real_acc = tf.reduce_mean(tf.cast(tf.equal(tf.round(model(X_real)), y_real), tf.float32))
        fake_acc = tf.reduce_mean(tf.cast(tf.equal(tf.round(model(X_fake)), y_fake), tf.float32))
        print('>%d real=%.0f%% fake=%.0f%%' % (i+1, real_acc * 100, fake_acc * 100))

# Define the discriminator model
model = define_discriminator()
# Load image data
dataset = load_real_samples()
# Fit the model
train_discriminator(model, dataset)

