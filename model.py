import tensorflow as tf


def cnn_classifier(features, labels, mode, params):
    """Model function for CNN."""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # MNIST images are 28x28 pixels, and have one color channel
    inputs = tf.reshape(
        tensor=features["x"],
        shape=[-1, 28, 28, 1]
    )
    # Convolutional Layer #1
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 28, 28, 1]
    # Output Tensor Shape: [batch_size, 28, 28, 32]
    inputs = tf.layers.conv2d(
        inputs=inputs,
        filters=32,
        kernel_size=[5, 5],
        strides=[1, 1],
        padding="same",
        use_bias=False
    )
    inputs = tf.layers.batch_normalization(
        inputs=inputs,
        axis=-1,
        training=mode == tf.estimator.ModeKeys.TRAIN
    )
    inputs = tf.nn.relu(inputs)
    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 28, 28, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 32]
    inputs = tf.layers.max_pooling2d(
        inputs=inputs,
        pool_size=[2, 2],
        strides=[2, 2],
        padding="same"
    )
    # Convolutional Layer #2
    # Computes 64 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 14, 14, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 64]
    inputs = tf.layers.conv2d(
        inputs=inputs,
        filters=64,
        kernel_size=[5, 5],
        strides=[1, 1],
        padding="same",
        use_bias=False
    )
    inputs = tf.layers.batch_normalization(
        inputs=inputs,
        axis=-1,
        training=mode == tf.estimator.ModeKeys.TRAIN
    )
    inputs = tf.nn.relu(inputs)
    # Pooling Layer #2
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 14, 14, 64]
    # Output Tensor Shape: [batch_size, 7, 7, 64]
    inputs = tf.layers.max_pooling2d(
        inputs=inputs,
        pool_size=[2, 2],
        strides=[2, 2],
        padding="same"
    )
    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 7, 7, 64]
    # Output Tensor Shape: [batch_size, 7 * 7 * 64]
    inputs = tf.layers.flatten(inputs)
    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 7 * 7 * 64]
    # Output Tensor Shape: [batch_size, 1024]
    inputs = tf.layers.dense(
        inputs=inputs,
        units=1024,
        use_bias=False
    )
    inputs = tf.layers.batch_normalization(
        inputs=inputs,
        axis=-1,
        training=mode == tf.estimator.ModeKeys.TRAIN
    )
    inputs = tf.nn.relu(inputs)
    # Logits layer
    # Input Tensor Shape: [batch_size, 64]
    # Output Tensor Shape: [batch_size, 10]
    logits = tf.layers.dense(
        inputs=inputs,
        units=10
    )
    # Add `softmax` to the graph. It is used by the `logging_hook`.
    probabilities = tf.nn.softmax(
        logits=logits,
        name="softmax"
    )
    # Generate predictions (for PREDICT and EVAL mode)
    classes = tf.argmax(
        input=probabilities,
        axis=-1
    )
    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(
        labels=labels,
        logits=logits
    )
    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=params["optimizer"].minimize(
                    loss=loss,
                    global_step=tf.train.get_global_step()
                )
            )
    # Add evaluation metrics (for EVAL mode)
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            eval_metric_ops=dict(
                accuracy=tf.metrics.accuracy(
                    labels=labels,
                    predictions=classes
                )
            )
        )
