import tensorflow as tf
import numpy as np
import model
import opt

tf.logging.set_verbosity(tf.logging.INFO)


def main(unused_argv):

    mnist = tf.contrib.learn.datasets.load_dataset("mnist")

    print([tf.estimator.train_and_evaluate(
        estimator=tf.estimator.Estimator(
            model_fn=model.cnn_classifier,
            model_dir="mnist_{}_model".format(name),
            params=dict(optimizer=optimizer)
        ),
        train_spec=tf.estimator.TrainSpec(
            input_fn=tf.estimator.inputs.numpy_input_fn(
                x={"x": np.asarray(mnist.train.images, dtype=np.float32)},
                y=np.asarray(mnist.train.labels, dtype=np.int32),
                batch_size=100,
                num_epochs=None,
                shuffle=True
            ),
            max_steps=20000,
            hooks=[tf.train.LoggingTensorHook(
                tensors={"probabilities": "softmax"},
                every_n_iter=100
            )]
        ),
        eval_spec=tf.estimator.EvalSpec(
            input_fn=tf.estimator.inputs.numpy_input_fn(
                x={"x": np.asarray(mnist.test.images, dtype=np.float32)},
                y=np.asarray(mnist.test.labels, dtype=np.int32),
                num_epochs=1,
                shuffle=False
            ),
            steps=None
        )
    ) for name, optimizer in [("qhm", opt.QHMOptimizer()), ("qhadam", opt.QHAdamOptimizer())]])


if __name__ == "__main__":
    tf.app.run()
