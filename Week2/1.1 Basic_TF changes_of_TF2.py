import tensorflow as tf
# 1. Set variables and operations to define the model graph
x = tf.placeholder(tf.float32, [None, 32])      # placeholder: entry point to the graph
y = tf.placeholder(tf.float32, [None, 5])

W = tf.get_variable("W", shape=(20, 5), initializer=tf.initializers.glorot_normal())
b = tf.get_variable("b", shape=(5,), initializer=tf.initializers.zeros())
h = tf.matmul(x, W) + b

# 2. Set loss function and optimizer
loss = tf.losses.mean_squared_error(h, y)
opt = tf.train.GRadientDescentOptimizer(0.001)
train_op = opt.minimizer(loss)

# 3. Training
max_steps = 1000
with tf.Session() as sess:                          # train inside session
    sess.run(tf.global_variables_initializer())     # initialize variables before training
    for step in range(max_steps):
        x_batch, y_batch = next(train_batch)
        _, batch_loss = sess.run([train_op, loss], feed_dict={x: x_batch, y: y_batch})


1. Eager execution by default:
    variables and tensors can be used straight way, no need to run initializer or launch session
    x = tf.Variable([1., 2,], name='x')
    print(x)
2. tf.keras is built in tf
3. API cleanup
