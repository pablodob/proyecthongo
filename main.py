import load_data as ld
import tensorflow as tf

# Load data
data = ld.load_data()

labels_data=data[:,0:1]
inputs_data=data[:,1:len(data[0])]

# Variable definition
l = 0.01
epoch = 100
batch_size = 20
n_train = int(len(labels_data)*.9)

# Divide data in inputs and test
labels_train=labels_data[0:n_train]
labels_test=labels_data[n_train+1:len(labels_data)]

inputs_train=inputs_data[0:n_train,:]
inputs_test=inputs_data[n_train+1:len(inputs_data),:]

list_labels_train = []
list_inputs_train = []

num_data = int((len(labels_train)/batch_size))

for i in range (num_data):
    list_inputs_train.append(inputs_train[i*batch_size:(i+1)*batch_size])
    list_labels_train.append(labels_train[i*batch_size:(i+1)*batch_size])

# Inputs
inputs = tf.placeholder(shape=(None,len(inputs_train[0])), dtype=tf.float32, name="inputs")

# Labels
labels = tf.placeholder(shape=(None,1), dtype=tf.float32, name="labels")

# Initializater TensorFlow
init = tf.global_variables_initializer()

# Building the model
input_layer = tf.contrib.layers.fully_connected(inputs=inputs, num_outputs = 100, activation_fn = tf.nn.sigmoid)
hidden1_layer = tf.contrib.layers.fully_connected(inputs=input_layer, num_outputs = 25, activation_fn = tf.nn.sigmoid)
hidden2_layer = tf.contrib.layers.fully_connected(inputs=hidden1_layer, num_outputs = 5, activation_fn = tf.nn.sigmoid)
outputs = tf.contrib.layers.fully_connected(inputs=hidden2_layer, num_outputs = 1, activation_fn = tf.nn.sigmoid)

# Cost and Optimizer
with tf.name_scope("Cost_and_Optimizer"):
    cost = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=outputs, name="cost"))
    optimizer = tf.train.GradientDescentOptimizer(l)
    train = optimizer.minimize(cost)

# Train neural network
tf.summary.scalar("Cost", cost)

with tf.Session() as sess:
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./summaries/lamnda {} epoch {} batch {}".format(l,epoch,batch_size))
    writer.add_graph(sess.graph)
    sess.run(init)

    for i in range(epoch):
        for j in range(num_data):
            sess.run(train, feed_dict={inputs: list_inputs_train[j], labels: list_labels_train[j]})

    #To calcule acurracy and cost
    if i%10:
        partial_cost = sess.run(cost, feed_dict={inputs: inputs_train, labels: labels_train})
        outputs = sess.run(output, feed_dict={inputs: inputs_train, labels: labels_train})
        predictions = outputs > 0.5

        accuracy = np.sum(labels_train == predictions) / len(labels_train)

        print("Train: Epoch N°: {} - Cost = {} - Accuracy = {}%".format(i, partial_cost, accuracy * 100))

    s = sess.run(merged_summary, feed_dict={inputs: inputs_train, labels: labels_train})
    writer.add_summary(s, i)

# Print cost test and final acurracy
final_cost = sess.run(cost, feed_dict={x: inputs_test, y: labels_test})
outputs = sess.run(output, feed_dict={x: inputs_test, y: labels_test})
predictions = outputs > 0.5

final_accuracy = np.sum(labels_test == predictions) / len(labels_test)

print("Test: Cost = {} - Accuracy = {}%".format(final_cost, final_accuracy* 100))