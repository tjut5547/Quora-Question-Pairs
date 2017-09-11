import tensorflow as tf
from tensorflow.python.framework import graph_util

checkpoint = tf.train.get_checkpoint_state('./model')
input_checkpoint = checkpoint.model_checkpoint_path
absolute_model_folder = "/".join(input_checkpoint.split('/')[:-1])
output_graph = absolute_model_folder + "/frozen_model.pb"
output_node_names = "Accuracy/predictions"
clear_devices = True
saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)
graph = tf.get_default_graph()
input_graph_def = graph.as_graph_def()

print(saver, graph, input_graph_def)

with tf.Session() as sess:
    saver.restore(sess, input_checkpoint)

    # We use a built-in TF helper to export variables to constant
    output_graph_def = graph_util.convert_variables_to_constants(
        sess,
        input_graph_def,
        output_node_names.split(",")  # We split on comma for convenience
    )

    # Finally we serialize and dump the output graph to the filesystem
    with tf.gfile.GFile(output_graph, "wb") as f:
        f.write(output_graph_def.SerializeToString())
    print("%d ops in the final graph." % len(output_graph_def.node))