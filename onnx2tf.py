import onnx
from onnx_tf.backend import prepare

# onnx load
onnx_model = onnx.load('convert/onnx/test.onnx')
# onnx to tensorflow convert
tf_rep = prepare(onnx_model)
# tensorflow model save
tf_rep.export_graph("convert/TF/tensorflow")