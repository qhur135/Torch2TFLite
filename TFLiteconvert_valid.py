# https://blog.naver.com/PostView.naver?blogId=seodaewoo&logNo=222043145688&parentCategoryNo=&categoryNo=62&viewDate=&isShowPopularPosts=false&from=postView
# 필요한 import문
import numpy as np
import torch
import network
import tensorflow as tf

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# torch model path
TORCH_PATH = './convert/torch/best_for_test_256.pth'
# tflite model path
TFLITE_PATH = './convert/test_f16.tflite'
batch_size = 1
# input data
input = torch.randn(batch_size, 3, 256, 256, requires_grad=True)
input_data = np.array(to_numpy(input), dtype=np.float32)

# 모델을 미리 학습된 가중치로 초기화합니다
map_location = lambda storage, loc: storage
if torch.cuda.is_available():
    map_location = None
# torch load
network.torch_model.load_state_dict(torch.load(TORCH_PATH, map_location=map_location))
# 모델을 추론 모드로 전환합니다
network.torch_model.eval()
# torch output
torch_out = network.torch_model(torch.from_numpy(input_data))

# tflite모델 로딩 및 텐서 할당
interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)

# my_signature = interpreter.get_signature_runner()
# output = my_signature(input_data)

interpreter.allocate_tensors()
# 입출력 텐서 가져오기
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
output_shape = output_details[0]['shape']
# model에 input data 넣기
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
# tflite output
output_data = interpreter.get_tensor(output_details[0]['index']) # numpy

# test print
# tensor_details = interpreter.get_tensor_details()


# signatures = interpreter.get_signature_list()
# print(signatures) # {'serving_default': {'inputs': ['input'], 'outputs': ['output_0']}}
print(output_data.shape)
# print(tensor_details[1])


# ONNX 런타임과 PyTorch에서 연산된 결과값 비교
np.testing.assert_allclose(to_numpy(torch_out), output_data, rtol=1e-03, atol=1e-05)
print("Exported model has been tested with ONNXRuntime, and the result looks good!")
