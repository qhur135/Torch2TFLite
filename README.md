# Torch2TFLite
ETRI HRI Lab Internship

## Torch -> TFLite convert  
1. Torch2ONNX
2. onnx2tensorflow 
3. tensorflow2TFLite

라이브러리 버전에 따라 성능 크게 달라짐  
모델에 따라 적절한 버전 찾는 것 필요.

Face parsing, Age estimation 모델 requirements.txt 환경에서 변환하면 제대로 변환됨을 확인함.   
(성능 평가 결과 torch, onnx, tf, tflite 모두 성능 동일함)

|model|python inference time|android inference time|model size|f1 mean|
|:---:|:---:|:---:|:---:|:---:|
|torch|0.6679s|-|266,571KB|0.8080|
|onnx|0.4560s|-|265,715KB|0.8080|
|tensorflow|4.9054s|-|267,314KB|0.8080|
|TFLite float32|3.06s|-|266,358KB|0.8080|
|TFLite float16|3.32s|22.42946s|133,755KB|0.8080|
|TFLite dynamic|382.8495s|9.7638s|69,244KB|-| 

## Convert validation
Torch모델로 predict된 결과와 변환모델로 predict된 결과 비교하여 유사도 체크하는 방식으로 검증

1. ONNX로 변환 제대로 되었는지 확인   
pytorch2onnx.py 코드에 포함되어 있음   
2. tensorflow 변환 제대로 되었는지 확인   
TFconvert_valid.py
3. TFLite 변환 제대로 되었는지 확인    
TFLiteconvert_valid.py  
