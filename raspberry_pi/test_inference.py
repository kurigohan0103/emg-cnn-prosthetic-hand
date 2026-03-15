from inference import InferenceEngine
import time

engine = InferenceEngine()

for i in range(5):
    result = engine.run_inference()
    if result:
        print(f"推論{i+1}: Task {result}")
    else:
        print(f"推論{i+1}: エラー")
    time.sleep(1)