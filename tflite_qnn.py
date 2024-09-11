import numpy as np
import os
import tensorflow as tf
from PIL import Image
import tqdm

# Load your TFLite model
tflite_model_path = 'YOLOv8-Detection-Quantized.tflite'
with open(tflite_model_path, 'rb') as f:
    tflite_model = f.read()

QNN_SDK_PATH = '/home/aim/Documents/v2.26.0.240828/qairt/2.26.0.240828'
HEXAGON_VERSION = '68'
lib_path = f'{QNN_SDK_PATH}/lib/aarch64-ubuntu-gcc9.4/libQnnTFLiteDelegate.so'

# Set up the TFLite interpreter
skel_dir_path = f'{QNN_SDK_PATH}/lib/hexagon-v{HEXAGON_VERSION}/unsigned'

LD_LIBRARY_PATH = os.getenv("LD_LIBRARY_PATH")
ADSP_LIBRARY_PATH = os.getenv("ADSP_LIBRARY_PATH")

print(f"[INFO] LD_LIBRARY_PATH = {LD_LIBRARY_PATH}")
print(f"[INFO] LD_LIBRARY_PATH exists = {os.path.exists(LD_LIBRARY_PATH)}")
print(f"[INFO] ADSP_LIBRARY_PATH = {ADSP_LIBRARY_PATH}")
print(f"[INFO] ADSP_LIBRARY_PATH exists = {os.path.exists(ADSP_LIBRARY_PATH)}")
print(f"[INFO] Skel dir path = {skel_dir_path}")
print(f"[INFO] Path exists = {os.path.exists(skel_dir_path)}")

# Define backend type
backend_type = 'htp'  # 'htp' or 'qnn'

try:
    delegate = tf.lite.experimental.load_delegate(lib_path, options={'backend_type': backend_type})
    print(f"Delegate loaded successfully with '{backend_type}' backend")
except Exception as e:
    print(f"Failed to load delegate with '{backend_type}' backend: {e}")

interpreter = tf.lite.Interpreter(
    model_content=tflite_model,
    experimental_delegates=[delegate]
)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load and preprocess the image
def preprocess_image(image_path, target_size):
    image = Image.open(image_path).convert('RGB')
    image = image.resize(target_size)
    image_array = np.array(image)
    image_array = image_array.astype(np.uint8)  # Match input type expected by the model
    return image_array

# Path to your image file
image_path = 'people.jpg'

# Preprocess the image
input_shape = input_details[0]['shape']
preprocessed_image = preprocess_image(image_path, tuple(input_shape[1:3]))

# Check the input data type and shape
print(f"[INFO] Input shape: {input_shape}")
print(f"[INFO] Preprocessed image shape: {preprocessed_image.shape}")

# Set the input tensor
interpreter.set_tensor(input_details[0]['index'], np.expand_dims(preprocessed_image, axis=0))

# Run inference
# for _ in tqdm.trange(1):  # Single inference
interpreter.invoke()

# Get output details and shape
output_details = interpreter.get_output_details()
output_shapes = [detail['shape'] for detail in output_details]

print(f"[INFO] Output details: {output_details}")
print(f"[INFO] Output shapes: {output_shapes}")

# Get the output of the model
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)

