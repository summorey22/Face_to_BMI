import tensorflow as tf
import numpy as np
from keras.preprocessing.image import load_img, img_to_array 

def test_image(path):
    interpreter = tf.lite.Interpreter('bmi.tflite')
    interpreter.allocate_tensors()

    img = load_img(str(path), target_size=(128, 128))
    img_array = np.array(img_to_array(img))
    img_array = img_array.reshape(1, 128, 128, 3)
    img_array = img_array.astype('float32') / 255

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data)
