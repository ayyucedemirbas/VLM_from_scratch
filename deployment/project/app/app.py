from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow_text
import pickle


model = tf.keras.models.load_model('vlm.h5')
with open('tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)

app = Flask(__name__)

def preprocess_image(image, image_size):
    img = load_img(image, target_size=(image_size, image_size))
    img = img_to_array(img) / 255.0
    return np.expand_dims(img, axis=0)

@app.route('/generate_caption', methods=['POST'])
def generate_caption():
    image = request.files['image']
    image_path = f"/tmp/{image.filename}"
    image.save(image_path)

    image_size = 224
    max_length = 20

    preprocessed_image = preprocess_image(image_path, image_size)

    input_sequence = np.zeros((1, max_length))
    caption = []
    for i in range(max_length):
        predictions = model.predict({'input_layer_2': preprocessed_image, 'input_layer_3': input_sequence})
        predicted_word_id = np.argmax(predictions[0, i])

        if predicted_word_id == tokenizer.word_index.get('', None):
            break

        input_sequence[0, i] = predicted_word_id
        word = tokenizer.index_word.get(predicted_word_id, '')
        if word == '':
            break
        caption.append(word)

    return jsonify({'caption': ' '.join(caption)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
