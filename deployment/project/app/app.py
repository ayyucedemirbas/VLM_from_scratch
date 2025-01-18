from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import layers, models
import numpy as np
import pickle
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

image_size = 224
vocab_size = 5001
max_length = 50
embedding_dim = 256
num_heads = 8
ff_dim = 512
num_transformer_blocks = 4

# Define the model architecture
def create_vit_encoder(image_size):
    inputs = layers.Input(shape=(image_size, image_size, 3))

    patch_size = 16
    num_patches = (image_size // patch_size) ** 2
    projection_dim = 768

    patches = layers.Conv2D(
        filters=projection_dim,
        kernel_size=patch_size,
        strides=patch_size,
        padding="valid"
    )(inputs)

    patches = layers.Reshape((num_patches, projection_dim))(patches)

    positional_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)
    positions = tf.range(start=0, limit=num_patches, delta=1)
    encoded_patches = patches + positional_embedding(positions)

    for _ in range(num_transformer_blocks):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim)(x1, x1)
        x2 = layers.Add()([x1, attention_output])

        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        ffn_output = layers.Dense(ff_dim, activation="relu")(x3)
        ffn_output = layers.Dense(projection_dim)(ffn_output)
        encoded_patches = layers.Add()([x2, ffn_output])

    model = models.Model(inputs, encoded_patches)
    return model

def create_text_decoder(vocab_size, embedding_dim, max_length):
    inputs = layers.Input(shape=(max_length,))

    # Word Embeddings + Positional Embeddings
    word_embeddings = layers.Embedding(vocab_size, embedding_dim)(inputs)
    positional_embeddings = layers.Embedding(max_length, embedding_dim)(tf.range(start=0, limit=max_length, delta=1))
    embeddings = word_embeddings + positional_embeddings

    x = embeddings
    for _ in range(num_transformer_blocks):
        x1 = layers.LayerNormalization(epsilon=1e-6)(x)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embedding_dim)(x1, x1)
        x2 = layers.Add()([x1, attention_output])

        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        ffn_output = layers.Dense(ff_dim, activation="relu")(x3)
        ffn_output = layers.Dense(embedding_dim)(ffn_output)
        x = layers.Add()([x2, ffn_output])

    outputs = layers.Dense(vocab_size, activation="softmax")(x)
    model = models.Model(inputs, outputs)
    return model

def create_image_captioning_model(image_size, vocab_size, embedding_dim, max_length):
    vit_encoder = create_vit_encoder(image_size)
    text_decoder = create_text_decoder(vocab_size, embedding_dim, max_length)

    image_inputs = layers.Input(shape=(image_size, image_size, 3), name='input_layer_2')
    text_inputs = layers.Input(shape=(max_length,), name='input_layer_3')

    encoded_image = vit_encoder(image_inputs)

    encoded_image = layers.GlobalAveragePooling1D()(encoded_image)
    encoded_image = layers.Dense(embedding_dim, activation="relu")(encoded_image)
    encoded_image = layers.RepeatVector(max_length)(encoded_image)

    embeddings = layers.Concatenate(axis=2)([encoded_image, text_decoder(text_inputs)])

    outputs = layers.Dense(vocab_size, activation="softmax")(embeddings)

    model = models.Model(inputs=[image_inputs, text_inputs], outputs=outputs)
    return model

# Create the model
model = create_image_captioning_model(image_size, vocab_size, embedding_dim, max_length)

# Load the weights
model.load_weights('vlm.weights.h5')

def preprocess_image(image_path, image_size):
    img = load_img(image_path, target_size=(image_size, image_size))
    img = img_to_array(img) / 255.0
    return np.expand_dims(img, axis=0)

def generate_caption(model, tokenizer, image_path, image_size, max_length):
    image = preprocess_image(image_path, image_size)

    # Initialize the input sequence with zeros
    input_sequence = np.zeros((1, max_length))

    for i in range(max_length):
        predictions = model.predict({'input_layer_2': image, 'input_layer_3': input_sequence})

        predicted_word_id = np.argmax(predictions[0, i])

        if predicted_word_id == tokenizer.word_index.get('', None):
            break

        input_sequence[0, i] = predicted_word_id

    caption = []
    for word_id in input_sequence[0]:
        if word_id == 0:
            continue
        word = tokenizer.index_word.get(word_id, '')
        if word == '':
            break
        caption.append(word)

    return ' '.join(caption)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            caption = generate_caption(model, tokenizer, file_path, image_size, max_length)
            print("Generated Caption:", caption)  # Debug print
            return render_template('index.html', caption=caption, image_url=file_path)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)