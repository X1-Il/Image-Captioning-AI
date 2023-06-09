import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding
from tensorflow.keras.optimizers import Adam

# Load the pre-trained InceptionV3 model
inception_model = InceptionV3(weights='imagenet')
inception_model = Model(inception_model.input, inception_model.layers[-2].output)

# Define the caption generation model
inputs1 = Input(shape=(2048,))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)

inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(256)(se2)

decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)
caption_model = Model(inputs=[inputs1, inputs2], outputs=outputs)

# Compile the model
caption_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001))

# Function to preprocess and encode the image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(299, 299))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

# Function to generate a caption for an image
def generate_caption(image_path):
    img = preprocess_image(image_path)
    features = inception_model.predict(img)

    caption_input = np.zeros((1, max_length))
    caption_input[0, 0] = tokenizer.word_index['start']

    for i in range(max_length - 1):
        caption_output = caption_model.predict([features, caption_input])
        index = np.argmax(caption_output[0, i, :])
        caption_input[0, i + 1] = index

        if tokenizer.index_word[index] == 'end':
            break

    caption = [tokenizer.index_word[i] for i in caption_input[0] if i != 0]
    caption = ' '.join(caption[1:-1])
    return caption

# Run the image captioning system
image_path = 'path_to_image.jpg'  # Add the path to your image
caption = generate_caption(image_path)
print("Image Caption:", caption)
