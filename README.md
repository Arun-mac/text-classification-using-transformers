Transformer-Based Text Classification
Overview
This repository contains a TensorFlow implementation of a transformer-based neural network for text classification. The model is designed for sentiment analysis using the IMDB movie reviews dataset.

Architecture
The architecture consists of custom layers for transformer blocks and positional embedding:

Transformer Block:

Multi-head self-attention mechanism.
Feedforward neural network with ReLU activation.
Layer normalization and dropout for stabilization.
Token and Position Embedding:

Combines token embeddings and positional embeddings.
Utilizes separate embedding layers for tokens and positions.
Dataset
The IMDB movie reviews dataset is used for training and validation. The dataset is preprocessed to limit the vocabulary to the top 20,000 words, and sequences are truncated or padded to a maximum length of 400.

python
Copy code
vocab_size = 20000
maxlen = 400

(x_train, y_train), (x_val, y_val) = imdb.load_data(num_words=vocab_size)
print(len(x_train), "Training sequences")
print(len(x_val), "Validation sequences")
Model Training
The model is defined with an embedding layer, a transformer block, global average pooling, dropout for regularization, and dense layers with ReLU activation. It uses softmax activation for binary classification.

python
Copy code
embed_dim = 32
num_heads = 2
ff_dim = 32

inputs = Input(shape=(maxlen,))
embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
x = embedding_layer(inputs)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(x)
x = GlobalAveragePooling1D()(x)
x = Dropout(0.1)(x)
x = Dense(20, activation="relu")(x)
x = Dropout(0.1)(x)
outputs = Dense(2, activation="softmax")(x)

model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

history = model.fit(x_train, y_train, batch_size=64, epochs=2, validation_data=(x_val, y_val))
Model Evaluation
The trained model is evaluated on the validation data, and results are printed, including metrics such as loss and accuracy.

python
Copy code
results = model.evaluate(x_val, y_val, verbose=2)

for name, value in zip(model.metrics_names, results):
    print("%s: %.3f" % (name, value))
The model achieves an accuracy of 89.01% on the validation set. The trained weights are saved to "predict_class.h5".
