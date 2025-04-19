# NLP Sentiment Analysis using IMDB Dataset

This project implements a sentiment analysis model that classifies movie reviews as positive or negative using the IMDB dataset. The model is built using TensorFlow 2.x and Keras, and it uses text preprocessing techniques like tokenization and padding to prepare the dataset for model training.

## Project Overview

The goal of this project is to predict the sentiment of a given movie review as either positive or negative. We use a deep learning model consisting of an embedding layer followed by a global average pooling layer and fully connected layers to output the sentiment label.

## Technologies Used

- **TensorFlow 2.x**: The core library for building and training the model.
- **Keras**: A high-level API for building and training neural networks.
- **Pandas**: For data manipulation and loading the dataset.
- **NumPy**: For handling numerical data and arrays.
- **Matplotlib** (optional, not used in the code): For plotting training and validation metrics if needed.

## Installation

To run this project locally, you need to have Python installed along with the following dependencies:

1. Install TensorFlow:
    ```bash
    pip install tensorflow
    ```

2. Install other dependencies:
    ```bash
    pip install pandas numpy
    ```

## Dataset

This model uses the IMDB dataset, which contains movie reviews labeled as either "positive" or "negative." The dataset is loaded from a CSV file named `IMDBDataset.csv` and preprocessed to prepare the text for training.

### Data Format

The dataset contains two columns:

- **review**: A text field containing the movie review.
- **sentiment**: A label indicating the sentiment of the review, either "positive" or "negative".

Example:

| review | sentiment |
|--------|-----------|
| One of the best movies I've ever seen! | positive |
| The movie was terrible, I would not recommend it. | negative |

## Model Architecture

The model consists of the following layers:

1. **Embedding Layer**: Converts words into dense vectors of fixed size (16 in this case).
2. **Global Average Pooling Layer**: Averages the word embeddings across the entire sequence to produce a fixed-length output.
3. **Dense Layer**: A fully connected layer with 6 units and ReLU activation.
4. **Output Layer**: A final fully connected layer with a single unit and a sigmoid activation function to output the sentiment label (0 or 1).

## Training the Model

The model is trained on the IMDB dataset using the following parameters:

- **vocab_size**: 30,000 words
- **embedding_dim**: 16
- **max_length**: 220 (maximum length of input sequences)
- **epochs**: 30
- **batch_size**: Not specified, defaults to TensorFlow's batch size.

The model uses binary cross-entropy loss and the Adam optimizer, and it tracks the accuracy metric during training.

## Usage

To classify a movie review, you can use the trained model to predict whether the sentiment of a given review is positive or negative.

Example:

```python
# Sample review
test_review = ["I had high expectations for this movie given the hype around it, but I was left feeling disappointed. The plot was thin and predictable, and the characters lacked depth."]

# Preprocess the review
test_sequences = tokenizer.texts_to_sequences(test_review)
test_padded = tf.keras.preprocessing.sequence.pad_sequences(test_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

# Predict with the model
prediction = model.predict(test_padded)
if prediction[0] > 0.5:
    print("Positive Review")
else:
    print("Negative Review")
