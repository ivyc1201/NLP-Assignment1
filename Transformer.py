import os
import time
import tensorflow as tf
import numpy as np
from datasets import Dataset
from sklearn.model_selection import train_test_split
import re


# Step 1: Load Data
def load_files_and_convert_encoding(directory):
    """Loads files from a directory, reads ANSI encoded content, and converts it to UTF-8."""
    files_paths = [os.path.join(directory, filename) for filename in os.listdir(directory) if os.path.isfile(os.path.join(directory, filename))]
    texts = []
    for file_path in files_paths:
        with open(file_path, 'r', encoding='ansi', errors='ignore') as file:
            text = file.read()
            texts.append(text)
    return texts

# Define directory path
data_directory = './jyxstxtqj_downcc.com'

# Load texts using the function
texts = load_files_and_convert_encoding(data_directory)

# Create a Dataset object
dataset = Dataset.from_dict({"text": texts})

# Data Preprocessing

# Tokenization Function
def tokenize_and_encode(examples):
    """
    Encodes the text data using the provided tokenizer. 
    Returns a dictionary with 'input_ids' as keys and lists of token ids as values.
    """
    return {"input_ids": [tokenizer.encode(text) for text in examples["text"]]}

# Apply tokenization to the dataset
tokenized_datasets = dataset.map(tokenize_and_encode, batched=True, remove_columns=["text"])

# Padding Sequences
def pad_sequences(examples):
    """
    Pads sequences to a fixed length 'max_length'. 
    'post' padding adds zeros at the end of sequences shorter than 'max_length'.
    """
    return {"padded_input_ids": tf.keras.preprocessing.sequence.pad_sequences(examples["input_ids"], maxlen=max_length, padding='post')}

# Pad tokenized sequences
max_length = 512
padded_datasets = tokenized_datasets.map(pad_sequences, batched=True)

# Split into Training and Validation Sets
train_val_split = padded_datasets.train_test_split(test_size=0.1, seed=42)  # Added seed for reproducibility

train_input_ids = train_val_split['train']['padded_input_ids']
val_input_ids = train_val_split['test']['padded_input_ids']

# Creating TensorFlow Datasets
def create_tf_dataset(input_ids, batch_size=4, shuffle=True):
    """
    Creates a TensorFlow dataset from input IDs, with options to shuffle and define batch size.
    """
    dataset = tf.data.Dataset.from_tensor_slices(input_ids)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(input_ids))
    return dataset.batch(batch_size)

# Prepare training and validation datasets
train_dataset = create_tf_dataset(train_input_ids)
val_dataset = create_tf_dataset(val_input_ids)

class TransformerLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, nhead):
        super(TransformerLayer, self).__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=nhead, key_dim=d_model//nhead)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dim_feedforward, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])

    def call(self, inputs, training=False):
        attn_output = self.mha(inputs, inputs)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + ffn_output)

class TransformerModel(tf.keras.Model):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, max_seq_length=512):
        super(TransformerModel, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.positional_encoding = tf.Variable(tf.zeros((1, max_seq_length, d_model)), trainable=True, name="positional_encoding")
        self.enc_layers = [TransformerLayer(d_model, nhead) for _ in range(num_layers)]
        self.dec_layers = [TransformerLayer(d_model, nhead) for _ in range(num_layers)]
        self.fc_out = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, training=False, tgt_seq=None):
        src_seq, tgt_seq = inputs[0], tgt_seq
        src_emb = self.embedding(src_seq) + self.positional_encoding[:, :src_seq.shape[1], :]
        for enc_layer in self.enc_layers:
            src_emb = enc_layer(src_emb, training=training)
        if tgt_seq is None:
            return None
        tgt_emb = self.embedding(tgt_seq) + self.positional_encoding[:, :tgt_seq.shape[1], :]
        for dec_layer in self.dec_layers:
            tgt_emb = dec_layer(tgt_emb, enc_output=src_emb, training=training)
        return self.fc_out(tgt_emb)

# Define model hyperparameters
vocab_size = tokenizer.vocab_size
model = TransformerModel(vocab_size)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Training steps
start_time = datetime.now()
with tf.device(device):
    model.fit(train_dataset.batch(batch_size), epochs=3, validation_data=val_dataset.batch(batch_size))

end_time = datetime.now()
elapsed_time = (end_time - start_time).total_seconds()
print(f"Training completed in: {elapsed_time // 3600}h {(elapsed_time % 3600) // 60}m {elapsed_time % 60}s")

# Save the model and tokenizer
model.save_weights("./transformer-finetuned-novels-tf.h5")
np.save("./transformer-finetuned-novels-tf-vocab.npy", tokenizer.vocab)

def generate_text(model, tokenizer, prompt, max_length=200):
    input_ids = tokenizer.encode(prompt)
    output_ids = input_ids
    for _ in range(max_length):
        output_probs = model.predict([tf.expand_dims(output_ids, axis=0)] * 2)[-1, -1, :]  # Duplicate input for encoder and decoder
        next_token_id = tf.argmax(output_probs).numpy()
        if next_token_id == tokenizer.token_to_id.get('[SEP]', -1):
            break
        output_ids.append(next_token_id)
    return tokenizer.decode(output_ids)

# Generate text from a novel snippet
prompt = "苏鲁克和车尔库见四周情势凶险"
generated_text = generate_text(model, tokenizer, prompt)
print(generated_text)