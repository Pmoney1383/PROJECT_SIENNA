import os
import math
import random
import numpy as np
import torch

vocab_size = 0
embedding_dim = 64
hidden_dim = 128
batch_size = 32
epochs = 10
learning_rate = 0.01

input_embedding = None
output_embedding = None
rnn_weights = None
output_weights = None

input_sequences = []
output_sequences = []

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_data():
    global vocab_size

    max_token = 0

    input_path = os.path.join(BASE_DIR, "_PREP_DATA", "padded_inputs.txt")
    output_path = os.path.join(BASE_DIR, "_PREP_DATA", "padded_outputs.txt")

    with open(input_path, "r") as f:
        for line in f:
            tokens = list(map(int, line.strip().split()))
            input_sequences.append(tokens)
            max_token = max(max_token, max(tokens, default=0))

    with open(output_path, "r") as f:
        for line in f:
            tokens = list(map(int, line.strip().split()))
            output_sequences.append(tokens)
            max_token = max(max_token, max(tokens, default=0))

    vocab_size = max_token + 1

def initialize_parameters():
    global input_embedding, output_embedding, rnn_weights, output_weights

    input_embedding = (np.random.rand(vocab_size, embedding_dim) - 0.5) * 0.1
    output_embedding = (np.random.rand(vocab_size, embedding_dim) - 0.5) * 0.1
    rnn_weights = (np.random.rand(hidden_dim, embedding_dim + hidden_dim) - 0.5) * 0.1
    output_weights = (np.random.rand(vocab_size, hidden_dim) - 0.5) * 0.1

def split_into_chunks(data, chunk_size):
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

def create_batches():
    inputs = input_sequences.copy()
    targets = output_sequences.copy()
    combined = list(zip(inputs, targets))
    random.shuffle(combined)
    inputs[:], targets[:] = zip(*combined)
    input_batches = split_into_chunks(inputs, batch_size)
    output_batches = split_into_chunks(targets, batch_size)
    return input_batches, output_batches

def get_row(matrix, row):
    return matrix[row]

def concatenate(a, b):
    return np.concatenate((a, b))

def tanh(x):
    return np.tanh(x)

def softmax(logits):
    logits = logits - np.max(logits)
    exps = np.exp(logits)
    return exps / np.sum(exps)

def multiply_on_gpu(matrix, vector):
    matrix_tensor = torch.tensor(matrix, dtype=torch.float32).cuda()
    vector_tensor = torch.tensor(vector, dtype=torch.float32).cuda()
    result = torch.matmul(matrix_tensor, vector_tensor)
    return result.cpu().numpy()

def train_step(inputs, targets):
    total_loss = 0

    for input_seq, target_seq in zip(inputs, targets):
        hidden = np.zeros((len(input_seq), hidden_dim))
        prev_hidden = np.zeros(hidden_dim)

        for t, token in enumerate(input_seq):
            x = get_row(input_embedding, token)
            combined = concatenate(x, prev_hidden)
            h = tanh(multiply_on_gpu(rnn_weights, combined))
            hidden[t] = h
            prev_hidden = h

        logits = multiply_on_gpu(output_weights, prev_hidden)
        probs = softmax(logits)
        target_token = target_seq[-1]
        loss = -math.log(probs[target_token] + 1e-9)
        total_loss += loss

    return total_loss

def save_matrix(file_path, matrix):
    with open(file_path, "w") as f:
        for row in matrix:
            f.write(" ".join([f"{val:.6f}" for val in row]) + "\n")

def save_all_weights():
    os.makedirs("_MODEL_WEIGHTS", exist_ok=True)
    save_matrix("_MODEL_WEIGHTS/inputEmbedding.txt", input_embedding)
    save_matrix("_MODEL_WEIGHTS/outputEmbedding.txt", output_embedding)
    save_matrix("_MODEL_WEIGHTS/rnnWeights.txt", rnn_weights)
    save_matrix("_MODEL_WEIGHTS/outputWeights.txt", output_weights)
    print("💾 All weights saved to _MODEL_WEIGHTS folder.")

def main():
    print("🔄 Loading padded training data...")
    load_data()

    print("⚙️ Initializing model parameters...")
    initialize_parameters()

    print("🚀 Starting training...")
    for epoch in range(1, epochs + 1):
        total_loss = 0
        total_tokens = 0

        input_batches, output_batches = create_batches()
        print(f"\n📦 Epoch {epoch}/{epochs} | Batches: {len(input_batches)}")

        for i, (inputs, targets) in enumerate(zip(input_batches, output_batches)):
            loss = train_step(inputs, targets)
            total_loss += loss
            total_tokens += len(inputs) * len(inputs[0])
            percent = ((i + 1) / len(input_batches)) * 100
            print(f"\r   🔁 Batch {i + 1}/{len(input_batches)} ({percent:.1f}%) | Loss: {loss:.4f}", end="")

        print(f"\n✅ Epoch {epoch}/{epochs} | Avg Loss per Token: {total_loss / total_tokens:.4f}")

    print("🎉 Training complete!")
    save_all_weights()

main()
