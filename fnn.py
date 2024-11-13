import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
from tqdm import tqdm
import json
from argparse import ArgumentParser
import pandas as pd
import matplotlib.pyplot as plt

unk = '<UNK>'

class FFNN(nn.Module):
    def __init__(self, input_dim, h, dropout_rate=0.1):
        super(FFNN, self).__init__()
        self.h = h
        self.W1 = nn.Linear(input_dim, h)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.output_dim = 5
        self.W2 = nn.Linear(h, self.output_dim)
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, input_vector):
        x1 = self.activation(self.W1(input_vector))
        x1 = self.dropout(x1)
        y = self.W2(x1)
        predicted_vector = self.softmax(y.view(-1, self.output_dim))
        return predicted_vector

# Other helper functions as before (make_vocab, make_indices, convert_to_vector_representation, load_data, load_test_data)
# Returns: 
# vocab = A set of strings corresponding to the vocabulary
def make_vocab(data):
    vocab = set()
    for document, _ in data:
        for word in document:
            vocab.add(word)
    return vocab 


# Returns:
# vocab = A set of strings corresponding to the vocabulary including <UNK>
# word2index = A dictionary mapping word/token to its index (a number in 0, ..., V - 1)
# index2word = A dictionary inverting the mapping of word2index
def make_indices(vocab):
    vocab_list = sorted(vocab)
    vocab_list.append(unk)
    word2index = {}
    index2word = {}
    for index, word in enumerate(vocab_list):
        word2index[word] = index 
        index2word[index] = word 
    vocab.add(unk)
    return vocab, word2index, index2word 


# Returns:
# vectorized_data = A list of pairs (vector representation of input, y)
def convert_to_vector_representation(data, word2index):
    vectorized_data = []
    for document, y in data:
        vector = torch.zeros(len(word2index)) 
        for word in document:
            index = word2index.get(word, word2index[unk])
            vector[index] += 1
        vectorized_data.append((vector, y))
    return vectorized_data



def load_data(train_data, val_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)
    tra = []
    val = []
    for elt in training:
        tra.append((elt["text"].split(),int(elt["stars"]-1)))
    for elt in validation:
        val.append((elt["text"].split(),int(elt["stars"]-1)))

    return tra, val

def load_test_data(test_data):
    with open(test_data) as test_f:
        test = json.load(test_f)

    tes = []
    for elt in test:
        tes.append((elt["text"].split(),int(elt["stars"]-1)))
    return tes

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required=True, help="hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required=True, help="num of epochs to train")
    parser.add_argument("--train_data", required=True, help="path to training data")
    parser.add_argument("--val_data", required=True, help="path to validation data")
    parser.add_argument("--test_data", default="to fill", help="path to test data")
    parser.add_argument('--do_train', action='store_true')
   # parser.add_argument('-lr', '--learning_rate', type=float, default=0.01, help="learning rate")
    args = parser.parse_args()
    


    random.seed(42)
    torch.manual_seed(42)

    print("========== Loading data ==========")
    train_data, valid_data = load_data(args.train_data, args.val_data)
    vocab = make_vocab(train_data)
    vocab, word2index, index2word = make_indices(vocab)

    print("========== Vectorizing data ==========")
    train_data = convert_to_vector_representation(train_data, word2index)
    valid_data = convert_to_vector_representation(valid_data, word2index)

    model = FFNN(input_dim=len(vocab), h=args.hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

    plt.style.use('ggplot')
    print("========== Training for {} epochs ==========".format(args.epochs))
    train_errors, valid_errors = [], []
    train_accuracies, valid_accuracies = [], []
    epochs = []

    for epoch in range(args.epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        correct, total = 0, 0
        epoch_loss = 0
        random.shuffle(train_data)
        minibatch_size = 32
        N = len(train_data)
        
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_vector, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                predicted_vector = model(input_vector)
                predicted_label = torch.argmax(predicted_vector)
                correct += int(predicted_label == gold_label)
                total += 1
                example_loss = model.compute_Loss(predicted_vector.view(1, -1), torch.tensor([gold_label]))
                epoch_loss += example_loss.item()
                loss = example_loss if loss is None else loss + example_loss
            
            loss = loss / minibatch_size
            loss.backward()
            optimizer.step()
        
        train_accuracy = correct / total
        train_errors.append(1 - train_accuracy)
        train_accuracies.append(train_accuracy)
        epochs.append(epoch)

        print(f"Training accuracy for epoch {epoch + 1}: {train_accuracy:.4f}")
        
        # Validation
        model.eval()
        correct, total = 0, 0
        valid_loss = 0
        with torch.no_grad():
            for minibatch_index in tqdm(range(len(valid_data) // minibatch_size)):
                for example_index in range(minibatch_size):
                    input_vector, gold_label = valid_data[minibatch_index * minibatch_size + example_index]
                    predicted_vector = model(input_vector)
                    predicted_label = torch.argmax(predicted_vector)
                    correct += int(predicted_label == gold_label)
                    total += 1
                    example_loss = model.compute_Loss(predicted_vector.view(1, -1), torch.tensor([gold_label]))
                    valid_loss += example_loss.item()
        
        valid_accuracy = correct / total
        valid_errors.append(1 - valid_accuracy)
        valid_accuracies.append(valid_accuracy)

        print(f"Validation accuracy for epoch {epoch + 1}: {valid_accuracy:.4f}")

    # Plotting the learning curves
    plt.plot(epochs, train_errors, label='Training Error')
    plt.plot(epochs, valid_errors, label='Validation Error')
    plt.ylabel('Error')
    plt.xlabel('Epochs')
    plt.title('Learning Curves for FFNN Model')
    plt.legend()
    #plt.savefig(f'res_ffnn_h{args.hidden_dim}_e{args.epochs}_lr{args.lr}_error.png')
    plt.savefig(f'res_ffnn_h{args.hidden_dim}_e{args.epochs}_lr0.001_error.png')

    plt.show()

    # Print final accuracy values for reference
    print("\nFinal Training Accuracy:", train_accuracies[-1])
    print("Final Validation Accuracy:", valid_accuracies[-1])

    # Test Prediction and Save Results to CSV
    test_data = load_test_data(args.test_data)
    test_data = convert_to_vector_representation(test_data, word2index)
    results = []

    model.eval()
    with torch.no_grad():
        for input_vector, _ in test_data:  # Unpacking the tuple
            predicted_vector = model(input_vector)
            predicted_label = torch.argmax(predicted_vector).item()
            results.append(predicted_label)

    # Save results to CSV
    df = pd.DataFrame({"id": np.arange(len(results)), "stars": results})
    df.to_csv('results_ffnn.csv', index=False)
    print("Test predictions saved to 'results_ffnn.csv'")
