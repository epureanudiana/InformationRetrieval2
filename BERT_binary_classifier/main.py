import torch
from torch import nn
from transformers import BertTokenizer, AdamW

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from preprocess_data import preprocess_data
from dataset import create_data_loader
from binary_classifier import BinaryClassifier
from train import train_epoch
from evaluation import eval_model
from console_args import arg_parser

from sklearn.model_selection import train_test_split



"""
check distribution of tokens in order to set the maximum length of the encoded sequence
this is because bert needs to have the input sequences of the same length. 
It uses [PAD] tokens to reach the MAX_LEN 
"""


def find_sequence_max_length(data):
    token_lens = []

    for msg in data.Message:
        tokens = tokenizer.encode(msg)
        token_lens.append(len(tokens))

    # plot distribution of token lengths
    sns.distplot(token_lens)
    plt.xlim([0, 256])
    plt.xlabel('Token count')


if __name__ == "__main__":

    FLAGS = arg_parser()

    np.random.seed(FLAGS.random_seed)
    torch.manual_seed(FLAGS.random_seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    path_dataset = "BiOpenDialKG.csv"
    data = preprocess_data(path_dataset)

    # split into train, validation and test
    data_train, data_test = train_test_split(data, test_size=0.2, shuffle=True, random_state=FLAGS.random_seed)
    data_val, data_test = train_test_split(data_test, test_size=0.5, shuffle=True, random_state=FLAGS.random_seed)

    # find_sequence_max_length(data)

    # load bert model and tokenizer
    tokenizer = BertTokenizer.from_pretrained(FLAGS.pre_trained_model_name)

    train_data_loader = create_data_loader(data_train, tokenizer, FLAGS.max_len, FLAGS.batch_size)
    val_data_loader = create_data_loader(data_val, tokenizer, FLAGS.max_len, FLAGS.batch_size)
    test_data_loader = create_data_loader(data_test, tokenizer, FLAGS.max_len, FLAGS.batch_size)

    model = BinaryClassifier(2)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=FLAGS.learning_rate, correct_bias=False)
    total_steps = len(train_data_loader) * FLAGS.epochs

    loss_fn = nn.CrossEntropyLoss().to(device)

    best_accuracy = 0

    for epoch in range(FLAGS.epochs):

        print(f'Epoch {epoch + 1}/{FLAGS.epochs}')
        print('----------')

        train_acc, train_loss = train_epoch(model, train_data_loader, loss_fn, optimizer, device, len(data_train))
        print(f'Train loss = {train_loss},  Accuracy =  {train_acc}')

        val_acc, val_loss = eval_model(model, val_data_loader, loss_fn, device, len(data_val))
        print(f'Validation loss = {val_loss},  Accuracy {val_acc}')

        print()

        if val_acc > best_accuracy:
            torch.save(model.state_dict(), 'best_model_state.bin')
            best_accuracy = val_acc
