import argparse

RANDOM_SEED = 42
MAX_LEN = 65
BATCH_SIZE = 16
EPOCHS = 1
LEARNING_RATE = 2e-5
PRE_TRAINED_MODEL_NAME = 'bert-base-cased'

"""
Method used for parsing command line arguments.
"""
def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=RANDOM_SEED,
                        help='random seed')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='Batch size to run trainer.')
    parser.add_argument('--pre_trained_model_name', type=str, default=PRE_TRAINED_MODEL_NAME,
                        help='Pre-trained model name')
    parser.add_argument('--max_len', type=str, default=MAX_LEN,
                        help='Sequence maximum length')
    FLAGS, un_parsed = parser.parse_known_args()

    return FLAGS
