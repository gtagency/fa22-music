import argparse

from lofi.data.maestro import load_maestro
from lofi.models.rnn import create_model
from lofi.train.simple import simple_train, get_loss_and_opt


def main():
    data, seq_length = load_maestro()
    model = create_model(seq_length)
    loss, optimizer = get_loss_and_opt(0.005)
    simple_train(model, loss, optimizer, data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fall 2022 Agency project')
    parser.add_argument('--data', dest='data', default='maestro',
                        help='dataset to use')
    parser.add_argument('--model', dest='model', default='rnn',
                        help='model to train on')
    parser.add_argument('--train', dest='train', default='simple',
                        help='training method')
    args = parser.parse_args()
    main()
