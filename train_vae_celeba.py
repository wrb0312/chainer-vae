#!/usr/bin/env python
"""
Chainer example: train a VAE on MNIST
"""
import argparse
import os
from glob import glob
from collections import OrderedDict
import numpy as np
import chainer

import trainer
import net_celebA


def main():
    parser = argparse.ArgumentParser(description='Chainer example: VAE')
    parser.add_argument('--initmodel', '-m', default='',
                        help='Initialize the model from given file')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--data', '-d', required=True, type=str,
                        help='data path')
    parser.add_argument('--batch_size', '-b', type=int, default=64,
                        help='learning minibatch size')
    parser.add_argument('--save_path', '-s', default='results',
                        help='Directory to output the result')
    parser.add_argument('--epochs', '-e', default=100, type=int,
                        help='number of epochs to learn')
    parser.add_argument('--beta', default=1.0, type=float,
                        help='Regularization coefficient for '
                             'the second term of ELBO bound')
    parser.add_argument('--k', '-k', default=1, type=int,
                        help='Number of Monte Carlo samples used in '
                             'encoded vector')
    parser.add_argument('--test', action='store_true',
                        help='Use tiny datasets for quick tests')
    parser.add_argument('--num_snap', '-n', default=10000, type=int,
                        help='snapshot')
    args = parser.parse_args()

    for i in OrderedDict(args.__dict__):
        print("{}: {}".format(i, getattr(args, i)))
    print('')

    # Prepare VAE model, defined in net.py
    encoder = net_celebA.Encoder()
    decoder = net_celebA.Decoder()
    prior = net_celebA.Prior()

    if args.gpu >= 0:
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        encoder.to_gpu(args.gpu)
        decoder.to_gpu(args.gpu)
        prior.to_gpu(args.gpu)

    # Setup an optimizer
    # opt_encoder = chainer.optimizers.Adam(alpha=0.0002)
    # opt_decoder = chainer.optimizers.Adam(alpha=0.0002)
    opt_encoder = chainer.optimizers.Adam()
    opt_decoder = chainer.optimizers.Adam()
    opt_encoder.setup(encoder)
    opt_decoder.setup(decoder)

    # Initialize
    if args.initmodel:
        chainer.serializers.load_npz(args.encoder, encoder)
        chainer.serializers.load_npz(args.decoder, decoder)

    # Load the dataset
    files = np.array(glob(os.path.join(args.data, "*.jpg")))
    num_all = len(files)
    num_train = int(num_all * 0.9)
    num_test = num_all - num_train
    np.random.seed(100)
    id_all = np.random.choice(num_all, num_all, replace=False)
    id_test = id_all[:num_test]
    id_train = id_all[num_test:]
    files_train, files_test = files[id_train], files[id_test]

    if args.test:
        files_train, _ = chainer.datasets.split_dataset(files_train, 100)
        files_test, _ = chainer.datasets.split_dataset(files_test, 100)

    iter_train = chainer.iterators.SerialIterator(files_train, args.batch_size, repeat=False, shuffle=True)
    iter_test = chainer.iterators.SerialIterator(files_test, args.batch_size*2,
                                                 repeat=False, shuffle=False)

    print(num_all)
    print("num train data", len(files_train))
    print("num test data", len(files_test))
    print("")

    N = len(files_train) // args.batch_size if len(files_train) % args.batch_size == 0 else len(files_train) // args.batch_size + 1
    N_test = len(files_test) // args.batch_size if len(files_test) % args.batch_size == 0 else len(files_test) // args.batch_size + 1

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    # Set up an updater. StandardUpdater can explicitly specify a loss function
    # used in the training with 'loss_func' option
    Trainer = trainer.Trainer(
        args,
        iterator=(iter_train, iter_test),
        models=(encoder, decoder, prior),
        optimizers=(opt_encoder, opt_decoder),
        num_snapshot=10000,
        N=N,
        N_test=N_test)
    Trainer()


if __name__ == '__main__':
    main()
