import chainer
import chainer.distributions as D
import chainer.functions as F
import chainer.links as L
from instance_normalization import InstanceNormalization
import numpy as np


class ResBlock(chainer.Chain):
    def __init__(self, ch_in, ch_out, initialW):
        super(ResBlock, self).__init__()
        with self.init_scope():
            if ch_in != ch_out:
                self.shortcut = L.Convolution2D(ch_in, ch_out, 1, 1, 0, initialW=initialW)
                # self.in0 = InstanceNormalization(ch_out)
                self.in0 = L.BatchNormalization(ch_out)
                self.is_shortcut = True
            else:
                self.is_shortcut = False
            self.c1 = L.Convolution2D(ch_in, ch_out, 3, 1, 1, initialW=initialW)
            self.c2 = L.Convolution2D(ch_out, ch_out, 3, 1, 1, initialW=initialW)

            # self.in1 = InstanceNormalization(ch_out)
            # self.in2 = InstanceNormalization(ch_out)
            self.in1 = L.BatchNormalization(ch_out)
            self.in2 = L.BatchNormalization(ch_out)

    def __call__(self, x):
        h = F.relu(self.in1(self.c1(x)))
        h = self.in2(self.c2(h))
        if self.is_shortcut:
            x = self.in0(self.shortcut(x))
        h = F.relu(h + x)
        return h


class Encoder(chainer.Chain):
    def __init__(self):
        super(Encoder, self).__init__()
        self.n_ch = 64
        w = chainer.initializers.HeNormal()
        # w = chainer.initializers.Normal(scale=0.02, dtype=None)
        with self.init_scope():
            self.c1 = L.Convolution2D(3, self.n_ch, 5, 1, 2, initialW=w)
            self.res2 = ResBlock(self.n_ch, self.n_ch*2, initialW=w)
            self.res3 = ResBlock(self.n_ch*2, self.n_ch*4, initialW=w)
            self.res4 = ResBlock(self.n_ch*4, self.n_ch*8, initialW=w)
            self.res5 = ResBlock(self.n_ch*8, self.n_ch*8, initialW=w)
            # self.res6 = ResBlock(256, 256)
            # self.c7 = L.Convolution2D(self.n_ch*4, self.n_ch*8, 4, 1, 0)
            self.l6 = L.Linear(None, self.n_ch*8, initialW=w)
            # self.l_std = L.Linear(None, self.n_ch*4, initialW=w)

            self.bn_c1 = L.BatchNormalization(self.n_ch)
            self.bn_l6 = L.BatchNormalization(self.n_ch*8)
            # self.bn_std = L.BatchNormalization(self.n_ch*4)

    def __call__(self, x):
        h = F.relu(self.bn_c1(self.c1(x)))
        h = F.average_pooling_2d(h, 2, 2, 0)

        h = F.relu(self.res2(h))
        h = F.average_pooling_2d(h, 2, 2, 0)

        h = F.relu(self.res3(h))
        h = F.average_pooling_2d(h, 2, 2, 0)

        h = F.relu(self.res4(h))
        h = F.average_pooling_2d(h, 2, 2, 0)

        h = F.relu(self.res5(h))
        # h = F.average_pooling_2d(h, 2, 2, 0)

        # h = F.relu(self.res6(h))

        # h = self.c7(h)
        h = self.bn_l6(self.l6(h))
        # s = self.bn_std(self.l_std(h))

        return D.Normal(loc=h[:, :self.n_ch*4], log_scale=h[:, self.n_ch*4:])
        # return D.Normal(loc=m, log_scale=s)


class Decoder(chainer.Chain):
    def __init__(self):
        super(Decoder, self).__init__()
        self.n_ch = 64
        w = chainer.initializers.HeNormal()
        # w = chainer.initializers.Normal(scale=0.02, dtype=None)
        with self.init_scope():
            # self.c1 = L.Convolution2D(self.n_ch*4, self.n_ch*4, 4, 1, 3)
            self.l1 = L.Linear(None, self.n_ch*4*4*4, initialW=w)
            # self.res2 = ResBlock(256, 256)
            self.res3 = ResBlock(self.n_ch*4, self.n_ch*8, initialW=w)
            self.res4 = ResBlock(self.n_ch*8, self.n_ch*8, initialW=w)
            self.res5 = ResBlock(self.n_ch*8, self.n_ch*4, initialW=w)
            self.res6 = ResBlock(self.n_ch*4, self.n_ch*2, initialW=w)
            self.res7 = ResBlock(self.n_ch*2, self.n_ch, initialW=w)
            self.c8 = L.Convolution2D(self.n_ch, 3, 5, 1, 2, initialW=w)

            self.bn_l1 = L.BatchNormalization(self.n_ch*4)

    def __call__(self, z):
        k, b, u = z.shape
        z = F.reshape(z, (k*b, u))
        h = self.l1(z)
        h = h.reshape(k*b, self.n_ch*4, 4, 4)
        h = F.relu(self.bn_l1(h))

        # h = F.relu(self.res2(h))
        # h = F.unpooling_2d(h, 2, 2, 0, outsize=(8, 8))

        h = F.relu(self.res3(h))
        h = F.unpooling_2d(h, 2, 2, 0, outsize=(8, 8))
        # h = F.unpooling_2d(h, 2, 2, 0, outsize=(16, 16))

        h = F.relu(self.res4(h))
        h = F.unpooling_2d(h, 2, 2, 0, outsize=(16, 16))
        # h = F.unpooling_2d(h, 2, 2, 0, outsize=(32, 32))

        h = F.relu(self.res5(h))
        h = F.unpooling_2d(h, 2, 2, 0, outsize=(32, 32))
        # h = F.unpooling_2d(h, 2, 2, 0, outsize=(64, 64))

        h = F.relu(self.res6(h))
        h = F.unpooling_2d(h, 2, 2, 0, outsize=(64, 64))
        # h = F.unpooling_2d(h, 2, 2, 0, outsize=(128, 128))

        h = F.relu(self.res7(h))
        x_hat = F.tanh(self.c8(h))

        return x_hat


class Encoder_2(chainer.Chain):
    def __init__(self):
        super(Encoder_2, self).__init__()
        with self.init_scope():
            self.c1 = L.Convolution2D(3, 32, 5, 1, 2)
            self.c2 = L.Convolution2D(32, 64, 5, 1, 2)
            self.c3 = L.Convolution2D(64, 128, 5, 1, 2)
            self.c4 = L.Convolution2D(128, 256, 5, 1, 2)
            self.c5 = L.Convolution2D(256, 256, 5, 1, 2)
            self.c6 = L.Convolution2D(256, 256, 5, 1, 2)
            self.c7 = L.Convolution2D(256, 512, 4, 1, 0)

    def __call__(self, x):
        h = F.relu(self.c1(x))
        h = F.average_pooling_2d(h, 2, 2, 0)

        h = F.relu(self.c2(h))
        h = F.average_pooling_2d(h, 2, 2, 0)

        h = F.relu(self.c3(h))
        h = F.average_pooling_2d(h, 2, 2, 0)

        h = F.relu(self.c4(h))
        h = F.average_pooling_2d(h, 2, 2, 0)

        h = F.relu(self.c5(h))
        h = F.average_pooling_2d(h, 2, 2, 0)

        h = F.relu(self.c6(h))

        h = self.c7(h)

        return D.Normal(loc=h[:, :256], log_scale=h[:, 256:])


class Decoder_2(chainer.Chain):
    def __init__(self):
        super(Decoder_2, self).__init__()
        with self.init_scope():
            self.c1 = L.Convolution2D(256, 256, 4, 1, 3)
            self.c2 = L.Convolution2D(256, 256, 5, 1, 2)
            self.c3 = L.Convolution2D(256, 256, 5, 1, 2)
            self.c4 = L.Convolution2D(256, 256, 5, 1, 2)
            self.c5 = L.Convolution2D(256, 128, 5, 1, 2)
            self.c6 = L.Convolution2D(128, 64, 5, 1, 2)
            self.c7 = L.Convolution2D(64, 32, 5, 1, 2)
            self.c8 = L.Convolution2D(32, 3, 5, 1, 2)

    def __call__(self, z):
        k, b, c, h, w = z.shape
        z = F.reshape(z, (k*b, c, h, w))
        h = F.relu(self.c1(z))

        h = F.relu(self.c2(h))
        h = F.unpooling_2d(h, 2, 2, 0, outsize=(8, 8))

        h = F.relu(self.c3(h))
        h = F.unpooling_2d(h, 2, 2, 0, outsize=(16, 16))

        h = F.relu(self.c4(h))
        h = F.unpooling_2d(h, 2, 2, 0, outsize=(32, 32))

        h = F.relu(self.c5(h))
        h = F.unpooling_2d(h, 2, 2, 0, outsize=(64, 64))

        h = F.relu(self.c6(h))
        h = F.unpooling_2d(h, 2, 2, 0, outsize=(128, 128))

        h = F.relu(self.c7(h))
        x_hat = F.tanh(self.c8(h))

        return x_hat


class Prior(chainer.Link):

    def __init__(self):
        super(Prior, self).__init__()

        n_latent = 256
        self.loc = np.zeros(n_latent, np.float32)
        self.scale = np.ones(n_latent, np.float32)
        self.register_persistent('loc')
        self.register_persistent('scale')

    def forward(self):
        return D.Normal(self.loc, scale=self.scale)
