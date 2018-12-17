import data
import chainer
import chainer.functions as F
from report import report
from tqdm import tqdm
from chainer import serializers
import os


class Trainer():
    def __init__(self, args, **kwargs):
        self.epochs = args.epochs
        self.k = args.k
        self.num_snapshot = args.num_snap
        self.batch_size = args.batch_size
        self.save_path = args.save_path
        self.beta = args.beta
        self.iter_train, self.iter_test = kwargs.pop("iterator")
        self.encoder, self.decoder, self.prior = kwargs.pop("models")
        self.opt_encoder, self.opt_decoder = kwargs.pop("optimizers")
        self.N = kwargs.pop("N")
        self.N_test = kwargs.pop("N_test")

        self.reporter = report(self.save_path)
        self.reporter.init_log()
        self.xp = self.encoder.xp

    def __call__(self):

        for e in range(1, self.epochs+1):
            bar = tqdm(desc="Training", total=self.N, leave=False)
            for files in self.iter_train:
                self.update(e, files)
                self.reporter.num_iter += 1
                self.reporter.num_data_train += self.batch_size
                bar.set_description("iter: {}, Rec: {:.6f}, KL: {:.6f}".format(
                    self.reporter.num_iter,
                    self.reporter.loss_rec_train / self.reporter.num_data_train,
                    self.reporter.loss_kl_train / self.reporter.num_data_train),
                    refresh=False)
                if self.reporter.num_iter % self.num_snapshot == 0:
                    self.snapshot(e)
                bar.update()
            bar.close()

            bar = tqdm(desc="Test", total=self.N_test, leave=False)
            with chainer.using_config('train', False):
                for files in self.iter_test:
                    self.update(e, files)
                    self.reporter.num_data_test += self.batch_size
                    bar.update()
            bar.close()

            self.reporter(e)
            self.reporter.init_log()

            self.iter_train.reset()
            self.iter_test.reset()

    def update(self, epoch, files):
        x = data.load(files)
        x = self.xp.asarray(x)
        x_v = chainer.Variable(x)

        loss_rec, loss_kl = self.cal_loss(x_v)
        loss = loss_rec + self.beta * loss_kl

        if chainer.config.train:
            self.encoder.cleargrads()
            self.decoder.cleargrads()
            loss.backward()
            self.opt_encoder.update()
            self.opt_decoder.update()
            loss.unchain_backward()

            self.reporter.loss_rec_train += float(loss_rec.data * self.batch_size)
            self.reporter.loss_kl_train += float(loss_kl.data * self.batch_size)
        else:
            self.reporter.loss_rec_test += float(loss_rec.data * self.batch_size)
            self.reporter.loss_kl_test += float(loss_kl.data * self.batch_size)

    def cal_loss(self, x):
        q_z = self.encoder(x)
        z = q_z.sample(self.k)
        x_hat = self.decoder(z)
        loss_rec = F.mean_squared_error(x_hat, x)
        loss_kl = F.mean(F.sum(chainer.kl_divergence(q_z, self.prior()), axis=-1))
        return loss_rec, loss_kl

    def snapshot(self, epoch):
        path = os.path.join(self.save_path, "model")
        if not os.path.exists(path):
            os.makedirs(path)

        serializers.save_npz(os.path.join(path, "epoch{}_iter{}.enc".format(epoch, self.reporter.num_iter)), self.encoder)
        serializers.save_npz(os.path.join(path, "epoch{}_iter{}.dec".format(epoch, self.reporter.num_iter)), self.decoder)
        serializers.save_npz(os.path.join(path, "epoch{}_iter{}.opt_enc".format(epoch, self.reporter.num_iter)), self.opt_encoder)
        serializers.save_npz(os.path.join(path, "epoch{}_iter{}.opt_dec".format(epoch, self.reporter.num_iter)), self.opt_decoder)

    # def _update(self, loss, model, opt, unchain=True):
    #     model.cleargrads()
    #     loss.backward()
    #     opt.update()
    #     if unchain:
    #         loss.unchain_backward()
