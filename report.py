from collections import OrderedDict
import json
import os


class report():
    def __init__(self, save_path):
        self.num_iter = 0
        self.log_file = os.path.join(save_path, "log")

        if os.path.isfile(self.log_file):
            os.remove(self.log_file)

        log_txt = ""
        key = ["epoch", "iter", "REC_train", "KL_train", "REC_test", "KL_test"]
        for k in key:
            log_txt += (k + "\t")
        print(log_txt)

    def __call__(self, epoch):
        log_txt = "{}\t{}\t".format(epoch, self.num_iter)
        log = OrderedDict()

        log["loss_rec_train"] = self.loss_rec_train / self.num_data_train
        log["loss_kl_train"] = self.loss_kl_train / self.num_data_train
        log["loss_rec_test"] = self.loss_rec_test / self.num_data_test
        log["loss_kl_test"] = self.loss_kl_test / self.num_data_test

        for i in log:
            log_txt += ("{:.6f}\t".format(log[i]))
        print(log_txt)

        log["epoch"] = epoch
        log["iter"] = self.num_iter

        with open(self.log_file, "a") as f:
            json.dump(log, f, indent=2)

    def init_log(self):
        self.num_data_train = 0
        self.loss_rec_train = 0
        self.loss_kl_train = 0
        self.num_data_test = 0
        self.loss_rec_test = 0
        self.loss_kl_test = 0
