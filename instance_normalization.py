import numpy

import chainer
from chainer import configuration
from chainer import cuda
from chainer import functions
from chainer import initializers
from chainer import link
from chainer.utils import argument
from chainer import variable


class InstanceNormalization(link.Link):
    def __init__(self,
                 size,
                 decay=0.9,
                 eps=2e-5,
                 dtype=numpy.float32,
                 use_gamma=True,
                 use_beta=True,
                 initial_gamma=None,
                 initial_beta=None):
        super(InstanceNormalization, self).__init__()
        self.size = size
        self.N = 0
        self.register_persistent('N')
        self.decay = decay
        self.eps = eps

        with self.init_scope():
            if use_gamma:
                if initial_gamma is None:
                    initial_gamma = 1
                initial_gamma = initializers._get_initializer(initial_gamma)
                initial_gamma.dtype = dtype
                self.gamma = variable.Parameter(initial_gamma, size)
            if use_beta:
                if initial_beta is None:
                    initial_beta = 0
                initial_beta = initializers._get_initializer(initial_beta)
                initial_beta.dtype = dtype
                self.beta = variable.Parameter(initial_beta, size)

    def __call__(self, x, **kwargs):
        """__call__(self, x, finetune=False)
        Invokes the forward propagation of BatchNormalization.
        In training mode, the BatchNormalization computes moving averages of
        mean and variance for evaluation during training, and normalizes the
        input using batch statistics.
        .. warning::
           ``test`` argument is not supported anymore since v2.
           Instead, use ``chainer.using_config('train', False)``.
           See :func:`chainer.using_config`.
        Args:
            x (Variable): Input variable.
            finetune (bool): If it is in the training mode and ``finetune`` is
                ``True``, BatchNormalization runs in fine-tuning mode; it
                accumulates the input array to compute population statistics
                for normalization, and normalizes the input using batch
                statistics.
        """
        # check argument
        argument.check_unexpected_kwargs(
            kwargs,
            test='test argument is not supported anymore. '
            'Use chainer.using_config')
        finetune, = argument.parse_kwargs(kwargs, ('finetune', False))

        # reshape input x
        original_shape = x.shape
        batch_size, n_ch = original_shape[:2]
        new_shape = (1, batch_size * n_ch) + original_shape[2:]
        reshaped_x = functions.reshape(x, new_shape)

        if hasattr(self, 'gamma'):
            gamma = self.gamma
        else:
            with cuda.get_device_from_id(self._device_id):
                gamma = variable.Variable(
                    self.xp.ones(self.size, dtype=x.dtype))
        if hasattr(self, 'beta'):
            beta = self.beta
        else:
            with cuda.get_device_from_id(self._device_id):
                beta = variable.Variable(
                    self.xp.zeros(self.size, dtype=x.dtype))

        gamma = chainer.as_variable(self.xp.hstack([gamma.array] * batch_size))
        beta = chainer.as_variable(self.xp.hstack([beta.array] * batch_size))

        head_ndim = gamma.ndim + 1
        self.axis = (0, ) + tuple(range(head_ndim, reshaped_x.ndim))

        if configuration.config.train:
            if finetune:
                self.N += 1
                decay = 1. - 1. / self.N
            else:
                decay = self.decay

            ret = functions.batch_normalization(
                reshaped_x,
                gamma,
                beta,
                eps=self.eps,
                running_mean=None,
                running_var=None,
                decay=decay)
        else:
            # Use running average statistics or fine-tuned statistics.
            mean = reshaped_x.data.mean(axis=self.axis)
            var = reshaped_x.data.var(axis=self.axis)
            ret = functions.fixed_batch_normalization(reshaped_x, gamma, beta,
                                                      mean, var, self.eps)

        # ret is normalized input x
        return functions.reshape(ret, original_shape)
