import paddle.fluid as fluid
import numpy as np
import paddle
class SimpleImgConvPool(fluid.dygraph.Layer):
    def __init__(self,
                 name_scope,
                 num_filters,
                 filter_size,
                 pool_size,
                 pool_stride,
                 pool_padding=0,
                 pool_type='max',
                 global_pooling=False,
                 conv_stride=1,
                 conv_padding=0,
                 conv_dilation=1,
                 conv_groups=1,
                 act=None,
                 use_cudnn=False,
                 param_attr=None,
                 bias_attr=None):
        super(SimpleImgConvPool, self).__init__(name_scope)

        self._conv2d = fluid.dygraph.Conv2D(
            self.full_name(),
            num_filters=num_filters,
            filter_size=filter_size,
            stride=conv_stride,
            padding=conv_padding,
            dilation=conv_dilation,
            groups=conv_groups,
            param_attr=param_attr,
            bias_attr=bias_attr,
            act=act,
            use_cudnn=use_cudnn)

        self._pool2d = fluid.dygraph.Pool2D(
            self.full_name(),
            pool_size=pool_size,
            pool_type=pool_type,
            pool_stride=pool_stride,
            pool_padding=pool_padding,
            global_pooling=global_pooling,
            use_cudnn=use_cudnn)

    def forward(self, inputs):
        x = self._conv2d(inputs)
        x = self._pool2d(x)
        return x


class MNIST(fluid.dygraph.Layer):
    def __init__(self, name_scope):
        super(MNIST, self).__init__(name_scope)

        self._simple_img_conv_pool_1 = SimpleImgConvPool(
            self.full_name(), 20, 5, 2, 2, act="relu")

        self._simple_img_conv_pool_2 = SimpleImgConvPool(
            self.full_name(), 50, 5, 2, 2, act="relu")

        pool_2_shape = 50 * 4 * 4
        SIZE = 10
        scale = (2.0 / (pool_2_shape**2 * SIZE))**0.5
        self._fc = fluid.dygraph.FC(self.full_name(),
                                    10,
                                    param_attr=fluid.param_attr.ParamAttr(
                                        initializer=fluid.initializer.NormalInitializer(
                                            loc=0.0, scale=scale)),
                                    act="softmax")

    def forward(self, inputs,label=None):
        x = self._simple_img_conv_pool_1(inputs)
        x = self._simple_img_conv_pool_2(x)
        x = self._fc(x)
        if label != None:
            acc = fluid.layers.accuracy(input=x, label=label)
            return x, acc
        else:
            return x


epoch_num = 1
BATCH_SIZE = 64
exe = fluid.Executor(fluid.CPUPlace())

mnist = MNIST("mnist")
sgd = fluid.optimizer.SGDOptimizer(learning_rate=1e-3)
train_reader = paddle.batch(
    paddle.dataset.mnist.train(), batch_size=BATCH_SIZE, drop_last=True)
img = fluid.layers.data(
    name='pixel', shape=[1, 28, 28], dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='int64')
#img_tmp = {"0":img}
cost = mnist(img)
loss = fluid.layers.cross_entropy(cost, label)
avg_loss = fluid.layers.mean(loss)
sgd.minimize(avg_loss)

out = exe.run(fluid.default_startup_program())

for epoch in range(epoch_num):
    for batch_id, data in enumerate(train_reader()):
        static_x_data = np.array(
            [x[0].reshape(1, 28, 28)
             for x in data]).astype('float32')
        y_data = np.array(
            [x[1] for x in data]).astype('int64').reshape([BATCH_SIZE, 1])
        ins = {'0':static_x_data,'1':y_data}
        fetch_list = [avg_loss.name]
        out = exe.run(
            fluid.default_main_program(),
            feed={"pixel": static_x_data,
                'label':y_data,
                  },
            fetch_list=fetch_list)

        static_out = out[0]

        if batch_id % 100 == 0 and batch_id is not 0:
            print("epoch: {}, batch_id: {}, loss: {}".format(epoch, batch_id, static_out))
