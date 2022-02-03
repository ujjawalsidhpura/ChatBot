import tensorflow
import tflearn


def model_maker(training, output):

    tensorflow.reset_default_graph()

    net = tflearn.input_data(shape=[None, len(training[0])])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, len(output[0]), activation='softmax')
    net = tflearn.regression(net)

    model = tflearn.DNN(net)

    try:
        model.load('model.tflearn')

        return model

    except:
        model.fit(training, output, n_epoch=1000,
                  batch_size=8, show_metric=True)
        model.save('model.tflearn')

        return model
