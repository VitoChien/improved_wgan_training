import numpy as np
import os
import urllib
import gzip
import cPickle as pickle

def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo)
    fo.close()
    return dict['data'], dict['labels']

def Imagenet_generator(filenames, batch_size, data_dir):
    all_data = []
    all_labels = []
    for filename in filenames:
        img_path = os.path.join(data_dir)
        txt_path = os.path.join(data_dir,'list.txt')
        print "generating list"
        image_num = len(open(txt_path, 'rb').readlines())
        index = [ x for x in range(image_num)]

    #images = np.concatenate(all_data, axis=0)
    #labels = np.concatenate(all_labels, axis=0)

    def get_epoch():
        print "shuffling"
        rng_state = np.random.get_state()
        np.random.shuffle(index)
        print "generating batches"
        for i in xrange(len(index) / batch_size):
            yield (index[i * batch_size:(i + 1) * batch_size])

    return get_epoch


def load(batch_size, data_dir):
    return (
        Imagenet_generator(['train'], batch_size, data_dir),
        Imagenet_generator(['val'], batch_size, data_dir)
    )