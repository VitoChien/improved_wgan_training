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
        img_path = os.path.join(data_dir,'Data','CLS-LOC',filename)
        txt_path = os.path.join(data_dir,'Label',filename+'.txt')
        print "generating list"
        for line in open(txt_path, 'rb'):
            #print line.strip().split(' ')[0]
            img_path = os.path.join(img_path ,line.strip().split(' ')[0])
            labels = int(line.strip().split(' ')[1])
            all_data.append(img_path)
            all_labels.append(labels)

    #images = np.concatenate(all_data, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    def get_epoch():
        print "shuffling"
        rng_state = np.random.get_state()
        np.random.shuffle(all_data)
        np.random.set_state(rng_state)
        np.random.shuffle(labels)
        print "generating batches"
        for i in xrange(len(all_data) / batch_size):
            yield (all_data[i * batch_size:(i + 1) * batch_size], labels[i*batch_size:(i+1)*batch_size])

    return get_epoch


def load(batch_size, data_dir):
    return (
        Imagenet_generator(['train'], batch_size, data_dir),
        Imagenet_generator(['test'], batch_size, data_dir)
    )