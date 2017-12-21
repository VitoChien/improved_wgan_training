"""WGAN-GP ResNet for CIFAR-10"""

import os, sys
sys.path.append(os.getcwd())

import tflib as lib
import tflib.ops.linear
import tflib.ops.cond_batchnorm
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.save_images
import tflib.cifar10
import tflib.makeup
import tflib.inception_score
import tflib.plot

import numpy as np
import tensorflow as tf
#import sklearn.datasets

import time
import functools
import locale
locale.setlocale(locale.LC_ALL, '')

import linecache
import cv2
from scipy import misc

# Download CIFAR-10 (Python version) at
# https://www.cs.toronto.edu/~kriz/cifar.html and fill in the path to the
# extracted files here!
DATA_DIR = '/home/vito/DATASET/make_up/masked_v2/'
if len(DATA_DIR) == 0:
    raise Exception('Please specify path to data directory in gan_cifar.py!')

N_GPUS = 2
if N_GPUS not in [1,2]:
    raise Exception('Only 1 or 2 GPUs supported!')

BATCH_SIZE = 16 # Critic batch size
GEN_BS_MULTIPLE = 2 # Generator batch size, as a multiple of BATCH_SIZE
ITERS = 100000 # How many iterations to train for
DIM_G = 128 # Generator dimensionality
DIM_D = 128 # Critic dimensionality
NORMALIZATION_G = True # Use batchnorm in generator?
NORMALIZATION_D = False # Use batchnorm (or layernorm) in critic?
#OUTPUT_DIM = 3072 # Number of pixels in CIFAR10 (32*32*3)
OUTPUT_DIM = 49152 # Number of pixels in CIFAR10 (32*32*3)
LR = 2e-4 # Initial learning rate
DECAY = False # Whether to decay LR over learning
N_CRITIC = 5 # Critic steps per generator steps
INCEPTION_FREQUENCY = 100000 # How frequently to calculate Inception score

CONDITIONAL = True # Whether to train a conditional or unconditional model
ACGAN = True # If CONDITIONAL, whether to use ACGAN or "vanilla" conditioning
ACGAN_SCALE_D = 0.00000001 # How to scale the critic's ACGAN loss relative to WGAN loss
ACGAN_SCALE_G = 0.0001 # How to scale generator's ACGAN loss relative to WGAN loss
REC_SCALE = 0.00001

if CONDITIONAL and (not ACGAN) and (not NORMALIZATION_D):
    print "WARNING! Conditional model without normalization in D might be effectively unconditional!"

DEVICES = ['/gpu:{}'.format(i) for i in xrange(N_GPUS)]
if len(DEVICES) == 1: # Hack because the code assumes 2 GPUs
    DEVICES = [DEVICES[0], DEVICES[0]]

lib.print_model_settings(locals().copy())

def nonlinearity(x):
    return tf.nn.relu(x)

def Normalize(name, inputs,labels=None, is_training = False):
    """This is messy, but basically it chooses between batchnorm, layernorm, 
    their conditional variants, or nothing, depending on the value of `name` and
    the global hyperparam flags."""
    if not CONDITIONAL:
        labels = None
    if CONDITIONAL and ACGAN and ('Discriminator' in name):
        labels = None

    if ('Discriminator' in name) and NORMALIZATION_D:
        """
        if conditional_instance_normed:
            return lib.ops.cond_batchnorm.Batchnorm(name, [1, 2, 3], inputs, labels=labels, n_labels=1000,is_training=is_training)
        else:
            return lib.ops.layernorm.Layernorm(name,[1,2,3],inputs,labels=labels,n_labels=10)
        """
        #return lib.ops.batchnorm.Batchnorm(name,[0,2,3],inputs,fused=True)
        return lib.ops.batchnorm.Batchnorm(name,[0,2,3],inputs,fused=True)
    elif ('Generator' in name) and ('mid' in name) and NORMALIZATION_G:
        #print name
        return lib.ops.batchnorm.Batchnorm(name,[0,2,3],inputs,fused=True)
    elif ('Generator' in name) and NORMALIZATION_G:
        if labels is not None:
            #print("##################################################")
            return lib.ops.cond_batchnorm.Batchnorm(name,[0,2,3],inputs,labels=labels,n_labels=10)
        else:
            return lib.ops.batchnorm.Batchnorm(name,[0,2,3],inputs,fused=True)
    else:
        return inputs

def ConvMeanPool(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True, spectralnorm=False, update_collection = None):
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, inputs, he_init=he_init, biases=biases,
        spectralnorm = spectralnorm, update_collection = update_collection)
    output = tf.add_n([output[:,:,::2,::2], output[:,:,1::2,::2], output[:,:,::2,1::2], output[:,:,1::2,1::2]]) / 4.
    return output

def MeanPoolConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True, spectralnorm=False, update_collection = None):
    output = inputs
    output = tf.add_n([output[:,:,::2,::2], output[:,:,1::2,::2], output[:,:,::2,1::2], output[:,:,1::2,1::2]]) / 4.
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases,
            spectralnorm = spectralnorm, update_collection = update_collection)
    return output

def UpsampleConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True, spectralnorm=False, update_collection = None):
    output = inputs
    output = tf.concat([output, output, output, output], axis=1)
    output = tf.transpose(output, [0,2,3,1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0,3,1,2])
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases,
            spectralnorm = spectralnorm, update_collection = update_collection)
    return output

def ResidualBlock(name, input_dim, output_dim, filter_size, inputs, resample=None, no_dropout=False, labels=None,
                  update_collection = None,spectralnorm = False, is_training = None, he_init=True):
    """2
    resample: None, 'down', or 'up'
    """
    if resample=='down':
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim, spectralnorm=spectralnorm,
                                      update_collection=update_collection)
        conv_2        = functools.partial(ConvMeanPool, input_dim=input_dim, output_dim=output_dim, spectralnorm=spectralnorm,
                                      update_collection=update_collection)
        conv_shortcut = ConvMeanPool
    elif resample=='up':
        conv_1        = functools.partial(UpsampleConv, input_dim=input_dim, output_dim=output_dim, spectralnorm=spectralnorm, \
                                      update_collection=update_collection)
        conv_shortcut = UpsampleConv
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim, spectralnorm=spectralnorm, \
                                      update_collection=update_collection)
    elif resample==None:
        conv_shortcut = lib.ops.conv2d.Conv2D
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=output_dim, spectralnorm=spectralnorm, \
                                      update_collection=update_collection)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim, spectralnorm=spectralnorm, \
                                      update_collection=update_collection)
    else:
        raise Exception('invalid resample value')

    if output_dim==input_dim and resample==None:
        shortcut = inputs # Identity skip-connection
    else:
        shortcut = conv_shortcut(name+'.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1, he_init=False,\
                                 biases=True, inputs=inputs, spectralnorm=spectralnorm, update_collection=update_collection)

    output = inputs
    output = Normalize(name+'.N1', output, labels=labels, is_training = is_training)
    output = nonlinearity(output)
    output = conv_1(name+'.Conv1', filter_size=filter_size, inputs=output, he_init=he_init, biases=False, spectralnorm=spectralnorm, \
                                      update_collection=update_collection)
    output = Normalize(name+'.N2', output, labels=labels, is_training = is_training)
    output = nonlinearity(output)
    output = conv_2(name+'.Conv2', filter_size=filter_size, inputs=output, he_init=he_init, spectralnorm=spectralnorm, \
                                      update_collection=update_collection)

    return shortcut + output

def OptimizedResBlockDisc1(name, inputs, input_dim, output_dim, labels, update_collection = None, spectralnorm = False, is_training = None):
    conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=output_dim, spectralnorm=spectralnorm,
                                  update_collection=update_collection)
    conv_2        = functools.partial(ConvMeanPool, input_dim=output_dim, output_dim=output_dim, spectralnorm=spectralnorm,
                                      update_collection=update_collection)
    conv_shortcut = MeanPoolConv
    shortcut = conv_shortcut(name+'.Shortcut', input_dim, output_dim, spectralnorm=spectralnorm, filter_size=1,
                             update_collection=update_collection, inputs=inputs, he_init = False)

    output = inputs
    output = conv_1(name+'.Conv1', inputs=output, he_init=False, biases=False, spectralnorm=spectralnorm, filter_size=3,\
                                      update_collection=update_collection)
    output = Normalize(name+'.N1', output, labels=labels, is_training = is_training)
    output = nonlinearity(output)
    output = conv_2(name+'.Conv2', inputs=output, he_init=False, spectralnorm=spectralnorm, filter_size=3,\
                                      update_collection=update_collection)
    output = Normalize(name+'.N2', output, labels=labels, is_training = is_training)

    return shortcut + output

def Generator(input_img, labels, is_training = True):
    spectralnorm_flag = False
    with tf.variable_scope("Generator"):
        update_collection = None
        output = tf.reshape(input_img, [-1, 3, 128, 128])
        output = OptimizedResBlockDisc1('Generator.1', output, 3, 64, labels, update_collection = update_collection,
                                        is_training = is_training, spectralnorm = spectralnorm_flag)
        output = OptimizedResBlockDisc1('Generator.2', output, 64, 128, labels, update_collection = update_collection,
                                        is_training=is_training, spectralnorm = spectralnorm_flag)
        output = ResidualBlock('Generator.3', 128, 128, 3, output, resample=None, labels=labels, \
                               spectralnorm = spectralnorm_flag,update_collection = update_collection ,is_training=is_training)
        output = ResidualBlock('Generator.4', 128, 128, 3, output, resample=None, labels=labels, \
                               spectralnorm = spectralnorm_flag,update_collection = update_collection,is_training=is_training)
        output = ResidualBlock('Generator.5', 128, 128, 3, output, resample=None, labels=labels, \
                               spectralnorm = spectralnorm_flag,update_collection = update_collection,is_training=is_training)
        output = ResidualBlock('Generator.6', 128, 128, 3, output, resample=None, labels=labels, \
                               spectralnorm = spectralnorm_flag,update_collection = update_collection,is_training=is_training)
        output = ResidualBlock('Generator.7', 128, 128, 3, output, resample=None, labels=labels, \
                               spectralnorm = spectralnorm_flag,update_collection = update_collection,is_training=is_training)
        output = ResidualBlock('Generator.8', 128, 128, 3, output, resample=None, labels=labels, \
                               spectralnorm = spectralnorm_flag,update_collection = update_collection,is_training=is_training)
        output = ResidualBlock('Generator.9', 128, 64, 3, output, resample='up', labels=labels, spectralnorm = spectralnorm_flag, \
                               update_collection = update_collection)
        output = ResidualBlock('Generator.10', 64, 32, 3, output, resample='up', labels=labels, spectralnorm = spectralnorm_flag, \
                               update_collection = update_collection)
        #output = Normalize('Generator.OutputN', output, labels=labels)
        output = Normalize('Generator.mid', output, is_training)
        output = nonlinearity(output)
        output = lib.ops.conv2d.Conv2D('Generator.Output', 32, 3, 3, output, spectralnorm = spectralnorm_flag, \
                                       update_collection = update_collection)
        output = tf.tanh(output,name = 'Generator_tanh')
        return tf.reshape(output, [-1, OUTPUT_DIM])

def Discriminator(inputs, labels ,update_collection, is_training = False):
    spectralnorm_flag = True
    with tf.variable_scope("Discriminator"):
        #update_collection = tf.GraphKeys.UPDATE_OPS
        output = tf.reshape(inputs, [-1, 3, 128, 128])
        output = OptimizedResBlockDisc1('Discriminator.1', output, 3, 64, labels, update_collection = update_collection,
                                        is_training=is_training, spectralnorm = spectralnorm_flag)
        output = ResidualBlock('Discriminator.3', 64, 128, 3, output, resample='down', labels=labels, spectralnorm = spectralnorm_flag,\
                               update_collection = update_collection)
        output = ResidualBlock('Discriminator.4', 128, 256, 3, output, resample='down', labels=labels, spectralnorm = spectralnorm_flag,\
                               update_collection = update_collection)
        label_one_hot = tf.one_hot(labels, 10, name = 'Discriminator.onehot')
        #embed = lib.ops.linear.Linear('Discriminator.embed', 10, 128, label_one_hot, spectralnorm = spectralnorm_flag, \
                                     #update_collection = update_collection)
        embed = tf.reshape(label_one_hot, [-1, 10, 1, 1])
        embed_tiled = tf.tile(embed, [1, 1, 16, 16], name = 'Discriminator.embed_tile')  # shape (3, 1)
        output = tf.concat([output,embed_tiled] , axis=1, name = 'Discriminator.embed_concate')
        output = ResidualBlock('Discriminator.6', 256 + 10, 512, 3, output, resample='down', labels=labels, \
                               spectralnorm = spectralnorm_flag,update_collection = update_collection)
        output = ResidualBlock('Discriminator.7', 512, 1024, 3, output, resample='down', labels=labels, \
                               spectralnorm = spectralnorm_flag,update_collection = update_collection)
        output = ResidualBlock('Discriminator.8', 1024, 1024, 3, output, resample=None, labels=labels, \
                               spectralnorm = spectralnorm_flag,update_collection = update_collection)

        output = tf.reduce_sum(output, axis=[2,3], name = 'Discriminator.reduce_sum')
        output = nonlinearity(output)
        output_wgan = lib.ops.linear.Linear('Discriminator.Output', 1024, 1, output, spectralnorm = spectralnorm_flag,\
                                            update_collection = update_collection)
        output_wgan = tf.reshape(output_wgan, [-1])
        output_acgan = lib.ops.linear.Linear('Discriminator.ACGANOutput', 1024, 10, output)
        return output_wgan, output_acgan


with tf.Session() as session:
    _iteration = tf.placeholder(tf.int32, shape=None)

    if DECAY:
        decay = tf.maximum(0., 1.-(tf.cast(_iteration, tf.float32)/ITERS))
    else:
        decay = 1.

    real_data_int = tf.placeholder(tf.int32, shape=[BATCH_SIZE, OUTPUT_DIM])
    real_labels = tf.placeholder(tf.int32, shape=[BATCH_SIZE])

    real_data = tf.reshape(2*((tf.cast(real_data_int, tf.float32)/256.)-.5), [BATCH_SIZE, OUTPUT_DIM])
    real_data += tf.random_uniform(shape=[BATCH_SIZE,OUTPUT_DIM],minval=0.,maxval=1./128) # dequantize

    fake_data = Generator(real_data, real_labels, is_training = True)

    #Get loss for D
    disc_costs = []
    disc_acgan_costs = []
    disc_acgan_accs = []
    disc_acgan_fake_accs = []
    disc_real, disc_real_acgan = Discriminator(real_data, real_labels, update_collection=None)
    disc_fake, disc_fake_acgan = Discriminator(fake_data, real_labels, update_collection="NO_OPS")
    discriminator_loss_real = tf.reduce_mean(tf.maximum(0., 1. - disc_real))
    discriminator_loss_fake = tf.reduce_mean(tf.maximum(0., 1. + disc_fake))
    disc_costs.append(discriminator_loss_real + discriminator_loss_fake)
    #disc_real_acgan = tf.Print(disc_real_acgan,[disc_real_acgan],summarize=BATCH_SIZE*10)
    #real_labels = tf.Print(real_labels,[real_labels],summarize=BATCH_SIZE*10)
    disc_acgan_costs.append(tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=disc_real_acgan, labels=real_labels)
    ))

    disc_acgan_accs.append(tf.reduce_mean(
        tf.cast(
            tf.equal(
                tf.to_int32(tf.argmax(disc_real_acgan, dimension=1)),
                real_labels
            ),
            tf.float32
        )
    ))
    disc_acgan_fake_accs.append(tf.reduce_mean(
        tf.cast(
            tf.equal(
                tf.to_int32(tf.argmax(disc_fake_acgan, dimension=1)),
                real_labels
            ),
            tf.float32
        )
    ))

    disc_r_and_f = tf.add_n(disc_costs)
    disc_acgan = tf.add_n(disc_acgan_costs)
    #disc_cost = disc_r_and_f + (ACGAN_SCALE_D*disc_acgan)
    disc_cost = disc_r_and_f

    disc_acgan_acc = tf.add_n(disc_acgan_accs)
    disc_acgan_fake_acc = tf.add_n(disc_acgan_fake_accs)

    disc_params = lib.params_with_name('Discriminator.')


    #Get loss for G
    ge_costs = []
    gen_costs = []
    gen_acgan_costs = []
    rec_costs = []

    n_samples = BATCH_SIZE
    fake_labels = tf.cast(tf.random_uniform([n_samples])*10, tf.int32)
    fake_img = Generator(real_data,fake_labels)
    disc_fake, disc_fake_acgan = Discriminator(fake_img, fake_labels, update_collection="NO_OPS")
    rec_img = Generator(fake_img, real_labels)
    ge_costs.append(-tf.reduce_mean(disc_fake))

    gen_acgan_costs.append(tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=disc_fake_acgan, labels=fake_labels)
    ))
    rec_costs.append(tf.reduce_mean(tf.reduce_sum(tf.abs(real_data - rec_img))))


    ge_cost = tf.add_n(ge_costs)
    gen_acgan = tf.add_n(gen_acgan_costs)
    rec_cost = tf.add_n(rec_costs)
    gen_cost = ge_cost
    #gen_cost += ACGAN_SCALE_G * gen_acgan
    gen_cost += REC_SCALE * rec_cost

    gen_opt = tf.train.AdamOptimizer(learning_rate=LR*decay, beta1=0., beta2=0.9)
    disc_opt = tf.train.AdamOptimizer(learning_rate=LR*decay, beta1=0., beta2=0.9)
    gen_gv = gen_opt.compute_gradients(gen_cost, var_list=lib.params_with_name('Generator'))
    disc_gv = disc_opt.compute_gradients(disc_cost, var_list=disc_params)
    gen_train_op = gen_opt.apply_gradients(gen_gv)
    disc_train_op = disc_opt.apply_gradients(disc_gv)

    train_gen, dev_gen = lib.makeup.load(BATCH_SIZE, DATA_DIR)

    def inf_train_gen():
        while True:
            for index_list in train_gen():
                all_images = []
                all_labels = []
                for index in index_list:
                    line = linecache.getline(os.path.join(DATA_DIR,'list.txt'),index+1)
                    line = line.strip()
                    img = cv2.imread(os.path.join(DATA_DIR,line.split(' ')[0]))
                    img = cv2.resize(img, (128, 128))
                    img = img.transpose((2,0,1))
                    all_images.append(img.reshape((1,-1)))
                    all_labels.append(int(line.split(' ')[1]))
                all_labels = np.array(all_labels)
                all_labels = all_labels[np.newaxis]
                images = np.concatenate(all_images, axis=0)
                labels = np.concatenate(all_labels, axis=0)
                yield images,labels


    dev_gen = lib.makeup.load_test(10, DATA_DIR)
    def inf_dev_gen():
        while True:
            for index_list in dev_gen():
                all_images = []
                all_labels = []
                for index in index_list:
                    line = linecache.getline(os.path.join(DATA_DIR,'list.txt'),index+1)
                    line = line.strip()
                    img = cv2.imread(os.path.join(DATA_DIR,line.split(' ')[0]))
                    img = cv2.resize(img, (128, 128))
                    img = img.transpose((2,0,1))
                    for i in range(10):
                        all_images.append(img.reshape((1,-1)))
                    for i in range(10):
                        all_labels.append(int(line.split(' ')[1]))
                all_labels = np.array(all_labels)
                all_labels = all_labels[np.newaxis]
                images = np.concatenate(all_images, axis=0)
                labels = np.concatenate(all_labels, axis=0)
                yield images,labels

    # Function for generating samples
    frame_i = [0]
    fixed_labels = tf.constant(np.array([0,1,2,3,4,5,6,7,8,9]*10,dtype='int32'))
    gen_dev = inf_dev_gen()
    img_test, label_tets = gen_dev.next()
    #print(img_test.shape)
    img_test = tf.reshape(2 * ((tf.cast(img_test, tf.float32) / 256.) - .5), [100, OUTPUT_DIM])
    fixed_noise_samples = Generator(img_test, fixed_labels)
    def generate_image(frame, true_dist):
        samples = session.run(fixed_noise_samples)
        samples = ((samples+1.)*(255./2)).astype('int32')
        lib.save_images.save_images(samples.reshape((100, 3, 128, 128)), 'makeup_mid', 'samples_{}.png'.format(frame))

    # Function for calculating inception score
    """
    fake_labels_100 = tf.cast(tf.random_uniform([100])*10, tf.int32)
    samples_100 = Generator(img_test, fake_labels_100)
    def get_inception_score(n):
        all_samples = []
        for i in xrange(n/100):
            all_samples.append(session.run(samples_100))
        all_samples = np.concatenate(all_samples, axis=0)
        all_samples = ((all_samples+1.)*(255.99/2)).astype('int32')
        all_samples = all_samples.reshape((-1, 3, 128, 128)).transpose(0,2,3,1)
        return lib.inception_score.get_inception_score(list(all_samples))
    """


    for name,grads_and_vars in [('G', gen_gv), ('D', disc_gv)]:
        print "{} Params:".format(name)
        total_param_count = 0
        for g, v in grads_and_vars:
            shape = v.get_shape()
            shape_str = ",".join([str(x) for x in v.get_shape()])

            param_count = 1
            for dim in shape:
                param_count *= int(dim)
            total_param_count += param_count

            if g == None:
                print "\t{} ({}) [no grad!]".format(v.name, shape_str)
            else:
                print "\t{} ({})".format(v.name, shape_str)
        print "Total param count: {}".format(
            locale.format("%d", total_param_count, grouping=True)
        )

    session.run(tf.initialize_all_variables())

    gen = inf_train_gen()

    saver = tf.train.Saver()
    is_start_iteration = True
    for iteration in xrange(ITERS):
        if (iteration + 1) % 2000 == 0 and not is_start_iteration:
            saver.save(session, 'snapshots/model.ckpt', global_step=iteration)
        start_time = time.time()

        if iteration > 0:
            _data,_labels = gen.next()
            _ = session.run([gen_train_op], feed_dict={real_data_int: _data, real_labels:_labels, _iteration:iteration})

        for i in xrange(N_CRITIC):
            _data,_labels = gen.next()
            _disc_cost, _disc_r_and_f, _disc_acgan, _disc_acgan_acc, _disc_acgan_fake_acc, _ge_cost, _gen_cost, _gen_acgan, _rec_cost, _disc_real_acgan,_ = \
                session.run([disc_cost, disc_r_and_f, disc_acgan, disc_acgan_acc, disc_acgan_fake_acc, ge_cost, gen_cost, gen_acgan,
                             rec_cost, disc_real_acgan, disc_train_op],
                feed_dict={real_data_int: _data, real_labels:_labels, _iteration:iteration})
            """
            _disc_cost, _disc_r_and_f, _disc_acgan, _disc_acgan_acc, _disc_acgan_fake_acc, _ge_cost, _gen_cost, _gen_acgan, _rec_cost, _ = \
                session.run([disc_cost, disc_r_and_f, disc_acgan, disc_acgan_acc, disc_acgan_fake_acc, ge_cost, gen_cost, gen_acgan,
                             rec_cost, disc_train_op],
                feed_dict={real_data_int: _data, real_labels:_labels, _iteration:iteration})
            """
        lib.plot.plot('disc_r_and_f', _disc_r_and_f)
        lib.plot.plot('disc_acgan', _disc_acgan)
        lib.plot.plot('dis_cost', _disc_cost)
        lib.plot.plot('ge_cost', _ge_cost)
        lib.plot.plot('gen_acgan', _gen_acgan)
        lib.plot.plot('rec_cost', _rec_cost)
        lib.plot.plot('gen_cost', _gen_cost)
        lib.plot.plot('acc_real', _disc_acgan_acc)
        lib.plot.plot('acc_fake', _disc_acgan_fake_acc)
        lib.plot.plot('time', time.time() - start_time)
        '''
        if iteration % INCEPTION_FREQUENCY == INCEPTION_FREQUENCY-1:
            inception_score = get_inception_score(50000)
            lib.plot.plot('inception_50k', inception_score[0])
            lib.plot.plot('inception_50k_std', inception_score[1])
        '''
        # Calculate dev loss and generate samples every 100 iters
        """
        if iteration % 100 == 99:
            dev_disc_costs = []
            for index_list in dev_gen():
                all_images = []
                all_labels = []
                for index in index_list:
                    line = linecache.getline(os.path.join(DATA_DIR,'list.txt'),index+1)
                    line = line.strip()
                    img = cv2.imread(os.path.join(DATA_DIR,line.split(' ')[0]))
                    img = cv2.resize(img, (128, 128))
                    img = img.transpose((2,0,1))
                    # print img.reshape((1,-1)).shape
                    all_images.append(img.reshape((1, -1)))
                    all_labels.append(int(line.split(' ')[1]))
                all_labels = np.array(all_labels)
                all_labels = all_labels[np.newaxis]
                images = np.concatenate(all_images, axis=0)
                labels = np.concatenate(all_labels, axis=0)
                _dev_disc_cost = session.run([disc_cost], feed_dict={all_real_data_int: images,all_real_labels:_labels})
                dev_disc_costs.append(_dev_disc_cost)
            lib.plot.plot('dev_cost', np.mean(dev_disc_costs))

            generate_image(iteration, _data)
        """

        def display_imgs(prefix, frame, true_dist):
            # print true_dist
            img = (true_dist).astype('int32')
            lib.save_images.save_images(img.reshape((-1, 3, 128, 128)), 'makeup_mid', prefix + '_samples_{}.png'.format(frame))

        if iteration % 10 == 0:
            generate_image(iteration, _data)
            display_imgs('train_data', iteration, _data)

        if (iteration < 500) or (iteration % 1000 == 999):
            lib.plot.flush()

        lib.plot.tick()
        is_start_iteration = False