import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, Input, multiply, Flatten, dot, GaussianNoise, LeakyReLU, Activation
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.activations import sigmoid
from evaluate import evaluate_model
from dataset import Dataset
from time import time
import math
import argparse
import numpy.matlib
import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"

#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run PURE.")
    parser.add_argument('--path', nargs='?', default='Data/ml-1m_rand10/', # modify it to 'Data/yelp_rand5/' to run Yelp
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-1m', # modify it to 'yelp' to run Yelp
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=100, # modify it to 200 to run Yelp
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=128, # modify it to 512 to run Yelp
                        help='Batch size.')
    parser.add_argument('--num_factors', type=int, default=8, # modify it to 16 to run Yelp
                        help='Embedding size.')
    parser.add_argument('--regs', nargs='?', default='[0.0, 0.0]',
                        help="Regularization for user and item embeddings.")
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    parser.add_argument('--pos_ratio', type=int, default=0.00001, # modify it to 0.000001 to run Yelp
                        help='the ratio of the positive data.')
    parser.add_argument('--pretrain', nargs='?', default='Pretrain/ml-1m_PUNCF_8_1602276229.h5') # modify it to 'Pretrain/yelp_PUNCF_16_1602648580.h5' to run Yelp

    return parser.parse_args()

def get_discriminator(num_users, num_items, latent_dim):
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')
    user_fake  = Input(shape=(latent_dim,), dtype='float32', name='user_fake')
    item_fake  = Input(shape=(latent_dim,), dtype='float32', name='item_fake')
    pos_mask   = Input(shape=(latent_dim,), dtype='float32', name='pos_mask')
    fake1_mask = Input(shape=(latent_dim,), dtype='float32', name='fake1_mask')
    fake2_mask = Input(shape=(latent_dim,), dtype='float32', name='fake2_mask')

    MF_Embedding_User = Embedding(input_dim=num_users, output_dim=latent_dim, name='user_embedding',
                                  embeddings_initializer='lecun_normal', embeddings_regularizer=l2(0), input_length=1)
    MF_Embedding_Item = Embedding(input_dim=num_items, output_dim=latent_dim, name='item_embedding',
                                  embeddings_initializer='lecun_normal', embeddings_regularizer=l2(0), input_length=1)
    user_latent = Flatten()(MF_Embedding_User(user_input))
    item_latent = Flatten()(MF_Embedding_Item(item_input))

    # Element-wise product of user and item embeddings
    product_pos = multiply([user_latent, item_latent, pos_mask])
    product_fake1 = multiply([user_latent, item_fake, fake1_mask])
    product_fake2 = multiply([user_fake, item_latent, fake2_mask])
    product_all = product_pos + product_fake1 + product_fake2

    # Shared final prediction layer
    fc = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name='fc', use_bias=True)
    prediction = fc(product_all)

    model = Model(inputs=[user_input, item_input, user_fake, item_fake, pos_mask, fake1_mask, fake2_mask],
                  outputs=[prediction, user_latent, item_latent])
    return model

def get_generator(latent_dim, std):
    # Input variables
    emb_user_dis = Input(shape=(latent_dim,), dtype='float32', name='emb_user_dis')
    emb_item_dis = Input(shape=(latent_dim,), dtype='float32', name='emb_item_dis')
    dense_dis = Input(shape=(latent_dim+1,), dtype='float32', name='fc_dis') # extra 1 dimension is bias

    noise = GaussianNoise(std)
    noise_4_user = noise(K.zeros_like(emb_user_dis))
    noise_4_item = noise(K.zeros_like(emb_item_dis))

    mlp_u = Dense(16, activation='relu', kernel_initializer='lecun_normal', name='user_mlp_1', use_bias=True)
    mlp2_u = Dense(latent_dim, activation='relu', kernel_initializer='lecun_normal', name='user_mlp_2', use_bias=True)
    mlp_i = Dense(16, activation='relu', kernel_initializer='lecun_normal', name='item_mlp_1', use_bias=True)
    mlp2_i = Dense(latent_dim, activation='relu', kernel_initializer='lecun_normal', name='item_mlp_2', use_bias=True)
    fake_emb_user = mlp2_u(mlp_u(noise_4_user))
    fake_emb_item = mlp2_i(mlp_i(noise_4_item))

    # Element-wise product of true embedding (from discriminator) and fake embedding
    predict_vector_u2i = multiply([emb_user_dis, fake_emb_item])
    predict_vector_i2u = multiply([emb_item_dis, fake_emb_user])

    # final prediction layer (dense layer from discriminator)
    prediction_u2i = sigmoid(dot([predict_vector_u2i, dense_dis[:,:latent_dim]], axes=1)+dense_dis[:,-1:])
    prediction_i2u = sigmoid(dot([predict_vector_i2u, dense_dis[:,:latent_dim]], axes=1)+dense_dis[:,-1:])

    # model abstract input and out flowd
    model = Model(inputs=[emb_user_dis, emb_item_dis, dense_dis],
                  outputs=[prediction_u2i, prediction_i2u, fake_emb_user, fake_emb_item])
    return model


def get_train_instances_discriminator(train, generator, discriminator, latent_dim, pos_ratio, num_negatives):
    user_input, item_input, labels, sample_weights = [], [], [], []
    user_fake_emb, item_fake_emb = [], []
    pos_mask, fake1_mask, fake2_mask = [], [], []
    all_ones = [1 for x in range(latent_dim)]
    all_zeros = [0 for x in range(latent_dim)]

    # Part 1: positive instance (positive risk)
    for (u, i) in train.keys():
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        sample_weights.append(pos_ratio)
        user_fake_emb.append(all_zeros)
        item_fake_emb.append(all_zeros)
        pos_mask.append(all_ones)
        fake1_mask.append(all_zeros)
        fake2_mask.append(all_zeros)

        # Part 2: positive instance (negative risk)
        user_input.append(u)
        item_input.append(i)
        labels.append(0)
        sample_weights.append(-1 * pos_ratio)
        user_fake_emb.append(all_zeros)
        item_fake_emb.append(all_zeros)
        pos_mask.append(all_ones)
        fake1_mask.append(all_zeros)
        fake2_mask.append(all_zeros)

    # Part 2: unlabelled instances (negative risk)
    for (u, i) in train.keys():
        for t in range(round(num_negatives/2)):
            j = np.random.randint(num_items)
            while (u, j) in train:
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
            sample_weights.append(1)
            user_fake_emb.append(all_zeros)
            item_fake_emb.append(all_zeros)
            pos_mask.append(all_ones)
            fake1_mask.append(all_zeros)
            fake2_mask.append(all_zeros)

    # part 3: instances of generator (negative risk)
    u_input, i_input, l_input, w_input = [],[],[],[]

    ui_keys = list(train.keys())
    # random.shuffle(ui_keys)
    for (u,i) in ui_keys:
        for t in range(round(num_negatives/2)):
            j = np.random.randint(num_items)
            while (u, j) in train:
                j = np.random.randint(num_items)
            u_input.append(u)
            i_input.append(j)
            l_input.append(0)
            w_input.append(1)

    dis_fc_w = discriminator.get_layer('fc').get_weights()[0]
    dis_fc_b = discriminator.get_layer('fc').get_weights()[1]
    dis_fc = np.append(dis_fc_w, dis_fc_b).reshape(-1, latent_dim + 1)

    # batch_size for generator
    batch_size = np.shape(u_input)[0]

    emb_size = (batch_size, latent_dim)
    _, user_emb_dis, item_emb_dis = discriminator.predict(
        [np.array(u_input), np.array(i_input), np.zeros(emb_size), np.zeros(emb_size), np.zeros(emb_size), np.zeros(emb_size), np.zeros(emb_size),
         ], batch_size=batch_size, verbose=0)

    _, _, fake_emb_user, fake_emb_item = generator.predict(
        [user_emb_dis, item_emb_dis, np.matlib.repmat(dis_fc, batch_size, 1)], batch_size=batch_size, verbose=0)

    # add up data
    user_input.extend(u_input * 2)
    item_input.extend(i_input * 2)
    labels.extend(l_input * 2)
    sample_weights.extend(w_input * 2)
    user_fake_emb.extend(fake_emb_user.tolist() * 2)
    item_fake_emb.extend(fake_emb_item.tolist() * 2)

    pos_mask.extend([all_zeros for x in range(batch_size)])
    pos_mask.extend([all_zeros for x in range(batch_size)])
    fake1_mask.extend([all_ones for x in range(batch_size)])
    fake1_mask.extend([all_zeros for x in range(batch_size)])
    fake2_mask.extend([all_zeros for x in range(batch_size)])
    fake2_mask.extend([all_ones for x in range(batch_size)])

    return np.array(user_input), np.array(item_input), np.array(labels), np.array(sample_weights), \
           np.array(user_fake_emb).reshape(-1, latent_dim), np.array(item_fake_emb).reshape(-1, latent_dim), \
           np.array(pos_mask).reshape(-1, latent_dim), np.array(fake1_mask).reshape(-1, latent_dim), \
           np.array(fake2_mask).reshape(-1, latent_dim)

def get_train_instances_generator(train, discriminator, latent_dim, num_negatives):
    user_input, item_input, labels = [], [], []

    ui_keys = list(train.keys())
    # random.shuffle(ui_keys)
    for (u, i) in ui_keys:
        user_input.append(u)
        item_input.append(i)
        labels.append(1)

        for t in range(num_negatives):
            # unlabelled instances
            j = np.random.randint(num_items)
            while (u, j) in train:
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(1)

    batch_size = len(user_input)

    dis_fc_w = discriminator.get_layer('fc').get_weights()[0]
    dis_fc_b = discriminator.get_layer('fc').get_weights()[1]
    dis_fc = np.append(dis_fc_w, dis_fc_b).reshape(-1, latent_dim + 1)

    emb_size = (batch_size, latent_dim)
    _, user_emb_dis, item_emb_dis = discriminator.predict(
        [np.array(user_input), np.array(item_input), np.zeros(emb_size), np.zeros(emb_size), np.zeros(emb_size), np.zeros(emb_size), np.zeros(emb_size),
         ], batch_size=batch_size, verbose=0)

    return np.array(labels), user_emb_dis, item_emb_dis, np.matlib.repmat(dis_fc, batch_size, 1)

if __name__ == '__main__':
    np.random.seed(70)
    tf.set_random_seed(3)

    args = parse_args()
    num_factors = args.num_factors
    regs = eval(args.regs)

    pos_ratio = args.pos_ratio
    assert(pos_ratio < 0.4)
    num_negatives = math.ceil(1.0 / (1 - 2*pos_ratio)**2)

    learner = args.learner
    learning_rate = args.lr
    epochs = args.epochs
    batch_size = args.batch_size
    verbose = args.verbose

    evaluation_threads = 1
    print("# of negative sampling instances is:", num_negatives)
    print("Input arguments: %s" % (args))
    model_out_file = 'Pretrain/%s_PURE_%d_%d.h5' % (args.dataset, num_factors, time())
    embeddings_out_file = 'Embeddings/embeddings_%s_PURE_%d_%d.pkl' % (args.dataset, num_factors, time())
    log_name = 'Log/log_%s_PURE_%d_%d.txt' % (args.dataset, num_factors, time())
    log_results = open(log_name, 'w')

    # Loading data
    t1 = time()
    dataset = Dataset(args.path + args.dataset)
    train, trainRatings, testRatings, testNegatives = dataset.trainMatrix, dataset.trainRatings, dataset.testRatings, dataset.testNegatives

    num_users, num_items = train.shape
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d"
          % (time() - t1, num_users, num_items, train.nnz, len(testRatings)))

    # Build model
    discriminator = get_discriminator(num_users, num_items, num_factors)
    generator = get_generator( num_factors, std=0.01)

    # initialize D with pre-train
    d_pretrain = get_discriminator(num_users, num_items, num_factors)
    d_pretrain.load_weights(args.pretrain)
    user_embedding = d_pretrain.get_layer('user_embedding').get_weights()
    item_embedding = d_pretrain.get_layer('item_embedding').get_weights()
    discriminator.get_layer('user_embedding').set_weights(user_embedding)
    discriminator.get_layer('item_embedding').set_weights(item_embedding)
    fc_weights = d_pretrain.get_layer('fc').get_weights()
    discriminator.get_layer('fc').set_weights(fc_weights)

    if learner.lower() == "adam":
        discriminator.compile(optimizer=Adam(lr=learning_rate), loss={'fc':'binary_crossentropy'})
        generator.compile(optimizer=Adam(lr=learning_rate),
                          loss={'tf_op_layer_Sigmoid': 'binary_crossentropy', 'tf_op_layer_Sigmoid_1': 'binary_crossentropy'},
                          loss_weights={'tf_op_layer_Sigmoid': 1.0, 'tf_op_layer_Sigmoid_1': 1.0})
    else:
        discriminator.compile(optimizer=SGD(lr=learning_rate), loss='binary_crossentropy')
        generator.compile(optimizer=SGD(lr=learning_rate),
                          loss={'tf_op_layer_Sigmoid': 'binary_crossentropy', 'tf_op_layer_Sigmoid_1': 'binary_crossentropy'},
                          loss_weights={'tf_op_layer_Sigmoid': 1.0, 'tf_op_layer_Sigmoid_1': 1.0})
    # print(discriminator.summary())
    # print(generator.summary())

    # Init performance
    t1 = time()
    results = evaluate_model(discriminator, trainRatings, testRatings, testNegatives, num_items, num_factors)

    print('Init: P@3 = %.4f, P@5 = %.4f, P@10 = %.4f, NDCG@3 = %.4f, NDCG@5 = %.4f, NDCG@10 = %.4f\t [%.1f s]'
                  % (results[0], results[1], results[2], results[3], results[4], results[5], time() - t1))

    # Train model
    best_p5, best_ndcg5, best_iter = results[1], results[4], -1
    for epoch in range(epochs):
        t1 = time()
        # Training D
        for d_epoch in range(1):
            user_input, item_input, labels, sample_weights, user_fake_emb, item_fake_emb, pos_mask, fake1_mask, fake2_mask = get_train_instances_discriminator(
                train, generator, discriminator, num_factors, pos_ratio, num_negatives)

            hist_dis = discriminator.fit(
                [user_input, item_input, user_fake_emb, item_fake_emb, pos_mask, fake1_mask, fake2_mask], # input
                [labels],
                batch_size=batch_size,
                sample_weight=sample_weights,
                epochs=1, verbose=0, shuffle=True)
            print('Discriminator loss: %.3f' % (hist_dis.history['loss'][0]))

        # Training G
        for g_epoch in range(10):
            labels, user_emb_dis, item_emb_dis, dense_dis = get_train_instances_generator(train, discriminator, num_factors, num_negatives)
            hist_gen = generator.fit(
                [user_emb_dis, item_emb_dis, dense_dis],  # input
                [labels, labels],  # labels
                batch_size=batch_size,
                epochs=1, verbose=0, shuffle=True)
            print('Generator loss: %.3f' % (hist_gen.history['loss'][0]))

        # Evaluation
        t2 = time()
        if epoch % verbose == 0:
            results = evaluate_model(discriminator, trainRatings, testRatings, testNegatives, num_items, num_factors)

            print('Iteration %d [%.1f s]: Evaluation: P@3 = %.4f, P@5 = %.4f, P@10 = %.4f, NDCG@3 = %.4f, NDCG@5 = %.4f, NDCG@10 = %.4f, MAP = %.4f, MRR = %.4f\t [%.1f s]'
                  % (epoch, t2-t1,results[0], results[1], results[2], results[3], results[4], results[5], results[6], results[7], time() - t2))
            if results[1] > best_p5:
                best_p5, best_ndcg5, best_iter = results[1], results[4], epoch
                if args.out > 0:
                    discriminator.save_weights(model_out_file, overwrite=True)
                print("best P@5: ", best_p5, 'ndcg@5: ', best_ndcg5)

        buf = '\t'.join(['%.4f' % (x) for x in [results[0], results[1], results[2], results[3], results[4], results[5], results[6], results[7]]])
        log_results.write(str(epoch) + '\t' + buf + '\n')
        log_results.flush()
    log_results.close()

    print("End. Best Iteration %d:  P@5 = %.4f, NDCG@5 = %.4f. " % (best_iter, best_p5, best_ndcg5))
    if args.out > 0:
        print("The best discriminator model is saved to %s" % (model_out_file))