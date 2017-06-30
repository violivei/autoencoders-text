#pylint: skip-file
import time
import sys
import numpy as np
import theano
import theano.tensor as T
from VAE import *
import data
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import csv

use_gpu(0)

lr = 0.001
drop_rate = 0.
batch_size = 20
hidden_size = 500
latent_size = 2
# try: sgd, momentum, rmsprop, adagrad, adadelta, adam, nesterov_momentum
optimizer = "adam"
continuous = False

train_idx, valid_idx, test_idx, other_data = data.apnews()
[docs, dic, w2i, i2w, targets] = other_data

dim_x = len(dic)
dim_y = dim_x
print("#features = ", dim_x, "#labels = ", dim_y)

print("compiling...")
model = VAE(dim_x, dim_x, hidden_size, latent_size, continuous, optimizer)

print("training...")
start = time.time()
for i in range(10):
    train_xy = data.batched_idx(train_idx, batch_size)
    error = 0.0
    in_start = time.time()
    for batch_id, x_idx in train_xy.items():
        X = data.batched_news(x_idx, other_data)
        cost, z = model.train(X, lr)
        error += cost
        #print i, batch_id, "/", len(train_xy), cost
    in_time = time.time() - in_start

    error /= len(train_xy);
    print("Iter = " + str(i) + ", Loss = " + str(error) + ", Time = " + str(in_time))

print("training finished. Time = " + str(time.time() - start))

print("save model...")
save_model("./model/vae_text.model", model)

print("lode model...")
load_model("./model/vae_text.model", model)

print("validation..")
valid_xy = data.batched_idx(valid_idx, batch_size)
error = 0
for batch_id, x_idx in valid_xy.items():
    X = data.batched_news(x_idx, other_data)
    cost, y = model.validate(X)
    error += cost
print("Loss = " + str(error / len(valid_xy)))

top_w = 20
## manifold 
if latent_size == 2:
    test_xy = data.batched_idx(test_idx, 1000)
    x_idx = test_xy[0]
    X = data.batched_news(x_idx, other_data)
    dict_doc = data.get_doc(x_idx, other_data)

    mu = np.array(model.project(X))

    kmeans = KMeans(init='k-means++', n_clusters=10, n_init=10)
    kmeans.fit(mu)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = mu[:, 0].min() - 1, mu[:, 0].max() + 1
    y_min, y_max = mu[:, 1].min() - 1, mu[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Output file
    df = pd.DataFrame({'col1': dict_doc['docs'], 'col2': dict_doc['targets'], 'col3': kmeans.predict(mu)})
    df.to_csv('output.csv', sep='\t')

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')

    plt.plot(mu[:, 0], mu[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
              'Centroids are marked with white cross')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()
    
    plt.figure(figsize=(8, 6)) 
    plt.scatter(mu[:, 0], mu[:, 1], c="r")
    plt.savefig("2dstructure.png", bbox_inches="tight")
    plt.show()

    nx = ny = 20
    v = 100
    x_values = np.linspace(-v, v, nx)
    y_values = np.linspace(-v, v, ny) 
    canvas = np.empty((28*ny, 20*nx))
    for i, xi in enumerate(x_values):
        for j, yi in enumerate(y_values):
            z = np.array([[xi, yi]], dtype=theano.config.floatX)
            y = model.generate(z)[0,:]
            ind = np.argsort(-y)
            print(xi, yi)
            for k in range(top_w):
                print(i2w[ind[k]])
            print("\n")

