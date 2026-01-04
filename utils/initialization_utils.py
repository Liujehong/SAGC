import torch
from sklearn.cluster import KMeans
from utils.metric import cluster_accuracy
import random
import numpy as np
import os

def init_centers(model, dataloader, n_class, device, is_labeled_pixel, seed=42):
    model.eval()
    labels_vector, y_pred_vector = [], []
    for i, (x, y) in enumerate(dataloader):
        x_list = [x_i.to(device) for x_i in x]
        with torch.no_grad():
            h = model.forward_embedding(x_list)
        y_pred_vector.extend(h.cpu().detach().numpy())
        labels_vector.extend(y.numpy())
    labels_vector = np.asarray(labels_vector)
    y_pred_vector = np.asarray(y_pred_vector)

    # Perform KMeans clustering
    clustering_model = KMeans(n_clusters=n_class, init='k-means++', random_state=seed)
    clustering_model.fit(y_pred_vector)

    if is_labeled_pixel:
        acc, kappa, nmi, ari, pur, ca = cluster_accuracy(labels_vector, clustering_model.labels_)
    else:
        indx_labeled = np.nonzero(labels_vector)[0]
        y = labels_vector[indx_labeled]
        y_pred = clustering_model.labels_[indx_labeled]
        acc, kappa, nmi, ari, pur, ca = cluster_accuracy(y, y_pred)

    print("Iterations:{}, Clustering ACC:{:.3f}, centers:{}".format(clustering_model.n_iter_, acc,
                                                                    clustering_model.cluster_centers_.shape))
    centers = torch.from_numpy(clustering_model.cluster_centers_)
    return centers, acc, kappa, nmi, ari, pur, ca


def set_global_random_seed(seed):
 # seed init.
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # torch seed init.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False # train speed is slower after enabling this opts.

    # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

    # avoiding nondeterministic algorithms (see https://pytorch.org/docs/stable/notes/randomness.html)
    torch.use_deterministic_algorithms(True)