import pdb
import os
import logging

import numpy as np
import matplotlib.pyplot as plt
from hungarian_algorithm import algorithm as Halg

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def get_mapping(preds, gts, labels):
    dists = compute_SAD_dist_matrix(preds, gts)

    # Create graph for hung_alg
    G = {}
    for ii in range(len(dists)):
        # from pred to gt (row to col)
        G[ii] = {labels[jj]: dists[ii, jj] for jj in range(dists.shape[1])}
    result_list = Halg.find_matching(G, matching_type="min", return_type="list")
    return result_list


def compute_SAD_dist_matrix(preds, gts):
    assert preds.shape == gts.shape

    R = preds.shape[0]
    results = np.zeros((R, R))

    for ii in range(R):
        v1 = preds[ii]
        n1 = np.linalg.norm(v1)
        for jj in range(R):
            v2 = gts[jj]
            n2 = np.linalg.norm(v2)
            results[ii, jj] = np.arccos(v1.dot(v2) / (n1 * n2))

    return results

def plot_endmembers(model, dataset):
    preds = model.extract_endmembers().detach().numpy().T
    gts = dataset.endmembers
    assert preds.shape == gts.shape

    R  = preds.shape[0]

    labels = dataset.labels
    reverse_labels = {v: k for k, v in labels.items()}

    H_mapping = get_mapping(preds, gts, labels)

    # Compute mean SAD
    mean_SAD = np.mean([v for k, v in H_mapping])
    print(f"Mean SAD: {mean_SAD}")

    # Create mapping based on Hungarian algorithm result
    pred_2_gt = {k[0]: reverse_labels[k[1]] for k, v in H_mapping}

    # Normalize endmembers
    gts = gts / np.linalg.norm(gts, axis=1, keepdims=True)
    preds = preds / np.linalg.norm(preds, axis=1, keepdims=True)

    # Create plot
    fig, ax = plt.subplots(1, R)
    plt.suptitle("Endmembers comparison")
    for ii in range(R):
        ax[ii].plot(preds[ii], c="r")
        ax[ii].plot(gts[pred_2_gt[ii]], c="b")
        ax[ii].set_title(f"{labels[ii]}")
    plt.show()




