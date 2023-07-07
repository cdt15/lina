"""
Python implementation of the LiNGAM algorithms.
The LiNGAM Project: https://sites.google.com/view/sshimizu06/lingam
"""
# import graphviz
# import matplotlib.pyplot as plt
# from matplotlib import colors as mcolors
# from matplotlib.colors import is_color_like
import numpy as np
# from sklearn import linear_model
# from sklearn.linear_model import LassoLarsIC, LinearRegression
# from sklearn.utils import check_array, check_scalar
# import networkx as nx

import igraph as ig
# from scipy.special import expit as sigmoid
import random

# from ._rcd import extract_ancestors

__all__ = [
    "print_causal_directions",
    "print_dagc",
    "make_prior_knowledge",
    "remove_effect",
    "make_dot",
    "make_dot_highlight",
    "predict_adaptive_lasso",
    "get_sink_variables",
    "get_exo_variables",
    "find_all_paths",
    "simulate_dag",
    "simulate_parameter",
    "simulate_linear_sem",
    "simulate_linear_mixed_sem",
    "is_dag",
    "count_accuracy",
    "set_random_seed",
    "likelihood_i",
    "log_p_super_gaussian",
    "variance_i",
    "extract_ancestors",
]

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def simulate_linear_sem(adjacency_matrix, n_samples, sem_type, noise_scale=1.0):
    """Simulate samples from linear SEM with specified type of noise.

    Parameters
    ----------
    adjacency_matrix : array-like, shape (n_features, n_features)
        Weighted adjacency matrix of DAG, where ``n_features``
        is the number of variables.
    n_samples : int
        Number of samples. n_samples=inf mimics population risk.
    sem_type : str
        SEM type. gauss, exp, gumbel, logistic, poisson.
    noise_scale : float
        scale parameter of additive noise.

    Returns
    -------
    X : array-like, shape (n_samples, n_features)
        Data generated from linear SEM with specified type of noise,
        where ``n_features`` is the number of variables.
    """

    def _simulate_single_equation(X, w):
        """Simulate samples from a single equation.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features_parents)
            Data of parents for a specified variable, where
            n_features_parents is the number of parents.
        w : array-like, shape (1, n_features_parents)
            Weights of parents.

        Returns
        -------
        x : array-like, shape (n_samples, 1)
            Data for the specified variable.
        """
        if sem_type == "gauss":
            z = np.random.normal(scale=noise_scale, size=n_samples)
            x = X @ w + z
        elif sem_type == "exp":
            z = np.random.exponential(scale=noise_scale, size=n_samples)
            x = X @ w + z
        elif sem_type == "gumbel":
            z = np.random.gumbel(scale=noise_scale, size=n_samples)
            x = X @ w + z
        elif sem_type == "logistic":
            x = np.random.binomial(1, sigmoid(X @ w)) * 1.0
        elif sem_type == "poisson":
            x = np.random.poisson(np.exp(X @ w)) * 1.0
        elif sem_type == "subGaussian":
            z = np.random.normal(scale=noise_scale, size=n_samples)
            q = 0.5 + 0.3 * np.random.rand(1)  # sub-Gaussian
            z = np.sign(z) * pow(np.abs(z), q)
            z = z - np.mean(z)
            z = z / np.std(z)
            x = X @ w + z
        elif sem_type == "supGaussian":
            z = np.random.normal(scale=noise_scale, size=n_samples)
            q = 1.2 + 0.8 * np.random.rand(1)  # super-Gaussian
            z = np.sign(z) * pow(np.abs(z), q)
            z = z - np.mean(z)
            z = z / np.std(z)
            x = X @ w + z
        elif sem_type == "nonGaussian":
            z = np.random.normal(scale=noise_scale, size=n_samples)
            qq = -1
            if qq == 1:
                q = 0.5 + 0.3 * np.random.rand(1)  # sub-Gaussian
            else:
                q = 1.2 + 0.8 * np.random.rand(1)  # super-Gaussian
            z = np.sign(z) * pow(np.abs(z), q)
            z = z - np.mean(z)
            z = z / np.std(z)
            x = X @ w + z
        elif sem_type == "uniform":
            z = np.random.uniform(0, 1, n_samples)
            z = z - np.mean(z)
            z = z / np.std(z)
            x = X @ w + z
        elif sem_type == "gamma":
            z = np.random.gamma(2, 2, n_samples)
            z = z - np.mean(z)
            z = z / np.std(z)
            x = X @ w + z
        elif sem_type == "laplace":
            z = np.random.laplace(0, scale=noise_scale, size=n_samples)
            x = X @ w + z
        else:
            raise ValueError("unknown sem type")
        return x

    n_features = adjacency_matrix.shape[0]
    if np.isinf(n_samples):
        if sem_type == "gauss":
            # make 1/n_features X'X = true cov
            X = (
                np.sqrt(n_features)
                * noise_scale
                * np.linalg.pinv(np.eye(n_features) - adjacency_matrix)
            )
            return X
        else:
            raise ValueError("population risk not available")
    X = np.zeros([n_samples, n_features])

    G = ig.Graph.Weighted_Adjacency(adjacency_matrix.tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == n_features

    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        X[:, j] = _simulate_single_equation(X[:, parents], adjacency_matrix[parents, j])
    return X


def simulate_parameter(B, w_ranges=((-2.0, -0.5), (0.5, 2.0))):
    """Simulate SEM parameters for a DAG.

    Parameters
    ----------
    B : array-like, shape (n_features, n_features)
        Binary adjacency matrix of DAG, where ``n_features``
        is the number of features.
    w_ranges : tuple
        Disjoint weight ranges.

    Returns
    -------
    adjacency_matrix : array-like, shape (n_features, n_features)
        Weighted adj matrix of DAG, where ``n_features``
        is the number of features.
    """

    adjacency_matrix = np.zeros(B.shape)
    S = np.random.randint(len(w_ranges), size=B.shape)  # which range
    for i, (low, high) in enumerate(w_ranges):
        U = np.random.uniform(low=low, high=high, size=B.shape)
        adjacency_matrix += B * (S == i) * U
    return adjacency_matrix


def simulate_dag(n_features, n_edges, graph_type):
    """Simulate random DAG with some expected number of edges.

    Parameters
    ----------
    n_features : int
        Number of features.
    n_edges : int
        Expected number of edges.
    graph_type : str
        ER, SF.

    Returns
    -------
    B : array-like, shape (n_features, n_features)
        binary adjacency matrix of DAG.
    """

    def _random_permutation(M):
        # np.random.permutation permutes first axis only
        P = np.random.permutation(np.eye(M.shape[0]))
        return P.T @ M @ P

    def _random_acyclic_orientation(B_und):
        return np.tril(_random_permutation(B_und), k=-1)

    def _graph_to_adjmat(G):
        return np.array(G.get_adjacency().data)

    if graph_type == "ER":
        # Erdos-Renyi
        G_und = ig.Graph.Erdos_Renyi(n=n_features, m=n_edges)
        B_und = _graph_to_adjmat(G_und)
        B = _random_acyclic_orientation(B_und)
    elif graph_type == "SF":
        # Scale-free, Barabasi-Albert
        G = ig.Graph.Barabasi(
            n=n_features, m=int(round(n_edges / n_features)), directed=True
        )
        B = _graph_to_adjmat(G)
    elif graph_type == "BP":
        # Bipartite, Sec 4.1 of (Gu, Fu, Zhou, 2018)
        top = int(0.2 * n_features)
        G = ig.Graph.Random_Bipartite(
            top, n_features - top, m=n_edges, directed=True, neimode=ig.OUT
        )
        B = _graph_to_adjmat(G)
    else:
        raise ValueError("unknown graph type")
    B_perm = _random_permutation(B)
    assert ig.Graph.Adjacency(B_perm.tolist()).is_dag()
    return B_perm

def simulate_linear_mixed_sem(
    adjacency_matrix, n_samples, sem_type, dis_con, noise_scale=None
):
    """Simulate mixed samples from linear SEM with specified type of noise.

    Parameters
    ----------
    adjacency_matrix : array-like, shape (n_features, n_features)
        Weighted adjacency matrix of DAG, where ``n_features``
        is the number of variables.
    n_samples : int
        Number of samples. n_samples=inf mimics population risk.
    sem_type : str
        SEM type. gauss, mixed_random_i_dis.
    dis_con : array-like, shape (1, n_features)
        Indicator of discrete/continuous variables, where "1"
        indicates a continuous variable, while "0" a discrete
        variable.
    noise_scale : float
        scale parameter of additive noise.

    Returns
    -------
    X : array-like, shape (n_samples, n_features)
        Data generated from linear SEM with specified type of noise,
        where ``n_features`` is the number of variables.
    """

    def _simulate_single_equation(X, w, scale, dis_con_j):
        """Simulate samples from a single equation.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features_parents)
            Data of parents for a specified variable, where
            n_features_parents is the number of parents.
        w : array-like, shape (1, n_features_parents)
            Weights of parents.
        scale : scale parameter of additive noise.
        dis_con_j : indicator of the j^th variable.

        Returns
        -------
        x : array-like, shape (n_samples, 1)
                    Data for the specified variable.
        """
        if sem_type == "gauss":
            z = np.random.normal(scale=scale, size=n_samples)
            x = X @ w + z
        elif sem_type == "mixed_random_i_dis":
            # randomly generated with fixed number of discrete variables.
            if dis_con_j:  # 1:continuous;   0:discrete
                z = np.random.laplace(0, scale=scale, size=n_samples)
                x = X @ w + z
            else:
                x = np.random.binomial(1, sigmoid(X @ w)) * 1.0
        else:
            raise ValueError("unknown sem type")
        return x

    n_features = adjacency_matrix.shape[0]
    if noise_scale is None:
        scale_vec = np.ones(n_features)
    elif np.isscalar(noise_scale):
        scale_vec = noise_scale * np.ones(n_features)
    else:
        if len(noise_scale) != n_features:
            raise ValueError("noise scale must be a scalar or has length n_features")
        scale_vec = noise_scale
    if not is_dag(adjacency_matrix):
        raise ValueError("adjacency_matrix must be a DAG")
    if np.isinf(n_samples):  # population risk for linear gauss SEM
        if sem_type == "gauss":
            # make 1/n_features X'X = true cov
            X = (
                np.sqrt(n_features)
                * np.diag(scale_vec)
                @ np.linalg.inv(np.eye(n_features) - adjacency_matrix)
            )
            return X
        else:
            raise ValueError("population risk not available")
    # empirical risk
    G = ig.Graph.Weighted_Adjacency(adjacency_matrix.tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == n_features
    X = np.zeros([n_samples, n_features])
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        # X[:, j] = _simulate_single_equation(X[:, parents], adjacency_matrix[parents, j], scale_vec[j])
        X[:, j] = _simulate_single_equation(
            X[:, parents], adjacency_matrix[parents, j], scale_vec[j], dis_con[0, j]
        )
    return X


def is_dag(W):
    """Check if W is a dag or not.

    Parameters
    ----------
    W : array-like, shape (n_features, n_features)
        Binary adjacency matrix of DAG, where ``n_features``
        is the number of features.

    Returns
    -------
     G: boolean
        Returns true or false.

    """
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    return G.is_dag()

