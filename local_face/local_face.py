import numpy as np

from .helpers.funcs import *
from scipy import spatial
from sklearn.neighbors import KernelDensity
import networkx as nx


class LocalFace:
    """
    Implementation of local_face

    Extracts Feasible Actionable Counterfactual Explanations using only locally acquired information

    Attributes
    -----------
    data : n x d dataframe
        a dataframe of n instances and d features. The data should be a good repsentation of the model
        and density functions
    model : object
        should be a fitted model. Requires a function that gives out a score (not a crisp classifier). The
        model is only used if a counterfactual point needs to be found
    dense : object
        a function that, in some way, measures the probability or likelihood of a point being sampled from
        the space e.g., kernal density estimation

    """

    def __init__(self, data, model, dense, cf=None, steps=None):
        self.cf = cf
        self.steps = steps
        self.data = data
        self.model = model
        self.dense = dense
        self.path = None
        self.G = None
        self.prob = None

    def find_cf(self, x0, k=10, thresh=0.9, mom=3, alpha=0.05):
        """
        Find a valid counterfactual by searching through nearby data points using momentum
        Args:
            x0: starting point n array
            k: positive integer of how many neighbours to consider
            thresh: minimum value of probability classifier to terminate algorithm
            mom: positive int of number of last steps used to build momentum
            alpha: positive float of maximum step size when using momentum
        Returns:
            steps: n by p array of p steps to get from x0 to a valid counterfactual
            cf: valid counterfactual (last entry in steps)
        """

        steps = np.zeros((2, len(x0)))  # Adaptable shape
        steps[0] = x0

        # set up tree for k nearest neighbours
        tree = spatial.KDTree(self.data)

        # find closest k points to x0
        close = tree.query(x0, k=k, p=2)[1]

        # print(close)
        # print(tree.data[close])

        # find probabilities of closest points
        vals = self.model.predict_proba(tree.data[close])[:, 1]

        # save best move and delete from tree and rebuild
        indx = np.argmax(vals)
        x_hat = tree.data[close[indx]]
        steps[1] = np.array(tree.data[close[indx]])
        cf = steps[1]
        temp = np.delete(tree.data, close[indx], 0)
        tree = spatial.KDTree(temp)

        # repeat until valid counterfactual is found
        i = 1
        while self.model.predict_proba([x_hat])[0, 1] < thresh:
            # find closes k points to x0
            nei = tree.query(steps[i], k=k, p=2)
            close = nei[1]

            # find weighted probabilities of closest points
            try:
                vals = (1 / (1 + nei[0])) * \
                    self.model.predict_proba(tree.data[close])[:, 1]
            except:
                raise ValueError('Failed to find a counterfactual')

            # save best move and delete from tree and rebuild
            indx = np.argmax(vals)
            x_hat = tree.data[close[indx]]
            best_step = np.array(tree.data[close[indx]])

            # momentum term
            if mom > 0:
                if i > mom:
                    mom_dir = np.zeros(2)
                    for j in range(i - mom, mom):
                        mom_dir += steps[j] - steps[j - 1]
                else:
                    mom_dir = np.zeros(2)
                    for j in range(i):
                        mom_dir += steps[j] - steps[j - 1]
                mom_dir = mom_dir / mom
                best_step = 0.5 * (steps[i] - best_step) + 0.5 * mom_dir
                best_step_len = np.linalg.norm(best_step, 2)
                if best_step_len > alpha:
                    best_step = (best_step / best_step_len) * alpha
                best_step = x_hat + best_step

            # save best step
            steps = np.append(steps, [best_step], axis=0)
            temp = np.delete(tree.data, close[indx], 0)
            tree = spatial.KDTree(temp)

            cf = best_step
            x_hat = best_step
            i += 1
        return steps, cf

    def generate_graph(self, x0, cf, k, thresh, tol, sample, method='strict', early=True):
        """
        Find best path through data from x0 to counterfactual via query balls of radius dist
        Args:
            x0: n array starting point
            cf: n array counterfactual point
            k: k-nn parameter
            thresh: float of threshold of value for function in order to classify a
            point as a viable counterfactual
            early: bool for whether to terminate early if a closer counterfactual is found

        Returns: n by p array of p steps to get from x0 to a valid counterfactual and a graph of the steps
        """
        xt = x0
        self.prob = tol
        steps = np.zeros((1, len(x0)))
        steps[0] = x0
        # set up tree for k nearest neighbours
        tree = spatial.KDTree(self.data)
        G = nx.Graph()
        G.add_node(0)
        i = 1
        while not np.array_equiv(xt, cf):
            # check if current point actually meets criteria
            if self.model.predict_proba([xt])[0, 1] >= thresh and early:
                print('Better solution located en route')
                break
            # get vector of the best direction of travel
            dir = xt - cf
            dir_len = np.linalg.norm(dir, ord=2)
            # find points within dist of x0
            indx = tree.query(xt, k, p=2)

            # find viable point that is along the path of best direction
            dot = -np.inf
            for j in indx[1]:
                xi = np.array(tree.data[j])
                # print(xi)
                v = xt - xi
                v_len = np.linalg.norm(v, ord=2)
                vdir_len = np.linalg.norm(cf - xi, ord=2)
                if v_len != 0:
                    # print(dir)
                    # print(v)
                    # print(np.dot(dir, v))
                    # print((xi + xt))
                    temp = (((1 + (np.dot(dir, v) / (dir_len * v_len))) / 2)
                            * self.dense.score([(xi + xt) / 2])) / dir_len
                    # temp = ((1 + (np.dot(dir, v) / (dir_len * v_len))) / 2) / vdir_len
                    if temp > dot:
                        dot = temp
                        best = j
                        """prob = kde.score([xi])
                        if prob > thresh_p:
                            dot = temp
                            best = j"""

            # if we have nowhere to go and we are at the beginning, terminate
            if len(indx) == 0 and np.array_equiv(xt, x0):
                print('No CF path found for {}-NN}'.format(k))
                break

            if len(indx) == 0:
                xt = x0
                steps = np.zeros((1, 2))
                steps[0] = x0
                G = nx.Graph()
                i = 0
                G.add_node(0)

            # edit tree and save step
            else:
                xt = tree.data[best]
                best_step = np.array(tree.data[best])
                steps = np.append(steps, [best_step], axis=0)
                temp = np.delete(tree.data, best, 0)
                tree = spatial.KDTree(temp)
                G.add_node(i)
                # find and calculate edges
                for l in range(i):
                    samples = np.zeros((len(steps[i]), sample + 1))
                    for u in range(len(samples)):
                        if len(samples) > 2:
                            if u == 0 or u == 21:
                                temp1 = np.ones(
                                    int(np.ceil((sample + 1) / 2))) * steps[i][u]
                                temp2 = np.ones(
                                    int(np.floor((sample + 1) / 2))) * steps[l][u]
                                samples[u] = np.concatenate((temp1, temp2))
                            else:
                                samples[u] = np.linspace(
                                    steps[i][u], steps[l][u], sample + 1)
                        else:
                            samples[u] = np.linspace(
                                steps[i][u], steps[l][u], sample + 1)
                    samples = np.array(samples).T

                    score = np.exp(self.dense.score_samples(samples))
                    test = np.sum(score) / (sample + 1)
                    if method == 'avg':
                        w = np.linalg.norm(steps[i] - steps[l], ord=2) * test
                        G.add_edge(i, l, weight=w)
                    elif method == 'strict':
                        if all(k >= tol for k in score):
                            w = np.linalg.norm(
                                steps[i] - steps[l], ord=2) * test
                            G.add_edge(i, l, weight=w)

            i += 1

        self.steps = steps
        self.G = G
        return steps, G

    def create_edges(self, tol=0, sample=10, method='strict'):
        """
        Create edges between viable nodes and calculate weight
        Args:
        Returns: Connected graph
        """
        for i in range(len(self.steps)):
            for j in range(i):
                if np.linalg.norm(self.steps[i] - self.steps[j]) > 0:
                    if len(self.steps[i]) == 2:
                        samples = np.array([np.linspace(self.steps[i][0], self.steps[j][0], sample + 1),
                                            np.linspace(self.steps[i][1], self.steps[j][1], sample + 1)])
                    else:
                        samples = np.zeros((len(self.steps[i]), sample + 1))
                        for u in range(len(samples)):
                            if len(samples) > 2:
                                if u == 0 or u == 21:
                                    temp1 = np.ones(
                                        int(np.ceil((sample + 1) / 2))) * self.steps[i][u]
                                    temp2 = np.ones(
                                        int(np.floor((sample + 1) / 2))) * self.steps[j][u]
                                    samples[u] = np.concatenate((temp1, temp2))
                                else:
                                    samples[u] = np.linspace(
                                        self.steps[i][u], self.steps[j][u], sample + 1)
                    samples = np.array(samples).T
                    score = np.exp(self.dense.score_samples(samples))
                    test = np.sum(score) / (sample + 1)
                    if method == 'avg':
                        if test > tol:
                            w = np.linalg.norm(
                                self.steps[i] - self.steps[j], ord=2) * test
                            self.G.add_edge(i, j, weight=w)
                    elif method == 'strict':
                        if all(k >= tol for k in score):
                            w = np.linalg.norm(
                                self.steps[i] - self.steps[j], ord=2) * test
                            self.G.add_edge(i, j, weight=w)
                    else:
                        print('no method selected')
                        return 0

        return self.G

    def shortest_path(self, method='strict'):
        """
        Calculate shortest path from factual to counterfactual
        Returns: shortest path through nodes

        """
        success = False
        prob = self.prob
        threshold_reduction = 0.001
        while not success:
            try:
                self.path = nx.shortest_path(
                    self.G, source=0, target=int(self.G.number_of_nodes() - 1))
                success = True
            except:
                print(
                    f'No path found, lowering probability density')
                prob = prob - threshold_reduction
                self.create_edges(tol=prob, method=method)
        print('Completed with density probability: {}'.format(prob))
        return self.path
