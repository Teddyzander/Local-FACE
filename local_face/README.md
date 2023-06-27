[![License](https://img.shields.io/github/license/xuanxuanxuan-git/facelift)](https://github.com/xuanxuanxuan-git/facelift/blob/main/LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2306.02786-red.svg)](https://arxiv.org/abs/2306.02786)

# Local-FACE 

This repository hosts `Local-FACE` – source code and useful resources for the following paper. 

> **Peaking into the Black-box: Actionable Interventions form Locally Acquired Counterfactual Explanations**
>
>  .

```bibtex
@article{}
```

## Tabular Data Example

The role of geometry in (counterfactual) explainability is captured by the following figure, which demonstrates the diverse characteristics of counterfactual paths for a two-dimensional toy data set with continuous numerical features.

<p style="text-align:center">
<img src="vector_spaces/plots/CF_paths_no_bound.png" width="450">
</p>

> Example of **explanatory multiverse** constructed for tabular data with two continuous (numerical) features.
> It shows various types of **counterfactual path geometry** – their *affinity*, *branching*, *divergence* and *convergence*.
> Each journey terminates in a (possibly the same or similar) counterfactual explanation but characteristics of the steps leading there make some explanations more attractive targets, e.g., by giving the explainee more agency through multiple actionable choices towards the end of a path.

When considered in *isolation*, these paths shown in the figure above have the following properties:

- **B** is short but terminates close to a decision boundary, thus carries high uncertainty;
- **A** while longer and leading to a high-confidence region, it lacks data along its journey, which signals that it may be infeasible;
- **C** addresses the shortcomings of A, but it terminates in an area of high instability (compared to D, E<sub>i</sub>, F, G & H);
- **G & H** also do not exhibit the deficiencies of A, but they lead to an area with a high error rate;
- **D & F** have all the desired properties, but they require the most travel; and
- **E<sub>i</sub>** are feasible, but they are *incomplete* by themselves.

To compose richer explanations, we introduce the concept of *explanatory multiverse*, which allows for *spatially-aware counterfactual explainability*.
Our approach encompasses all the possible counterfactual paths and helps to navigate, reason about and compare the *geometry* of these journeys – their affinity, branching, divergence and possible future convergence.

## MNIST Example

To run the example:

- instal Python dependencies with `pip install -r requirements.txt`;
- place the dataset files in `/data/raw_data/`; and
- run `python run_explainer.py` with the default configuration.

The code to generate counterfactual explanations for the MNIST dataset is available in [`mnist_example.ipynb`](examples/mnist_example.ipynb).
The following figure demonstrates counterfactual pathfinding in the MNIST dataset and the branching factors of these paths.

<p style="text-align:center">
<img src="examples/figures/mnist.png" width="450">
</p>

> Example counterfactual journeys identified in the MNIST data set of handwritten digits. Paths 1 (blue) and 2 (green) explain an instance $\mathring{x}$ classified as $\mathring{y} = 1$ for the counterfactual class $\check{y} = 9$.
> Paths leading to alternative classification outcomes are also possible (shown in grey).
> Path 1 is shorter than Path 2 at the expense of explainees' agency – which is reflected in its smaller branching factor – therefore switching to alternative paths leading to different classes is easier, i.e., less costly in terms of distance.

## Hyper-parameters

The hyper-parameters are defined in the [`params.yaml`](facelift/library/params.yaml) file.
(Command line configuration will be implemented in future releases.)

## Datasets

### MNIST

```
distance_threshold: 6.1
prediction_threshold: 0.6
penalty_term: 1.1

directed: True
distance_function: l2
method: knn
knn:
  n_neighbours: 5

start_point_idx: 1
target_class: 9
```

### HELOC

To use tabular dataset, we first need to do preprocessing:

- one-hot encoding of categorical features; and
- normalisation.
