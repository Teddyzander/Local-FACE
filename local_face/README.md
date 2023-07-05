[![License](https://img.shields.io/github/license/xuanxuanxuan-git/facelift)](https://github.com/xuanxuanxuan-git/facelift/blob/main/LICENSE)

# Local-FACE 

This repository hosts `Local-FACE` – source code and useful resources for the following paper. 

> **Peaking into the Black-box: Actionable Interventions form Locally Acquired Counterfactual Explanations**
> 
> Counterfactuals, coupled with algorithmic recourse, have become a powerful ad-hoc tool to turn artificial intelligence (AI) systems into explainable AI (XAI) systems.
> The concept is simple: given an individual has a outcome $y$ (the factual), what actions can they take such that their outcome becomes the desired outcome $y^\prime$
> (the counterfactual). This results in algorithmic recourse that is: (1) solely aligned with the goals of the individual; and (2) simple to constrain and interpret.
> However, the properties of what amounts to a ``good'' counterfactual are still largely debated, and the act of effectively locating a counterfactual, and subsequently
> the recourse necessary for a user to change their outcome, is still an open question. Some strategies use gradient driven methods. However, these have been shown to
> be open to adversarial attacks on carefully created manifolds, leading to unfairness and a lack of robustness, and they also offer no guarantees on the feasibility
> of the recourse. Other methods are data driven, mostly solving the feasibility problem, but they require access to the entire training data set and thus may violate
> privacy and expose intellectual property. Here we introduce Local-FACE, a model-agnostic technique that extracts Feasible, Actionable Counterfactual Explanations
> (FACE) using only locally acquired information in each step. Local-FACE preserves the privacy of users by only exploiting data that it specifically requires in
> order to construct actionable algorithmic recourse, and protects the model by only offering transparency in regions deemed necessary for the intervention.

```bibtex
@article{}
```

## Algorithm
![paper_figure_strict](https://github.com/Teddyzander/Local-FACE/assets/49641102/cac787ef-9e76-4f97-b561-3516812efc8d)

Local-FACE employs a three step process to locate a counterfactual, collect relevant local information, and then find an optimal path from the factual to the counterfactual.

### (1) Explore
The algorithm searches the decision manifold locally using k nearest neighbours and momentum/inertia to find a point such that f(x) is above a certain tolerance
### (2) Exploit
Armed with knowledge as to the location of the counterfactual, the algorithm searches the dataset in the general direction of the counterfactual and constructs a graph (V,E,W) where vertices V are only connected if they fulfil a probability density criteria (strict or average). Edges are weighted by distance and density.
### (3) Enhance
The optimal path through the graph is found.
