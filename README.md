# Empirical analysis with Hawkes processes
This repo contains a variety of tools and tutorials for fitting and analyzing uni- and multi-dimensional Bayesian Hawkes processes.

In the broadest possible terms, Hawkes processes are stochastic models for point processes (a.k.a. counts of events over time) with some form of endogenous feedback loop. They may be appropriate for modelling a variety of datasets comprised of series of timestamps and/or spatial coordinates. 

Our focus is on using them as tools for modelling human behavior, especially in business applications. Throughout our tutorial notebooks, you may assume that a "process iteration" is a time series for an individual person and that each "point" in the process is a website visit, a phone call, a post on social media, etc...

While there are many ways of fitting and studying Hawkes processes, many of their shortcomings have to with lack of robustness in small samples. That motivates a workaround by means of Bayesian hierarchical extensions - our estimates can converge more neatly when we have large cross-sectional samples but only individually sparse timeseries.

To test these notebooks, you can start by creating a conda environment with the requirements file:

```
conda create --name Hawkes-Experiments --file requirements.txt
```

Below is an index of our modules and notebooks. Notebooks in each module are numbered to provide a coherent reading experience.

### 0 - Auxiliary

Includes module files and sample datasets used in later notebooks.

### 1 - Simulation

Covers relevant point process simulation algorithms and python-specific data-formatting standards that facilitate subsequent data analysis. 

### 2 - Model Fitting 

Has worked examples for model-fitting and diagnostics through both independent MLE and hierarchical Bayesian procedures.
