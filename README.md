<img align="left" src="docs/logo.png"> &nbsp; &nbsp;

[![Documentation Status](https://readthedocs.org/projects/ansicolortags/badge/?version=latest)](http://ansicolortags.readthedocs.io/?badge=latest)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
                                                               
# Optimal Estimation of Gaussian (Poly)trees

 This is an implementation of the following paper:
 "[Optimal Estimation of Gaussian (Poly)trees]()" arXiv preprint arXiv (2023).

## Introduction
We develop optimal algorithms for learning undirected Gaussian trees and directed Gaussian polytrees from data. We consider both problems of distribution learning (i.e. in KL distance) and structure learning (i.e. exact recovery).
1. [**Chow-Liu algorithm**] The first approach is based on the Chow-Liu algorithm, and learns an optimal tree-structured distribution efficiently.
2. [**PC-tree**]  The second approach is a modification of the PC algorithm for polytrees that uses partial correlation as a conditional independence tester for constraint-based structure learning.
We derive explicit finite-sample guarantees for both approaches, and show that both approaches are optimal by deriving matching lower bounds.

## Prerequisites

- **Python 3.6+**
  - `argpase`
  - `numpy`
  - `pandas`
  - `scipy`
  - `sklearn`
  - `networkx`
  - `tqdm`
  - `causallearn`
  - [`Tetrad`](https://sites.google.com/view/tetradcausal)
  
## Contents

- `data.py` - generate synthetic data. 
- `config.py` - simulation parameters.
- `evaluate.py` - performance evauation
- `main.py` - main algorihtm.
- `method.py`- including mutual information tester, chow-liu tree algorithm, and pc-tree algorithm

## Parameters

| Parameter    | Type | Description                      | Options            |
| -------------|------| ---------------------------------|  :----------:      |
| `n`          |  int |  number of samples               |      -             |
| `d`          |  int |  number of variables             |      -             |
| `tg`         |  str |  type of graph (default: random tree)  |  -                 |


## Running a simple demo

The simplest way to try out Polytree is to run a simple example:
```bash
$ git clone https://github.com/YohannaWANG/Polytree.git
$ cd Polytree/
$ python $ cd Polytree/main.py
```

## Runing as a command

Alternatively, you can install the package and run the algorithm as a command:
```bash
$ pip install git+git://github.com/YohannaWANG/Polytree
$ cd Polytree
$ python main.py --d 100 --n 1000 
```

## Performance comparison
SHD        | PRR
:--------------------------------------------------------------------:|:-----------------------------------------------------------------------------------:
<img width="400" alt="characterization" src="/docs/gaussian_100_shd.png" >  |  <img width="400" alt="characterization" src="/docs/gaussian_100_prr.png" >
