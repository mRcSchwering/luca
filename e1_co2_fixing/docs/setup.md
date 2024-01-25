## Setup

### Selection

Each step cells are sampled for being killed or replicated.
Probability of being killed depends on intracellular E concentration and genome size,
for being replicated depends on intracellular X concentration.
All probability function have the form $p_{k,n}(x) = x^n / (x^n + k^n)$
where $x$ is either X concentration, E concentration, or genome size.
Values for $k$ and $n$ listed below were used in all simulations if not mentioned otherwise.

_**2.1. Cell sampling** Cells are sampled each step to be killed or replicated. Replication probability depends on X concentration, killing probability on genome size and E concentration._
| variable    |      k |   n |
|:------------|-------:|----:|
| [X]         |   30   |   3 |
| [E]         |    0.5 |  -2 |
| genome size | 2000   |   7 |

To force cells to constantly regenerate X they lose 4 X during cell division.
This means in addition to the sampling described above a cell also needed to contain at least 4 X.
Furthermore cells are only allowed to divide after 10 survived steps.
This is reduce the maximum growth rate a single cell could achieve to 0.1 per step.
Finally, cells with genome size exceeding 3000 are always killed.
This is a measure to limit memory usage.
Cells sometimes replicate their genome multiple times which leads to excessively large genomes,
and in turn to massive tensors on the GPU.

![](https://raw.githubusercontent.com/mRcSchwering/luca/main/e1_co2_fixing/imgs/cell_sampling.png)

_**2.1. Cell sampling probabilities** Sampling probability of cells in dependence to variables X (top row), E (middle row) and genome size (bottom row). (Left column) probability over variable value, (right column) probability of being sampled at least once over steps._
