
## Culturing

### Cell Sampling

_**2.1. Cell sampling** Cells are sampled each step to be killed or replicated. Replication probability depends on X concentration, killing probability on genome size and E concentration. Probability p is calculated as $p_{k,n}(x) = x^n / (x^n + k^n)$._
| variable    |      k |   n |
|:------------|-------:|----:|
| [X]         |   30   |   3 |
| [E]         |    0.5 |  -2 |
| genome size | 2000   |   7 |

[//]: # (end)

![](https://raw.githubusercontent.com/mRcSchwering/luca/main/e1_co2_fixing/imgs/cell_sampling.png)

_**2.1. Variable effect on sampling probability** Sampling probability of cells in dependence to variables with different X concentrations (top row), E concentrations (middle row) and genome sizes (bottom row). (Left column) probability of sampling over variable, (right column) probability of being sampled at least once over steps._