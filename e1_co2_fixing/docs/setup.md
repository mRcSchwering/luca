## Setup

Simulation mechanics are described in the [magicsoup docs](https://magic-soup.readthedocs.io/en/latest/mechanics/).
There is a circular 2D world map with cells and molecules.
Cells can live on this world map and carry attributes such as genomes and proteomes.
The simulation engine just integrates numbers irrelevant of what they represent.
However, all defaults and ranges were chosen with certain assumptions in mind.
A time step represents 1 second.
Molecule energies are given in Joules.
The side length of the map is given in $10 \mu m$.
The volume of a voxel or cell is therefore $1 pl$.
Molecule numbers in voxels or cells are given in $0.001 pmol$.
Voxel or cell concentrations therefore represent $mM$.

- [Selection](#selection)
- [Batch culture](#batch-culture)
- [ChemoStat](#chemostat)
- [Pathway training](#pathway-training)
- [Clustering](#clustering)

### Selection

Each step cells are sampled for being killed or replicated.
Probability of being killed depends on intracellular E concentration and genome size,
for being replicated depends on intracellular X concentration.
All probability function have the form $p_{k,n}(x) = x^n / (x^n + k^n)$
where $x$ is either X concentration, E concentration, or genome size.
Values for $k$ and $n$ listed below were used in all simulations if not mentioned otherwise.

_**Cell sampling** Cells are sampled each step to be killed or replicated. Replication probability depends on X concentration, killing probability on genome size and E concentration._
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

_**Cell sampling probabilities** Sampling probability of cells in dependence to variables X (top row), E (middle row) and genome size (bottom row). (Left column) probability over variable value, (right column) probability of being sampled at least once over steps._

([back to top](#setup))

### Batch culture

In batch culture experiments cells were passaged regularly.
Hereby a fraction of cells was taken and transfered onto a new map with fresh medium.
This means while cells grow and produce metabolites, molecule concentrations in the medium change.
If not mentioend otherwise cells were chosen randomly during passage.
They were passaged if confluency surpassed 70%.
A fraction of cells was taken that would create a confluency of 20% on the new map.
This process keeps cells in exponential phase and leads to a selection for the fastest growing cells.

([back to top](#setup))

### ChemoStat

In chemostat experiments a ChemoStat was simulated by creating a zone where fresh medium is constantly supplied,
and a zone where all medium is constantly removed.
The zone with fresh medium was rectangle in the middle of the map.
The zone with medium removal were two rectangles on the very left (west) and right (east) of the map.
All rectangles streched from bottom (south) to top (north).
If not mentioend otherwise their widths were 5% map size for each removal zone,
and 10% map size for the medium zone.
Over time (steps) this creates a 1D gradient across the world map x axis:
from low molecule concentrations, to high, to low.

([back to top](#setup))

### Pathway training

Cells were adapted to learn a pathway from scratch in multiple stages.
Each stage taught cells a new part of the pathway.
This was done in a batch culture process with passaging between 20% and 70% confluency.
All stages built up on each other by only continuing with successful cells of the previous stage.
Culturing medium consisted of substrates and additives.
During each stage medium was changed from one set of substrates (A) to another (B).
At the same time cells were transformed with genes that encoded the enzymes and transporters to handle the new environment.

Each stage was divided into 3 phases.
First, in an initial phase cells were grown in substrates A with base mutation rates.
Then, in an adaption phase cells were given new genes and grown in substrates B with high mutation rates.
Finally, in a final phase cells were grown in substrates B with base mutation rates.
Base mutation rates were multiplied with a factor to yield high mutation rates.
Cells had to grow a certain number of passages in a sufficiently high growth rate in order to progress to the next phase.
The mutation rate multiplier, minimum number of passages for each phase and the minimum growth rate for each run is shown below.
If not mentioned otherwise there were always 5 required passages (at minimum growth rate) for the initial, adaption, and final phase.
Minimum growth rate was 0.05 and all mutation rates were multiplied by 1000 during the adaption phase.

([back to top](#setup))

### Clustering

#### By labels

At the start of each simulation run each cell gets a unique label assigned.
If a cell divides both descendant cells carry on that label.
Thus, a cell label identifies a cell's heritage within a simulation run.

#### Genomic

Cells were clustered using a relative Levenshtein distance matrix and DBSCAN.
The relative Levenshtein distance is the absolute Levenshtein distance divided by the maximum length of 2 sequences.
DBSCAN was repeated with all combinations of epsilon values from 0.2 to 0.8 and minimum samples of 10 to 50.
The parameter set that yielded a clustering which assigned most cells to clusters was chosen.
If a clustering yielded more than 10 clusters, only the top 10 biggest clusters were regarded.
Cells with the smallest average within-cluster distance were labelled as representatives.

#### Proteomic

Cells were clustered using a relative proteome distance matrix and DBSCAN.
For the distance strings were generated for each protein in each cell.
These strings precisely identify each domain with its domain type, catalysed reaction and direction,
transport and direction, effector and direction, and the order of domains.
They do not include the parameters for maximum velocity ($v_{max}$), affinity ($K_M$), and Hill-coefficient ($h$).
So, each cells recieved a set of unique strings representing its proteome.
One minus the relative overlap of these string sets was used as distance between 2 proteomes.
DBSCAN was repeated with all combinations of epsilon values from 0.2 to 0.8 and minimum samples of 10 to 50.
The parameter set that yielded a clustering which assigned most cells to clusters was chosen.
If a clustering yielded more than 10 clusters, only the top 10 biggest clusters were regarded.
Cells with the smallest average within-cluster distance were labelled as representatives.

([back to top](#setup))