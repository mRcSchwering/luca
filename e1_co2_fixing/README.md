# Carbon Fixation

In this simulation cells are brought to fix CO2 from the environment.
For this the world was defined with a [chemistry](#chemistry) resembling 6 major CO2-fixing metabolic pathways.
Cells are grown in batches in media with high CO2 and energy content.
Evolutionary pressure is applied by killing cells with low energy levels
and replicating cells with high levels of fixed carbon.
Fixed carbon is defined by molecule species such as acetyl-CoA.

![result cell](./img/WL-training-result-cell.png)

_Illustration of final cells that were able to grow in a Chemostat on CO2 and energy alone. Cells developed a proteome resembling the Wood-Ljungdahl pathway. This is a summary proteome describing about 70% of cells. Rare and inactive proteins were left out. Only about 30% of cells have a protein to convert CO2 to CO. The others rely on passive CO-uptake._

- [\_\_main\_\_.py](./__main__.py) entrypoint for the simulation
- [src/chemistry.py](./src/chemistry.py) world's chemistry definition
- [src/experiment.py](./src/experiment.py) common experimental procedures
- [src/init_cells.py](./src/init_cells.py) initialize cells to grow in medium containing X (batch culture)
- [src/train_pathway.py](./src/train_pathway.py) stage-wise pathway training teaching cells to fix CO2 (batch culture)
- [src/validate_cells.py](./src/validate_cells.py) validate viability by growing cells in CO2 and E (ChemoStat)
- [runs/](./runs/) saved runs (in gitignore)
- [prep.ipynb](./prep.ipynb) estimating useful hyperparameter ranges

```
# follow help texts for commands
python -m e1_co2_fixing --help
...
tensorboard --host 0.0.0.0 --logdir=./e1_co2_fixing/runs
```

## Chemistry

The world's chemistry defines molecule species and reactions that can occur.
The 6 CO2-fixing pathways described in [Gong, 2016](https://link.springer.com/article/10.1007/s11427-016-0304-2)
were used as basis for reactions.
The pathways were simplified in some cases to reduce the overall number of involved molecule species
without limiting the diversity and possible interplay of these pathways.
One such simplification was to only use NADPH and ATP as high-energy electron and phosphate donors.

The output of each pathway is a biologically useful carbon carrier.
_E.g._ for the Wood-Ljungdahl pathway it is acetyl-CoA.
For each carbon carrier, an additional reaction was defined in which the carbon carrier
generates a new molecule species $X$.
_E.g._ one additional reaction is $\text{acetyl-CoA} \rightleftharpoons \text{HS-CoA} + 5 X$.
Thus, $X$ levels in the cell can be used a handle to measure how much carbon the cell has fixed.

As cells have no means of regenerating NADPH or ATP, an additional molecule species $E$ and 2 additional reactions were defined.
With $2 \text{ADP} + E \rightleftharpoons 2 \text{ATP}$ and $\text{NADP} + E \rightleftharpoons \text{NADPH}$
there is the possibility of regenerating ATP and NADPH.
Thus, $E$ represents available energy for a cell.

These reactions were defined together with all involved molecule species.
Cells can create proteins with catalytic domains for each reaction,
transporter domains for each molecule species, and regulatory domains for each molecule species.
Thus cells can either recreate one of these 6 pmetabolic pathways, or
create a new metabolic pathway by combining these reactions in a new way.
Energy levels of molecule species were set in a way that these 6 metabolic pathways
are thermodynamically possible and have roughly the same reaction energies.
All molecule species recieved a moderate diffusivity and 0 permeability.
Only CO2 was given a high diffusivity and high permeability.
See [chemistry.py](./chemistry.py) for details.

## Experimental Setup

One simulation run represents a one experiment in which certain cells are cultured
in a specific medium with certain interventions.
Each simulation run is repeated in trials multiple times.
In the simulation time advances step-by-step, incrementally calculating protein activities,
diffusion and permeation, and resulting molecule changes at each step.
Reaction velocities are calibrated in a way that 1 step could represent 1 second.
However, protein transcription and translation, as well as cell death and replication
happen instantaneously.
Thus, simulated cells grow and evolve much faster than their real-live equivalents.

### Cell Growth

During each step a cell can replicate and/or die.
At each step probability distribution functions
calculate probabilities for each cell for replication and dying.
The decision of whether a cell actually replicates or dies is sampled
with the calculated probability.

There are 3 probability distribution functions.
A function $p_x = [X]^{n_x} / ([X]^{n_x} + k_x^{n_x})$ calculates the probability
of replication for a cell based on its $X$ molecule contents,
$p_e = k_e^{n_e} / ([E]^{n_e} + k_e^{n_e})$ calculates the probability of dying
for a cell based on its $E$ molecule contents,
$p_s = s^{n_s} / (s^{n_s} + k_s^{n_s})$ calculates the probability of dying
for a cell based on its genome size $s$.
Initial ranges for $n_x, n_e, n_s$ and $k_x, k_e, k_s$ were estimated in [prep.ipynb](./ipynb),
concrete values were found by trial-and-error when running simulations.
They were set in a way that it is hard but not impossible for cells to grow.

### Culturing Systems

In the simulation the `world` represents a flask or plate in which the cells grow.
It's molecule contents represent the growth medium.
2 basic culturing systems are used: _Batch culture_ and a _Chemostat_.

During **batch culture** cells are placed and growth medium is set initially.
Then, cells are left to grow.
This means the contents of the growth medium change as cells grow.
Batch culture is usually coupled with passaging of cells.
If the number of cells exceeds a certain threshold (almost all of the `world` is overgrown with cells),
fresh growth medium is prepared and a randomly selected subset of cells is placed
in the fresh growth medium.
This culturing strategy can keep cells indefinitly in exponential growth phase.
It strongly selects for fast-growing cells.

In a **Chemostat** growth medium is continuously adjusted.
Fresh medium is continuously added while left over medium is removed.
This means a stable nutrient gradient can arise while cells grow.
Here, this is implemented by continously setting fresh medium in the middle of the `world`
while removing all medium on the edge of it.
This creates a nutrient gradient which is high in the middle and falls to zero toward the edge.
