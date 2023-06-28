# Carbon Fixation

In this simulation cells are brought to fix CO2 from the environment.
For this the world was defined with a [chemistry](#chemistry) resembling 6 major CO2-fixing metabolic pathways.
Cells are grown in batches in media with high CO2 and energy content.
Evolutionary pressure is applied by killing cells with low energy levels
and replicating cells with high levels of fixed carbon.
Fixed carbon is defined by molecule species such as acetyl-CoA.

![latest run](latest_run.png "latest run")
_World map for cells and CO2 is shown. Columns represent different time steps of the simulation, the top row shows cells, the bottom row shows CO2 concentrations. CO2 is constantly replenished on the vertical middle-line, creating a gradient. First, cells grow randomly over the map, exhausting acetyl-CoA. Then, most cells die. Only cells that know how to replenish acetyl-CoA from CO2 and that are close to the CO2 source survive._

- [**main**.py](./__main__.py) entrypoint for the simulation
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

## Training Experiment

Cells are grows in batch culture.
They are passaged if medium energy levels of CO2 levels are too low
or if the world map is overgrown.
In general, cells with low energy ($E$) levels have a high chance of dying
and cells with high fixed carbon levels ($X$) have a high chance of replicating.
Additionally, cells with large genomes have a high chance of dying.

Different culture strategies are tested to train more or less naive
cells (mostly random genomes) to efficiently fix carbon (create $X$).
These usually involve an adaption phase in which a complex medium
is changed to a minimal medium.
The complex medium consists of high levels of all molecule species.
The minimal medium only has high levels of essential molecule species,
such as $E$ and $CO2$.

The exact details - _e.g._ about passaging, the adaption phase, culture medium -
are hyperparameters which are varied.
In [prep.ipynb](./ipynb) useful ranges for these hyperparameters are estimated.
Different hyperparameter sets are tested where each set of hyperparameters is a
$run$ and replicates are $trials$.
See [src/train_pathway.py](./src/train_pathway.py) for details.
