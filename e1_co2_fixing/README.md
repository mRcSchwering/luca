# Carbon Fixation

Here, I am trying to bring the cells to fix CO2.
A chemistry representing the [Wood Ljungdahl pathway](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2646786/) is defined.
Under energy expenditure CO2 can acetylate HS-CoA within 6 steps.

In this simulation replication probability is tied to intracellular acetyl-CoA concentrations and acetyl-CoA is converted to HS-CoA when a cell replicates.
This should make cells find a way to constanly replenish acetyl-CoA.
New CO2, and energy in the form of ATP and NADPH is provided in abundance.

![latest run](latest_run.png "latest run")
_World map for cells and CO2 is shown. Columns represent different time steps of the simulation, the top row shows cells, the bottom row shows CO2 concentrations. CO2 is constantly replenished on the vertical middle-line, creating a gradient. First, cells grow randomly over the map, exhausting acetyl-CoA. Then, most cells die. Only cells that know how to replenish acetyl-CoA from CO2 and that are close to the CO2 source survive._

- [main.py](./main.py) runs the simulation
- [runs/](./runs/) saved runs (in gitignore)
- [analyze_2023-01-17_21-45.ipynb](./analyze_2023-01-17_21-45.ipynb) cells formed a stable colony, but they enlarged their genomes without end

```
# init world
python -m e1_co2_fixing.main --init

# short test run:
python -m e1_co2_fixing.main --n_steps=21 --n_trials=1
...
# watch progress on tensorboard
tensorboard --logdir=./e1_co2_fixing/runs
```

## Chemistry

I wanted to have a diverse set of reactions that would allow different
possibilities to fix carbon to some biologically useful molecule.
I started by extracting reactions and molecule species from _Gong, Fuyu & Cai, Zhen & Li, Yin. (2016). Synthetic biology for CO2 fixation_.
These are 6 carbon fixing pathways.
Molecules _G3P_, _acetyl-CoA_, and _pyruvate_ are the carbon carriers that the cell will consume to replicate.
I added a molecule species _X_ and reactions for how these carbon carriers can be converted to _X_.
With _X_ I have a single handle that can be used during the simulation.

Energy is provided by _NADPH_ and _ATP_.
If a reaction needed an electron donor I explicitly added $\text{NADPH} \rightleftharpoons \text{NADP}$.
If it needed a phophate group I explicitly added $\text{ATP} \rightleftharpoons \text{ADP}$.
Other reactions are defined without energy carriers, so that the cell can create its own way of driving the reaction.
To restore energy carriers I added a molecule species _Y_ and reactions with _Y_ to restore _NADPH_ and _ATP_.
Similar to _X_, _Y_ can be used as a handle to restore energy during the simulation.

The pathways already overlap in some molecule species.
I would have liked to add more crosslinks between those pathways (_e.g._ can't hydroxybutyrate be converted to malate?).
But I don't know enough about possible reactions.
Below is the current list of reactions with reaction energies.

$$
\begin{align*}
\text{Y} + 2 \; \text{ADP} & \rightleftharpoons 2 \; \text{ATP} \; & (-80 \; \text{kJ/mol}) \\
\text{Y} + \text{NADP} & \rightleftharpoons \text{NADPH} \; & (-80 \; \text{kJ/mol}) \\
\text{G3P} & \rightleftharpoons 8 \; \text{X} \; & (-20 \; \text{kJ/mol}) \\
\text{pyruvate} & \rightleftharpoons 6 \; \text{X} \; & (-30 \; \text{kJ/mol}) \\
\text{acetyl-CoA} & \rightleftharpoons \text{HS-CoA} + 5 \; \text{X} \; & (-35 \; \text{kJ/mol}) \\
\text{NADPH} & \rightleftharpoons \text{NADP} \; & (-70 \; \text{kJ/mol}) \\
\text{ATP} & \rightleftharpoons \text{ADP} \; & (-35 \; \text{kJ/mol}) \\
\text{CO2} + \text{RuBP} & \rightleftharpoons 2 \; \text{3PGA} \; & (-35 \; \text{kJ/mol}) \\
\text{ATP} + \text{3PGA} & \rightleftharpoons \text{1,3BPG} + \text{ADP} \; & (-15 \; \text{kJ/mol}) \\
\text{NADPH} + \text{1,3BPG} & \rightleftharpoons \text{G3P} + \text{NADP} \; & (-20 \; \text{kJ/mol}) \\
5 \; \text{G3P} & \rightleftharpoons 3 \; \text{Ru5P} \; & (-15 \; \text{kJ/mol}) \\
\text{Ru5P} + \text{ATP} & \rightleftharpoons \text{RuBP} + \text{ADP} \; & (-5 \; \text{kJ/mol}) \\
\text{CO2} + \text{NADPH} & \rightleftharpoons \text{formate} + \text{NADP} \; & (-10 \; \text{kJ/mol}) \\
\text{FH4} + \text{formate} & \rightleftharpoons \text{formyl-FH4} \; & (25 \; \text{kJ/mol}) \\
\text{formyl-FH4} + \text{NADPH} & \rightleftharpoons \text{methylen-FH4} + \text{NADP} \; & (-10 \; \text{kJ/mol}) \\
\text{NADPH} + \text{methylen-FH4} & \rightleftharpoons \text{methyl-FH4} + \text{NADP} \; & (-15 \; \text{kJ/mol}) \\
\text{CO2} + \text{NADPH} & \rightleftharpoons \text{CO} + \text{NADP} \; & (-5 \; \text{kJ/mol}) \\
\text{HS-CoA} + \text{methyl-FH4} + \text{CO} & \rightleftharpoons \text{FH4} + \text{acetyl-CoA} \; & (0 \; \text{kJ/mol}) \\
\text{CO2} + \text{acetyl-CoA} & \rightleftharpoons \text{malonyl-CoA} \; & (10 \; \text{kJ/mol}) \\
3 \; \text{NADPH} + \text{malonyl-CoA} & \rightleftharpoons \text{propionyl-CoA} + 3 \; \text{NADP} \; & (-30 \; \text{kJ/mol}) \\
\text{CO2} + \text{propionyl-CoA} & \rightleftharpoons \text{methylmalonyl-CoA} \; & (0 \; \text{kJ/mol}) \\
\text{methylmalonyl-CoA} & \rightleftharpoons \text{succinyl-CoA} \; & (0 \; \text{kJ/mol}) \\
\text{succinyl-CoA} & \rightleftharpoons \text{HS-CoA} + \text{succinate} \; & (-10 \; \text{kJ/mol}) \\
\text{succinate} + \text{NADP} & \rightleftharpoons \text{NADPH} + \text{fumarate} \; & (0 \; \text{kJ/mol}) \\
\text{fumarate} & \rightleftharpoons \text{malate} \; & (0 \; \text{kJ/mol}) \\
\text{HS-CoA} + \text{malate} & \rightleftharpoons \text{malyl-CoA} \; & (10 \; \text{kJ/mol}) \\
\text{malyl-CoA} & \rightleftharpoons \text{glyoxylate} + \text{acetyl-CoA} \; & (0 \; \text{kJ/mol}) \\
\text{propionyl-CoA} + \text{glyoxylate} & \rightleftharpoons \text{methylmalyl-CoA} \; & (-5 \; \text{kJ/mol}) \\
\text{methylmalyl-CoA} & \rightleftharpoons \text{citramalyl-CoA} \; & (0 \; \text{kJ/mol}) \\
\text{citramalyl-CoA} & \rightleftharpoons \text{pyruvate} + \text{acetyl-CoA} \; & (-5 \; \text{kJ/mol}) \\
\text{NADPH} + \text{oxalacetate} & \rightleftharpoons \text{malate} + \text{NADP} \; & (-5 \; \text{kJ/mol}) \\
\text{malate} & \rightleftharpoons \text{fumarate} \; & (0 \; \text{kJ/mol}) \\
\text{NADPH} + \text{fumarate} & \rightleftharpoons \text{succinate} + \text{NADP} \; & (0 \; \text{kJ/mol}) \\
\text{HS-CoA} + \text{succinate} & \rightleftharpoons \text{succinyl-CoA} \; & (10 \; \text{kJ/mol}) \\
\text{CO2} + \text{NADPH} + \text{succinyl-CoA} & \rightleftharpoons \text{alpha-ketoglutarate} + \text{HS-CoA} + \text{NADP} \; & (-35 \; \text{kJ/mol}) \\
\text{alpha-ketoglutarate} + \text{CO2} + \text{NADPH} & \rightleftharpoons \text{isocitrate} + \text{NADP} \; & (-20 \; \text{kJ/mol}) \\
\text{isocitrate} & \rightleftharpoons \text{citrate} \; & (0 \; \text{kJ/mol}) \\
\text{HS-CoA} + \text{citrate} & \rightleftharpoons \text{oxalacetate} + \text{acetyl-CoA} \; & (35 \; \text{kJ/mol}) \\
\text{CO2} + \text{NADPH} + \text{acetyl-CoA} & \rightleftharpoons \text{HS-CoA} + \text{pyruvate} + \text{NADP} \; & (-35 \; \text{kJ/mol}) \\
\text{pyruvate} + \text{ATP} & \rightleftharpoons \text{PEP} + \text{ADP} \; & (-15 \; \text{kJ/mol}) \\
\text{CO2} + \text{PEP} & \rightleftharpoons \text{oxalacetate} \; & (-10 \; \text{kJ/mol}) \\
\text{NADPH} + \text{succinyl-CoA} & \rightleftharpoons \text{HS-CoA} + \text{SSA} + \text{NADP} \; & (-30 \; \text{kJ/mol}) \\
\text{NADPH} + \text{SSA} & \rightleftharpoons \text{GHB} + \text{NADP} \; & (-5 \; \text{kJ/mol}) \\
\text{HS-CoA} + \text{GHB} & \rightleftharpoons \text{hydroxybutyryl-CoA} \; & (35 \; \text{kJ/mol}) \\
\text{hydroxybutyryl-CoA} + \text{NADP} & \rightleftharpoons \text{acetoacetyl-CoA} + \text{NADPH} \; & (5 \; \text{kJ/mol}) \\
\text{HS-CoA} + \text{acetoacetyl-CoA} & \rightleftharpoons 2 \; \text{acetyl-CoA} \; & (0 \; \text{kJ/mol}) \\
\end{align*}
$$
