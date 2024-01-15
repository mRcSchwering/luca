## WL Pathway Training

Cells were adapted to learn the WL pathway from scratch in multiple stages.
Each stage taught cells a new part of the pathway.
This was done in a batch culture process.
All stages built up on each other by only continuing with successful cells of the previous stage.
Cells of the first stage (WL-0) were spawned with genomes encoding only E and X transporters.

_**3.1 Training strategy** Training WL pathway in multiple stages, start each stage with successful cells of the previous stage._
| runname                          | stage   | prev-stage                          |
|:---------------------------------|:--------|:------------------------------------|
| train-pathway_2024-01-13_18-28_0 | WL-0    | init                                |
| train-pathway_2024-01-13_18-30_0 | WL-1    | train-pathway_2024-01-13_18-28_0:-1 |
| train-pathway_2024-01-13_18-32_0 | WL-2    | train-pathway_2024-01-13_18-30_0:-1 |
| train-pathway_2024-01-13_18-35_0 | WL-3    | train-pathway_2024-01-13_18-32_0:-1 |
| train-pathway_2024-01-13_19-01_0 | WL-4    | train-pathway_2024-01-13_18-35_0:-1 |

Culturing medium consisted of substrates and additives.
During each stage medium was changed from one set of substrates to another,
here simply called from substrates A to substrates B.
At the same time genes were added to all cells that would give them the required
enzymes and transporters to handle the new environment.

_**3.2 WL training stages** Training WL pathway in multiple stages, start each stage with successful cells of the previous stage._
| stage   | substrates A               | substrates B               | additives              | genes                                                    |
|:--------|:---------------------------|:---------------------------|:-----------------------|:---------------------------------------------------------|
| WL-0    | E, X                       | E, CO, methyl-FH4          | HS-CoA                 | 1 acetyl-CoA $\rightleftharpoons$ 1 HS-CoA + 5 X,                         |
|         |                            |                            |                        | 1 CO + 1 HS-CoA + 1 methyl-FH4 $\rightleftharpoons$ 1 FH4 + 1 acetyl-CoA, |
|         |                            |                            |                        | methyl-FH4 transporter,                                  |
|         |                            |                            |                        | HS-CoA transporter,                                      |
|         |                            |                            |                        | FH4 transporter                                          |
| WL-1    | E, CO, methyl-FH4          | E, CO, NADPH, formyl-FH4   | HS-CoA                 | 1 NADPH + 1 formyl-FH4 $\rightleftharpoons$ 1 NADP + 1 methylen-FH4,      |
|         |                            |                            |                        | 1 NADPH + 1 methylen-FH4 $\rightleftharpoons$ 1 NADP + 1 methyl-FH4,      |
|         |                            |                            |                        | formyl-FH4 transporter,                                  |
|         |                            |                            |                        | NADPH transporter,                                       |
|         |                            |                            |                        | NADP transporter                                         |
| WL-2    | E, CO, NADPH, formyl-FH4   | E, CO, NADPH, ATP, formate | HS-CoA, FH4            | 1 FH4 + 1 formate $\rightleftharpoons$ 1 formyl-FH4 | 1 ATP $\rightleftharpoons$ 1 ADP,    |
|         |                            |                            |                        | formate transporter,                                     |
|         |                            |                            |                        | ATP transporter,                                         |
|         |                            |                            |                        | ADP transporter                                          |
| WL-3    | E, CO, NADPH, ATP, formate | E, CO, formate             | HS-CoA, FH4, NADP, ADP | 1 E + 1 NADP $\rightleftharpoons$ 1 NADPH,                                |
|         |                            |                            |                        | 2 ADP + 1 E $\rightleftharpoons$ 2 ATP,                                   |
|         |                            |                            |                        | E transporter                                            |
| WL-4    | E, CO, formate             | E, CO2                     | HS-CoA, FH4, NADP, ADP | 1 CO2 + 1 NADPH $\rightleftharpoons$ 1 NADP + 1 formate,                  |
|         |                            |                            |                        | 1 CO2 + 1 NADPH $\rightleftharpoons$ 1 CO + 1 NADP                        |

Each stage was divided into 3 phases.
First, in an initial phase cells were grown in substrates A with base mutation rates.
Then, in an adaption phase cells were given new genes and grown in substrates B with high mutation rates.
Finally, in a final phase cells were grown in substrates B with base mutation rates.
Base mutation rates were multiplied with a factor to yield high mutation rates.
Cells had to grow a certain number of passages in a sufficiently high growth rate in order to progress to the next phase.
The mutation rate multiplier, minimum number of passages for each phase and the minimum growth rate for each run is shown below.

_**3.2 WL training hyperparameters** Hyperparameters for WL pathway training simulation runs._
| runname                          |   n_init_splits |   n_adapt_splits |   n_final_splits |   min_gr |   mutation_rate_mult |
|:---------------------------------|----------------:|-----------------:|-----------------:|---------:|---------------------:|
| train-pathway_2024-01-13_18-28_0 |               5 |                5 |                5 |     0.05 |                  100 |
| train-pathway_2024-01-13_18-30_0 |               5 |                5 |                5 |     0.05 |                  100 |
| train-pathway_2024-01-13_18-32_0 |               5 |                5 |                5 |     0.05 |                  100 |
| train-pathway_2024-01-13_18-35_0 |               5 |                5 |                5 |     0.05 |                  100 |
| train-pathway_2024-01-13_19-01_0 |               5 |                5 |                5 |     0.05 |                  100 |
