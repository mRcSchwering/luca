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
| stage   | substrates A               | substrates B               | additives              | genes                                                                                                                                                                                                                                                                                                                                                     |
|:--------|:---------------------------|:---------------------------|:-----------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| WL-0    | E, X                       | E, CO, methyl-FH4          | HS-CoA                 | <magicsoup.factories.CatalyticDomainFact object at 0x7fa69dac1750>, <magicsoup.factories.CatalyticDomainFact object at 0x7fa69dac1720>, <magicsoup.factories.TransporterDomainFact object at 0x7fa69dac1810>, <magicsoup.factories.TransporterDomainFact object at 0x7fa69dac1870>, <magicsoup.factories.TransporterDomainFact object at 0x7fa69dac18d0>  |
| WL-1    | E, CO, methyl-FH4          | E, CO, NADPH, formyl-FH4   | HS-CoA                 | <magicsoup.factories.CatalyticDomainFact object at 0x7fa69dac1960>, <magicsoup.factories.CatalyticDomainFact object at 0x7fa69dac19c0>, <magicsoup.factories.TransporterDomainFact object at 0x7fa69dac1a20>, <magicsoup.factories.TransporterDomainFact object at 0x7fa69dac1a80>, <magicsoup.factories.TransporterDomainFact object at 0x7fa69dac1ae0>  |
| WL-2    | E, CO, NADPH, formyl-FH4   | E, CO, NADPH, ATP, formate | HS-CoA, FH4            | <magicsoup.factories.CatalyticDomainFact object at 0x7fa69dac1b70> | <magicsoup.factories.CatalyticDomainFact object at 0x7fa69dac1bd0>, <magicsoup.factories.TransporterDomainFact object at 0x7fa69dac1c30>, <magicsoup.factories.TransporterDomainFact object at 0x7fa69dac1c90>, <magicsoup.factories.TransporterDomainFact object at 0x7fa69dac1cf0> |
| WL-3    | E, CO, NADPH, ATP, formate | E, CO, formate             | HS-CoA, FH4, NADP, ADP | <magicsoup.factories.CatalyticDomainFact object at 0x7fa69dac1d80>, <magicsoup.factories.CatalyticDomainFact object at 0x7fa69dac1de0>, <magicsoup.factories.TransporterDomainFact object at 0x7fa69dac1e40>                                                                                                                                              |
| WL-4    | E, CO, formate             | E, CO2                     | HS-CoA, FH4, NADP, ADP | <magicsoup.factories.CatalyticDomainFact object at 0x7fa69dac1ed0>, <magicsoup.factories.CatalyticDomainFact object at 0x7fa69dac1f30>                                                                                                                                                                                                                    |

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
