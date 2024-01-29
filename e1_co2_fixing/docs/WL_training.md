## Wood-Ljungdahl Training

Cells were transformed to learn the Wood-Ljungdahl pathway in 5 stages WL-0 to WL-4.
Substrate changes and transformed genes for each stage are shown below.
Finally, as a validation, cells were grown again for 100 generations in a ChemoStat
with only CO2 and E as substrates.

_**Wood-Ljungdahl training stages** Training WL pathway in multiple stages by transforming cells and changing substrates from A to B._
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

All successful training runs in sequence are shown below.
Dashed lines show transitions from initial to adaption and from adation to final phase.
Usually, growth rate decreases as medium in abruptly switched during the adaption phase.
Conversely, average cell age increases.
After switching to adaption phase it seems to take 2-3 passages for growth rate drop.
This roughly equates 4-6 cell divisions.
I assume that cells still have abundant molecules to undergo this amount of divisions before being affected by the medium change.
As cells are transformed their average genome size increases at the start of adaption phase.
However, throughout the run genome sizes usually slowly shrink, especially in later phases.
This could be the genome-size-dependent killing function taking effect.

![](../imgs/WL-pathway-training.png)

_**Wood-Ljungdahl training runs** Timeseries of batch culture runs of WL pathway training. Dashed vertical lines divide initial, adaption, and final phases of each stage. Generation, age, growth rate, and genome size show the average over all cells. Progress marks the progress from 0 to 1 for each stage._

As validation, the final state of the final stage was grown in a ChemoStat with only CO2 and E as substrates.
Cells were grown to average generation 100.

_**Wood-Ljungdahl training runs** Runs of WL pathway training and validation. Each stage is started with successful cells of the previous stage._
| runname                           | stage   | previous state                      | culturing     |
|:----------------------------------|:--------|:------------------------------------|---------------|
| train-pathway_2024-01-15_16-39_0  | WL-0    | init                                | batch culture |
| train-pathway_2024-01-15_16-43_0  | WL-1    | train-pathway_2024-01-15_16-39_0:-1 | batch culture |
| train-pathway_2024-01-15_16-45_0  | WL-2    | train-pathway_2024-01-15_16-43_0:-1 | batch culture |
| train-pathway_2024-01-15_16-48_0  | WL-3    | train-pathway_2024-01-15_16-45_0:-1 | batch culture |
| train-pathway_2024-01-15_16-57_0  | WL-4    | train-pathway_2024-01-15_16-48_0:-1 | batch culture |
| grow-chemostat_2024-01-15_17-43_0 |         | train-pathway_2024-01-15_16-57_0:-1 | chemostat     |

([back to top](#wood-ljungdahl-training))

### Result Cells

The final state of the validation run in a ChemoStat are shown below.
Cells are clustered based on their genome.
During the run the total number of cells grew slightly as cells started to grow further outwards towards the low concentration regions.
This could be because cells in the middle almost cease dividing due to space overcrowding.
Thus, they use up less substrates and eventually the concentration gradient starts moving further outwards.

The 3 biggest clusters are c1, c5, c0.
c0 only occupies the substrate rich middle zone, c5 only the outer zone with low substrates.
c1 grows thorugh both zones.
c5 consists of only 1 label, so they all descend from the same cell at the start of the ChemoStat run.
c1 consists of 3 main labels. c0 has the most diverse heritage.
Almost all clusters have all catalytic and transporter domains to realize the whole WL pathway in some form or another.
Only c5 is lacking NADPH $\rightleftharpoons$ NADP + E.
It can't regenrate NADPH.
Either it relies on extracellular NADPH or it has some non-trivial way of regenrating it.
Nevertheless, c5 has the highest growth rate.

Interestingly, cells have basically abandonned the idea of using ATP hydrolization to drive FH4 + formate $\rightleftharpoons$ formyl-FH4.
Instead they use the energy of CO2 + NADPH $\rightleftharpoons$ NADP + formate to drive FH4 + formate $\rightleftharpoons$ formyl-FH4.
This allows them get rid of ADP transporters and enzymes for ATP regeneration.

Additionally, basically all cells combined
1 acetyl-CoA $\rightleftharpoons$ 1 HS-CoA + 5 X with
1 HS-CoA + 1 methyl-FH4 + 1 CO $\rightleftharpoons$ 1 acetyl-CoA + 1 FH4.
This actually makes sense because the second reaction with 0kJ can be pushed by the first reaction with -35kJ.

![](../imgs/WL-trained-chemostat-final-state-genomic-clustering.png)

_**WL-trained cells after ChemoStat** WL-trained cells after growing for 100 generations in ChemoStat clustered by genomes. Colors indicate clusters. Cell map and cluster abundancies (top row), cell parameters (2nd row), molecule concentrations (3rd row), and proteins (bottom row) for most abundant clusters are shown. Bottom row shows frequency of proteins in each cluster for the overall 30 most abundant proteins._

#### Cluster c1

![](../imgs/WL-trained-chemostat-final-state-genomic-clustering-c1.png)

_**Cluster c1 representative** Genome and transcriptome of (lower) and molecule concentration over time in (upper) cluster representing cell. Cell was isolated and its extracellular environment kept constant while advancing time. Transcripts above the genome are encoded on the forward strand, below it on the reverse-complement strand. Colors represent domain types._

This is the only cluster that still has 2 ADP + E $\rightleftharpoons$ ATP.
The last shown CO2 to acetyl-CoA ratio is about 1.7 but its equilibrium is not reached yet.

_**Cluster c1 representative proteins** Protein encoded by each coding by each coding region. Molecule transport directions are only relevant in proteins with multiple domains. Multiple domains are concatenaed with `|`. `[i]` is intra-, `[e]` extracellular._
|   CDS | protein                                                                                    |
|------:|:-------------------------------------------------------------------------------------------|
|     0 | E exporter                                                                                 |
|     1 | 1 HS-CoA + 1 methyl-FH4 + 1 CO $\rightleftharpoons$ 1 acetyl-CoA + 1 FH4                                    |
|     2 | 1 acetyl-CoA $\rightleftharpoons$ 1 HS-CoA + 5 X \| 1 HS-CoA + 1 methyl-FH4 + 1 CO $\rightleftharpoons$ 1 acetyl-CoA + 1 FH4 |
|     3 | HS-CoA exporter                                                                            |
|     4 | 1 NADPH + 1 formyl-FH4 $\rightleftharpoons$ 1 NADP + 1 methylen-FH4                                         |
|     5 | 1 NADPH + 1 methylen-FH4 $\rightleftharpoons$ 1 NADP + 1 methyl-FH4                                         |
|     6 | NADP exporter                                                                              |
|     7 | 1 FH4 + 1 formate $\rightleftharpoons$ 1 formyl-FH4 \| 1 CO2 + 1 NADPH $\rightleftharpoons$ 1 NADP + 1 formate               |
|     8 | malate importer \| CO[i] activator                                                         |
|     9 | ADP exporter                                                                               |
|    10 | ATP exporter                                                                               |
|    11 | 1 NADPH $\rightleftharpoons$ 1 NADP + 1 E                                                                   |
|    12 | 2 ADP + 1 E $\rightleftharpoons$ 2 ATP                                                                      |
|    13 | 1 CO2 + 1 NADPH $\rightleftharpoons$ 1 NADP + 1 formate                                                     |
|    14 | E importer \| 1 CO2 + 1 NADPH $\rightleftharpoons$ 1 NADP + 1 formate                                       |
|    15 | 1 CO2 + 1 NADPH $\rightleftharpoons$ 1 NADP + 1 CO                                                          |
|    16 | succinyl-CoA importer                                                                      |
|    17 | succinyl-CoA importer                                                                      |
|    18 | succinyl-CoA importer                                                                      |
|    19 | FH4 importer                                                                               |
|    20 | 1 pyruvate $\rightleftharpoons$ 6 X                                                                         |


#### Cluster c5

![](../imgs/WL-trained-chemostat-final-state-genomic-clustering-c5.png)

_**Cluster c5 representative** Genome and transcriptome of (lower) and molecule concentration over time in (upper) cluster representing cell. Cell was isolated and its extracellular environment kept constant while advancing time. Transcripts above the genome are encoded on the forward strand, below it on the reverse-complement strand. Colors represent domain types._

I don't understand how this cell can grow so quickly.
It lacks NADP + E $\rightleftharpoons$ NADPH so it must regenerate NADPH differently.
There is also no NADPH transporter.
While the other cells have a NADPH to NADP ratio of 3 or 5, this cell has one of around 0.6.
This is not enough to reduce NADP.
There is a propionyl-CoA exporter coupled with 1 NADP + 1 formate $\rightleftharpoons$ 1 CO2 + 1 NADPH
and 1 methylmalonyl-CoA $\rightleftharpoons$ 1 CO2 + 1 propionyl-CoA
but I don't see how methylmalonyl-CoA would be replenished.
It's CO2 to acetyl-CoA ratio after 50s is about 54.
So whatever this cell is doing, it's not very efficient.


_**Cluster c5 representative proteins** Protein encoded by each coding by each coding region. Molecule transport directions are only relevant in proteins with multiple domains. Multiple domains are concatenaed with `|`. `[i]` is intra-, `[e]` extracellular._
|   CDS | protein                                                                                    |
|------:|:-------------------------------------------------------------------------------------------|
|     0 | 1 methylmalonyl-CoA $\rightleftharpoons$ 1 CO2 + 1 propionyl-CoA                                            |
|     1 | 1 CO2 + 1 NADPH $\rightleftharpoons$ 1 NADP + 1 CO                                                          |
|     2 | succinyl-CoA importer                                                                      |
|     3 | succinyl-CoA importer                                                                      |
|     4 | E importer                                                                                 |
|     5 | 1 HS-CoA + 1 methyl-FH4 + 1 CO $\rightleftharpoons$ 1 acetyl-CoA + 1 FH4                                    |
|     6 | 1 acetyl-CoA $\rightleftharpoons$ 1 HS-CoA + 5 X \| 1 HS-CoA + 1 methyl-FH4 + 1 CO $\rightleftharpoons$ 1 acetyl-CoA + 1 FH4 |
|     7 | 1 NADPH + 1 formyl-FH4 $\rightleftharpoons$ 1 NADP + 1 methylen-FH4                                         |
|     8 | 1 NADPH + 1 methylen-FH4 $\rightleftharpoons$ 1 NADP + 1 methyl-FH4                                         |
|     9 | 1 NADP + 1 formate $\rightleftharpoons$ 1 CO2 + 1 NADPH                                                     |
|    10 | propionyl-CoA exporter \| 1 NADP + 1 formate $\rightleftharpoons$ 1 CO2 + 1 NADPH                           |
|    11 | HS-CoA importer                                                                            |
|    12 | NADP exporter                                                                              |
|    13 | pyruvate importer                                                                          |
|    14 | 1 FH4 + 1 formate $\rightleftharpoons$ 1 formyl-FH4 \| 1 CO2 + 1 NADPH $\rightleftharpoons$ 1 NADP + 1 formate               |
|    15 | ADP exporter                                                                               |
|    16 | ADP exporter                                                                               |
|    17 | malate importer \| ADP exporter                                                            |
|    18 | ADP importer                                                                               |
|    19 | 1 methylmalonyl-CoA $\rightleftharpoons$ 1 CO2 + 1 propionyl-CoA                                            |
|    20 | 1 CO2 + 1 NADPH $\rightleftharpoons$ 1 NADP + 1 CO                                                          |
|    21 | ADP importer                                                                               |
|    22 | FH4 importer                                                                               |
|    23 | ADP importer                                                                               |


#### Cluster c0


![](../imgs/WL-trained-chemostat-final-state-genomic-clustering-c0.png)

_**Cluster c0 representative** Genome and transcriptome of (lower) and molecule concentration over time in (upper) cluster representing cell. Cell was isolated and its extracellular environment kept constant while advancing time. Transcripts above the genome are encoded on the forward strand, below it on the reverse-complement strand. Colors represent domain types._

The cell has all the necessary domains.
CO2 to acetyl-CoA ratio after 50 steps is about 23.

_**Cluster c0 representative proteins** Protein encoded by each coding by each coding region. Molecule transport directions are only relevant in proteins with multiple domains. Multiple domains are concatenaed with `|`. `[i]` is intra-, `[e]` extracellular._
|   CDS | protein                                                                                    |
|------:|:-------------------------------------------------------------------------------------------|
|     0 | E importer                                                                                 |
|     1 | 1 HS-CoA + 1 methyl-FH4 + 1 CO $\rightleftharpoons$ 1 acetyl-CoA + 1 FH4                                    |
|     2 | 1 acetyl-CoA $\rightleftharpoons$ 1 HS-CoA + 5 X \| 1 HS-CoA + 1 methyl-FH4 + 1 CO $\rightleftharpoons$ 1 acetyl-CoA + 1 FH4 |
|     3 | 1 NADPH + 1 formyl-FH4 $\rightleftharpoons$ 1 NADP + 1 methylen-FH4                                         |
|     4 | 1 NADPH + 1 methylen-FH4 $\rightleftharpoons$ 1 NADP + 1 methyl-FH4                                         |
|     5 | 1 NADP + 1 formate $\rightleftharpoons$ 1 CO2 + 1 NADPH                                                     |
|     6 | HS-CoA importer                                                                            |
|     7 | NADP exporter                                                                              |
|     8 | pyruvate importer                                                                          |
|     9 | 1 FH4 + 1 formate $\rightleftharpoons$ 1 formyl-FH4 \| 1 CO2 + 1 NADPH $\rightleftharpoons$ 1 NADP + 1 formate               |
|    10 | malate importer                                                                            |
|    11 | ADP exporter                                                                               |
|    12 | ADP exporter                                                                               |
|    13 | 1 NADP + 1 E $\rightleftharpoons$ 1 NADPH                                                                   |
|    14 | PEP[e] activator \| 1 NADP + 1 E $\rightleftharpoons$ 1 NADPH                                               |
|    15 | ADP importer                                                                               |
|    16 | 1 CO2 + 1 NADPH $\rightleftharpoons$ 1 NADP + 1 CO                                                          |
|    17 | acetoacetyl-CoA exporter \| acetoacetyl-CoA importer                                       |
|    18 | FH4 importer                                                                               |

([back to top](#wood-ljungdahl-training))
