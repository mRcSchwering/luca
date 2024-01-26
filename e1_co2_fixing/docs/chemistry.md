## Chemistry

The chemistry defines which molecule species and reactions exist in the simulation.
All simulations use the same chemistry.
See the [magicsoup docs](https://magic-soup.readthedocs.io/en/latest/mechanics/) for a more detailed explanation.

The 6 CO2-fixing pathways described in [Gong, 2016](https://link.springer.com/article/10.1007/s11427-016-0304-2)
were used as basis for reactions.
The pathways were simplified in some cases to reduce the overall number of involved molecule species
without limiting the diversity and possible interplay of these pathways.
One such simplification was to only use NADPH and ATP as high-energy electron and phosphate donors.

### Molecules

All molecule species are defined below.
Energy affects reaction equilibriums.
Diffusivity allows diffusion across the world map.
Permeability allows permeation across cell membranes.
X is a proxy molecule species for fixed carbon, E for energy.

_**Molecules** Definition of all molecule species._
|   index | name                |   energy[kJ] |   diffusivity |   permeability |
|--------:|:--------------------|-------------:|--------------:|---------------:|
|       0 | CO2                 |           10 |           1   |              1 |
|       1 | NADPH               |          200 |           0.1 |              0 |
|       2 | NADP                |          130 |           0.1 |              0 |
|       3 | ATP                 |          100 |           0.1 |              0 |
|       4 | ADP                 |           65 |           0.1 |              0 |
|       5 | acetyl-CoA          |          475 |           0.1 |              0 |
|       6 | HS-CoA              |          190 |           0.1 |              0 |
|       7 | pyruvate            |          330 |           0.1 |              0 |
|       8 | G3P                 |          420 |           0.1 |              0 |
|       9 | X                   |           50 |           0.1 |              0 |
|      10 | E                   |          150 |           0.1 |              0 |
|      11 | RuBP                |          725 |           0.1 |              0 |
|      12 | 3PGA                |          350 |           0.1 |              0 |
|      13 | 1,3BPG              |          370 |           0.1 |              0 |
|      14 | Ru5P                |          695 |           0.1 |              0 |
|      15 | methyl-FH4          |          410 |           0.1 |              0 |
|      16 | methylen-FH4        |          355 |           0.1 |              0 |
|      17 | formyl-FH4          |          295 |           0.1 |              0 |
|      18 | FH4                 |          200 |           0.1 |              0 |
|      19 | formate             |           70 |           0.1 |              0 |
|      20 | CO                  |           75 |           1   |              1 |
|      21 | malonyl-CoA         |          495 |           0.1 |              0 |
|      22 | propionyl-CoA       |          675 |           0.1 |              0 |
|      23 | methylmalonyl-CoA   |          685 |           0.1 |              0 |
|      24 | succinyl-CoA        |          685 |           0.1 |              0 |
|      25 | succinate           |          485 |           0.1 |              0 |
|      26 | fumarate            |          415 |           0.1 |              0 |
|      27 | malate              |          415 |           0.1 |              0 |
|      28 | malyl-CoA           |          615 |           0.1 |              0 |
|      29 | glyoxylate          |          140 |           0.1 |              0 |
|      30 | methylmalyl-CoA     |          810 |           0.1 |              0 |
|      31 | citramalyl-CoA      |          810 |           0.1 |              0 |
|      32 | oxalacetate         |          350 |           0.1 |              0 |
|      33 | alpha-ketoglutarate |          540 |           0.1 |              0 |
|      34 | isocitrate          |          600 |           0.1 |              0 |
|      35 | citrate             |          600 |           0.1 |              0 |
|      36 | PEP                 |          350 |           0.1 |              0 |
|      37 | SSA                 |          535 |           0.1 |              0 |
|      38 | GHB                 |          600 |           0.1 |              0 |
|      39 | hydroxybutyryl-CoA  |          825 |           0.1 |              0 |
|      40 | acetoacetyl-CoA     |          760 |           0.1 |              0 |

### Reactions

All reactions are defined below.
Each reaction is reversible.
Energy defines the reaction's chemical equilibrium.
Reactions involving X tie biologically fixed carbon to X.
Reactions involving E allow restoration of energy carriers.

_**1.2 Reactions** Definition of all reactions._
| reaction                                                                                        |   energy [kJ] |
|:------------------------------------------------------------------------------------------------|--------------:|
| 1 NADPH $\rightleftharpoons$ 1 NADP                                                             |           -70 |
| 1 ATP $\rightleftharpoons$ 1 ADP                                                                |           -35 |
| 2 ADP + 1 E $\rightleftharpoons$ 2 ATP                                                          |           -80 |
| 1 E + 1 NADP $\rightleftharpoons$ 1 NADPH                                                       |           -80 |
| 1 G3P $\rightleftharpoons$ 8 X                                                                  |           -20 |
| 1 pyruvate $\rightleftharpoons$ 6 X                                                             |           -30 |
| 1 acetyl-CoA $\rightleftharpoons$ 1 HS-CoA + 5 X                                                |           -35 |
| 1 CO2 + 1 RuBP $\rightleftharpoons$ 2 3PGA                                                      |           -35 |
| 1 3PGA + 1 ATP $\rightleftharpoons$ 1 1,3BPG + 1 ADP                                            |           -15 |
| 1 1,3BPG + 1 NADPH $\rightleftharpoons$ 1 G3P + 1 NADP                                          |           -20 |
| 5 G3P $\rightleftharpoons$ 3 Ru5P                                                               |           -15 |
| 1 ATP + 1 Ru5P $\rightleftharpoons$ 1 ADP + 1 RuBP                                              |            -5 |
| 1 CO2 + 1 NADPH $\rightleftharpoons$ 1 NADP + 1 formate                                         |           -10 |
| 1 FH4 + 1 formate $\rightleftharpoons$ 1 formyl-FH4                                             |            25 |
| 1 NADPH + 1 formyl-FH4 $\rightleftharpoons$ 1 NADP + 1 methylen-FH4                             |           -10 |
| 1 NADPH + 1 methylen-FH4 $\rightleftharpoons$ 1 NADP + 1 methyl-FH4                             |           -15 |
| 1 CO2 + 1 NADPH $\rightleftharpoons$ 1 CO + 1 NADP                                              |            -5 |
| 1 CO + 1 HS-CoA + 1 methyl-FH4 $\rightleftharpoons$ 1 FH4 + 1 acetyl-CoA                        |             0 |
| 1 CO2 + 1 acetyl-CoA $\rightleftharpoons$ 1 malonyl-CoA                                         |            10 |
| 3 NADPH + 1 malonyl-CoA $\rightleftharpoons$ 3 NADP + 1 propionyl-CoA                           |           -30 |
| 1 CO2 + 1 propionyl-CoA $\rightleftharpoons$ 1 methylmalonyl-CoA                                |             0 |
| 1 methylmalonyl-CoA $\rightleftharpoons$ 1 succinyl-CoA                                         |             0 |
| 1 succinyl-CoA $\rightleftharpoons$ 1 HS-CoA + 1 succinate                                      |           -10 |
| 1 NADP + 1 succinate $\rightleftharpoons$ 1 NADPH + 1 fumarate                                  |             0 |
| 1 fumarate $\rightleftharpoons$ 1 malate                                                        |             0 |
| 1 HS-CoA + 1 malate $\rightleftharpoons$ 1 malyl-CoA                                            |            10 |
| 1 malyl-CoA $\rightleftharpoons$ 1 acetyl-CoA + 1 glyoxylate                                    |             0 |
| 1 glyoxylate + 1 propionyl-CoA $\rightleftharpoons$ 1 methylmalyl-CoA                           |            -5 |
| 1 methylmalyl-CoA $\rightleftharpoons$ 1 citramalyl-CoA                                         |             0 |
| 1 citramalyl-CoA $\rightleftharpoons$ 1 acetyl-CoA + 1 pyruvate                                 |            -5 |
| 1 NADPH + 1 oxalacetate $\rightleftharpoons$ 1 NADP + 1 malate                                  |            -5 |
| 1 malate $\rightleftharpoons$ 1 fumarate                                                        |             0 |
| 1 NADPH + 1 fumarate $\rightleftharpoons$ 1 NADP + 1 succinate                                  |             0 |
| 1 HS-CoA + 1 succinate $\rightleftharpoons$ 1 succinyl-CoA                                      |            10 |
| 1 CO2 + 1 NADPH + 1 succinyl-CoA $\rightleftharpoons$ 1 HS-CoA + 1 NADP + 1 alpha-ketoglutarate |           -35 |
| 1 CO2 + 1 NADPH + 1 alpha-ketoglutarate $\rightleftharpoons$ 1 NADP + 1 isocitrate              |           -20 |
| 1 isocitrate $\rightleftharpoons$ 1 citrate                                                     |             0 |
| 1 HS-CoA + 1 citrate $\rightleftharpoons$ 1 acetyl-CoA + 1 oxalacetate                          |            35 |
| 1 CO2 + 1 NADPH + 1 acetyl-CoA $\rightleftharpoons$ 1 HS-CoA + 1 NADP + 1 pyruvate              |           -35 |
| 1 ATP + 1 pyruvate $\rightleftharpoons$ 1 ADP + 1 PEP                                           |           -15 |
| 1 CO2 + 1 PEP $\rightleftharpoons$ 1 oxalacetate                                                |           -10 |
| 1 NADPH + 1 succinyl-CoA $\rightleftharpoons$ 1 HS-CoA + 1 NADP + 1 SSA                         |           -30 |
| 1 NADPH + 1 SSA $\rightleftharpoons$ 1 GHB + 1 NADP                                             |            -5 |
| 1 GHB + 1 HS-CoA $\rightleftharpoons$ 1 hydroxybutyryl-CoA                                      |            35 |
| 1 NADP + 1 hydroxybutyryl-CoA $\rightleftharpoons$ 1 NADPH + 1 acetoacetyl-CoA                  |             5 |
| 1 HS-CoA + 1 acetoacetyl-CoA $\rightleftharpoons$ 2 acetyl-CoA                                  |             0 |

