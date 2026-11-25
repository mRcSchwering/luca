# CO2 fixation pathways

- Calvin cycle
- 3-hydroxypropionate cycle
- Wood-Ljungdahl cycle
- reductive TCA cycle
- dicarboxylate/4-hydroxybutyrate cycle
- 3-hydroxypropionate/4-hydroxybutyrate cycle

from Gong, Fuyu & Cai, Zhen & Li, Yin. (2016). Synthetic biology for CO2 fixation,
https://en.wikipedia.org/wiki/Biological_carbon_fixation#Overview%20of%20pathways, and https://equilibrator.weizmann.ac.il
Some reactions are combined for simplicity. Common intermediates between different pathways are kept to allow cell to use combinations of pathways.
Molecule energies were solved using z3 by minimizing the absolute difference between expected reaction energies and calculated reaction energies
while only allowing a maximum per reaction error of 15 kJ/mol.
Diffusion coefficients were derived from charge and size using CO2, CO, Glucose, Sucrose, NADH, CoA as anchor points.
Permeability was deemed neglectable for all but CO2, CO, Q, QH2.
Half-lifes were assigned to the molecule's most liable group, neglecting everything above 100 days.

The chemistry is defined in [chemistry.py](../src/chemistry.py).
These are the parameters.

## Molecules

| **name** | **energy [kJ/mol]** | **half-life [days]** | **diffusivity [cm²/s]** | **permeability [cm/s]** |
| --- | --- | --- | --- | --- |
| **1,3BPG** | 360 | 0.1 | 4e-06 | 0 |
| **3PGA** | 310 | -1 | 5e-06 | 0 |
| **4-hydroxy-butyrate** | 558 | -1 | 6e-06 | 0 |
| **ADP** | 32 | -1 | 2e-06 | 0 |
| **AMP** | 1 | -1 | 3e-06 | 0 |
| **ATP** | 62 | 30 | 2e-06 | 0 |
| **CO** | 125 | -1 | 2e-05 | 0.1 |
| **CO2** | 1 | -1 | 2e-05 | 0.1 |
| **FAD** | 1 | -1 | 2e-07 | 0 |
| **FADH2** | 43 | 1 | 2e-07 | 0 |
| **FH4** | 1 | 1 | 3e-07 | 0 |
| **Fd-ox** | 1 | 10 | 1e-07 | 0 |
| **Fd-red** | 52 | 1 | 1e-07 | 0 |
| **Fe-ox** | 1 | 10 | 1e-07 | 0 |
| **Fe-red** | 63 | 1 | 1e-07 | 0 |
| **G3P** | 396 | 100 | 5e-06 | 0 |
| **HS-CoA** | 1 | 10 | 3e-07 | 0 |
| **NADP** | 1 | -1 | 2e-07 | 0 |
| **NADPH** | 68 | 1 | 2e-07 | 0 |
| **PEP** | 397 | 1 | 5e-06 | 0 |
| **PPi** | 13 | -1 | 5e-06 | 0 |
| **Pi** | 12 | -1 | 1e-05 | 0 |
| **Q** | 1 | 10 | 1e-07 | 10 |
| **QH2** | 3 | 1 | 1e-07 | 10 |
| **Ru5P** | 660 | 100 | 5e-06 | 0 |
| **RuBP** | 659 | 10 | 4e-06 | 0 |
| **a-ketoglutarate** | 552 | 100 | 5e-06 | 0 |
| **acetoacetyl-CoA** | 515 | 0.1 | 3e-07 | 0 |
| **acetyl-CoA** | 258 | 1 | 3e-07 | 0 |
| **citramalyl-CoA** | 628 | 1 | 3e-07 | 0 |
| **citrate** | 618 | -1 | 5e-06 | 0 |
| **formate** | 68 | -1 | 1e-05 | 0 |
| **formyl-FH4** | 57 | 1 | 3e-07 | 0 |
| **fumarate** | 412 | -1 | 6e-06 | 0 |
| **glyoxylate** | 182 | 30 | 8e-06 | 0 |
| **hydroxybutyryl-CoA** | 572 | 1 | 3e-07 | 0 |
| **isocitrate** | 632 | -1 | 5e-06 | 0 |
| **malate** | 408 | -1 | 6e-06 | 0 |
| **malonyl-CoA** | 252 | 1 | 3e-07 | 0 |
| **malyl-CoA** | 445 | 1 | 3e-07 | 0 |
| **methyl-FH4** | 153 | 1 | 3e-07 | 0 |
| **methylen-FH4** | 109 | 1 | 3e-07 | 0 |
| **methylmalonyl-CoA** | 445 | 1 | 3e-07 | 0 |
| **methylmalyl-CoA** | 628 | 1 | 3e-07 | 0 |
| **oxalacetate** | 371 | 1 | 5e-06 | 0 |
| **propionyl-CoA** | 451 | 1 | 3e-07 | 0 |
| **pyruvate** | 360 | 30 | 8e-06 | 0 |
| **succinate** | 424 | -1 | 6e-06 | 0 |
| **succinic-semialdehyde** | 516 | 5 | 6e-06 | 0 |
| **succinyl-CoA** | 445 | -1 | 3e-07 | 0 |

## Reactions

| **reaction** | **energy [kJ/mol]** |
| --- | --- |
| ATP + Pi + pyruvate $\rightleftharpoons$ AMP + PEP + PPi | -23 |
| ATP + FH4 + formate $\rightleftharpoons$ ADP + Pi + formyl-FH4 | -30 |
| methylmalyl-CoA $\rightleftharpoons$ citramalyl-CoA | 0 |
| HS-CoA + malate $\rightleftharpoons$ malyl-CoA | 36 |
| fumarate $\rightleftharpoons$ malate | -4 |
| malate $\rightleftharpoons$ fumarate | 4 |
| malate $\rightleftharpoons$ fumarate | 4 |
| succinyl-CoA $\rightleftharpoons$ HS-CoA + succinate | -20 |
| CO2 + NADPH + a-ketoglutarate $\rightleftharpoons$ NADP + isocitrate | 12 |
| NADPH + succinic-semialdehyde $\rightleftharpoons$ 4-hydroxy-butyrate + NADP | -25 |
| NADPH + succinic-semialdehyde $\rightleftharpoons$ 4-hydroxy-butyrate + NADP | -25 |
| ATP + HS-CoA + citrate $\rightleftharpoons$ ADP + Pi + acetyl-CoA + oxalacetate | -8 |
| 1,3BPG + NADPH $\rightleftharpoons$ G3P + NADP | -31 |
| ATP + CO2 + acetyl-CoA $\rightleftharpoons$ ADP + Pi + malonyl-CoA | -25 |
| ATP + CO2 + acetyl-CoA $\rightleftharpoons$ ADP + Pi + malonyl-CoA | -25 |
| NADP + hydroxybutyryl-CoA $\rightleftharpoons$ NADPH + acetoacetyl-CoA | 10 |
| NADP + hydroxybutyryl-CoA $\rightleftharpoons$ NADPH + acetoacetyl-CoA | 10 |
| 2 Fd-red + fumarate $\rightleftharpoons$ 2 Fd-ox + succinate | -90 |
| 4-hydroxy-butyrate + ATP + HS-CoA $\rightleftharpoons$ AMP + PPi + hydroxybutyryl-CoA | -35 |
| 4-hydroxy-butyrate + ATP + HS-CoA $\rightleftharpoons$ AMP + PPi + hydroxybutyryl-CoA | -35 |
| ATP + Ru5P $\rightleftharpoons$ ADP + RuBP | -31 |
| FADH2 + fumarate $\rightleftharpoons$ FAD + succinate | -30 |
| ATP + CO2 + propionyl-CoA $\rightleftharpoons$ ADP + Pi + methylmalonyl-CoA | -25 |
| ATP + CO2 + propionyl-CoA $\rightleftharpoons$ ADP + Pi + methylmalonyl-CoA | -25 |
| CO + HS-CoA + methyl-FH4 $\rightleftharpoons$ FH4 + acetyl-CoA | -20 |
| Q + succinate $\rightleftharpoons$ QH2 + fumarate | -10 |
| 2 Fd-red + succinyl-CoA $\rightleftharpoons$ 2 Fd-ox + HS-CoA + succinic-semialdehyde | -30 |
| NADPH + formyl-FH4 $\rightleftharpoons$ NADP + methylen-FH4 | -15 |
| glyoxylate + propionyl-CoA $\rightleftharpoons$ methylmalyl-CoA | -5 |
| NADPH + oxalacetate $\rightleftharpoons$ NADP + malate | -30 |
| NADPH + oxalacetate $\rightleftharpoons$ NADP + malate | -30 |
| CO2 + NADPH $\rightleftharpoons$ NADP + formate | 0 |
| NADPH + succinyl-CoA $\rightleftharpoons$ HS-CoA + NADP + succinic-semialdehyde | 5 |
| ATP + 3 NADPH + malonyl-CoA $\rightleftharpoons$ AMP + 3 NADP + PPi + propionyl-CoA | -50 |
| ATP + 3 NADPH + malonyl-CoA $\rightleftharpoons$ AMP + 3 NADP + PPi + propionyl-CoA | -50 |
| CO2 + 2 Fd-red + acetyl-CoA $\rightleftharpoons$ 2 Fd-ox + HS-CoA + pyruvate | 0 |
| HS-CoA + acetoacetyl-CoA $\rightleftharpoons$ 2 acetyl-CoA | 0 |
| HS-CoA + acetoacetyl-CoA $\rightleftharpoons$ 2 acetyl-CoA | 0 |
| isocitrate $\rightleftharpoons$ citrate | -14 |
| CO2 + 2 Fd-red + succinyl-CoA $\rightleftharpoons$ 2 Fd-ox + HS-CoA + a-ketoglutarate | 5 |
| CO2 + RuBP $\rightleftharpoons$ 2 3PGA | -40 |
| 5 G3P $\rightleftharpoons$ 3 Ru5P | 0 |
| 3PGA + ATP $\rightleftharpoons$ 1,3BPG + ADP | 20 |
| malyl-CoA $\rightleftharpoons$ acetyl-CoA + glyoxylate | -5 |
| methylmalonyl-CoA $\rightleftharpoons$ succinyl-CoA | 0 |
| methylmalonyl-CoA $\rightleftharpoons$ succinyl-CoA | 0 |
| CO2 + 2 Fe-red $\rightleftharpoons$ CO + 2 Fe-ox | 0 |
| CO2 + PEP $\rightleftharpoons$ Pi + oxalacetate | -15 |
| ATP + HS-CoA + succinate $\rightleftharpoons$ ADP + Pi + succinyl-CoA | 2 |
| ATP + HS-CoA + succinate $\rightleftharpoons$ ADP + Pi + succinyl-CoA | 2 |
| NADPH + methylen-FH4 $\rightleftharpoons$ NADP + methyl-FH4 | -23 |
| citramalyl-CoA $\rightleftharpoons$ acetyl-CoA + pyruvate | -10 |