"""
# CO2 fixation pathways

- Calvin cycle
- 3-hydroxypropionate cycle
- Wood-Ljungdahl cycle
- reductive TCA cycle
- dicarboxylate/4-hydroxybutyrate cycle
- 3-hydroxypropionate/4-hydroxybutyrate cycle

from Gong, Fuyu & Cai, Zhen & Li, Yin. (2016). Synthetic biology for CO2 fixation.
and https://en.wikipedia.org/wiki/Biological_carbon_fixation#Overview%20of%20pathways

Some reactions are combined for simplicity.
Common intermediates between different pathways are kept to allow cell to use combinations of pathways.

## Calvin cycle

Stage 1:
RuBP + CO2 -> 2 3PGA (-35 kJ/mol)
3PGA + ATP -> 1,3BPG + ADP
1,3BPG + NADPH -> G3P + NADP (-21.9 kJ/mol)

Stage 2:
5 G3P -> 3 Ru5P
Ru5P + ATP -> RuBP + ADP

## 3-Hydroxypropionate Bicycle

CO2 fixation:
acetyl-CoA + CO2 + ATP -> malonyl-CoA + ADP
malonyl-CoA + 3 NADPH + ATP -> propionyl-CoA + 3 NADP + AMP
propionyl-CoA + CO2 + ATP -> methylmalonyl-CoA + ADP
methylmalonyl-CoA -> succinyl-CoA
succinyl-CoA -> succinate + HS-CoA
succinate + NADP -> fumarate + NADPH
fumarate -> malate
malate + HS-CoA -> malyl-CoA
malyl-CoA -> acetyl-CoA + glyoxylate

Glyoxylate Assimilation (-14 kJ/mol):
propionyl-CoA + glyoxylate -> methylmalyl-CoA
methylmalyl-CoA -> citramalyl-CoA
citramalyl-CoA -> acetyl-CoA + pyruvate

## Wood-Ljungdahl cycle

Methyl (Eastern) Branch:
CO2 + NADPH -> formate + NADP (- 21.5 kJ/mol)
formate + FH4 + ATP -> formyl-FH4 + ADP (-8.4 kJ/mol)
formyl-FH4 + NADPH -> methylen-FH4 + NADP
methylen-FH4 + NADPH -> methyl-FH4 + NADP (-57.3 kJ/mol)

Carbonyl (Western) Branch:
CO2 + 2 Fd-red -> CO + 2 Fd-ox
methyl-FH4 + CO + HS-CoA -> acetyl-CoA + FH4

## Reductive TCA

oxalacetate + NADPH -> malate + NADP (-64.2 kJ/mol)
malate -> fumarate (+2.3 kJ/mol)
fumarate + FADH -> succinate + FAD (-102.2 kJ/mol)
succinate + HS-CoA + ATP -> succinyl-CoA + ADP 
succinyl-CoA + FdRed + CO2 -> a-ketoglutarate + HS-CoA + FdOx (-14.8 kJ/mol incl prev)
a-ketoglutarate + CO2 + NADPH -> isocitrate + NADP (+3.8 kJ/mol)
isocitrate -> citrate (-8.5 kJ/mol)
citrate + HS-CoA + ATP -> oxalacetate + acetyl-CoA + ADP

## Dicarboxylate/4-Hydroxybutyrate Cycle

acetyl-CoA + Fd-red -> pyruvate + Fd-ox
pyruvate + ATP -> PEP + AMP
PEP + CO2 -> oxalacetate
oxalacetate + NADPH -> malate + NADP
malate -> fumarate
fumarate + 2 MV-red -> succinate + 2 MV-ox
succinate + HS-CoA + ATP -> succinyl-CoA + ADP
succinyl-CoA + 2 MV-red -> succinic semialdehyde + HS-CoA + 2 MV-ox
succinic semialdehyde + NADPH -> 4-hydroxy-butyrate + NADP
4-hydroxy-butyrate + HS-CoA + ATP -> hydroxybutyryl-CoA + AMP
hydroxybutyryl-CoA + NADP -> acetoacetyl-CoA + NADPH
acetoacetyl-CoA + HS-CoA -> 2 acetyl-CoA

## 3-Hydroxypropionate/4-Hydroxybutyrate Cycle

acetyl-CoA + CO2 + ATP -> malonyl-CoA + ADP
malonyl-CoA + 3 NADPH + ATP -> propionyl-CoA + 3 NADP + ADP
propionyl-CoA + CO2 + ATP -> methylmalonyl-CoA + ADP
methylmalonyl-CoA -> succinyl-CoA
succinyl-CoA + NADPH -> succinic semialdehyde + HS-CoA + NADP
succinic semialdehyde + NADPH -> 4-hydroxy-butyrate + NADP
4-hydroxy-butyrate + HS-CoA + ATP -> hydroxybutyryl-CoA + ADP
hydroxybutyryl-CoA + NADP -> acetoacetyl-CoA + NADPH
acetoacetyl-CoA + HS-CoA -> 2 acetyl-CoA

## Common Reactions

Some reactions are powered by ATP or NADPH.
Generally, cells should create energy coupling themselves.
So, the reaction is defined without ATP/NADPH.

ATP -> ADP (-30.5 kJ/mol)
NADPH -> NADP (-73.2 kcal/mol)

But in some cases the phosphorylation or electron transfer is important.
In these cases, the reaction will be defined together with ATP/NADPH.
Here, ATP -> ADP is the representative energy carrier and phosphate donor.
E.g. ADP -> AMP is not defined.
NADPH is the representative energy carrier and electron donor.
E.g. no FADH2 or Fd-red is defined.

Molecule X will be used to caputure biologically available carbon.
Molecule Y will be used to replenish energy carriers.

acetyl-CoA -> HS-CoA + 2 X
G3P -> 3 X
pyruvate -> 3 X
NADP + Y -> NADPH
3 ADP + Y -> 3 ATP
"""
from magicsoup.containers import Molecule, Chemistry


# TODO
# aufsetzen auf HS-CoA: ATP -> ADP (-20)
# CO2 auf CoA mit aufnehmen: ATP -> ADP (-20)
# 2e reduzieren: NADPH -> NADP (-50)

# TODO: propionyl-CoA + CO2 -> methylmalonyl-CoA (0 kJ/mol) is weird


def f(p: int, c: int, e: int) -> float:
    return 20 * p + 10 * c + 30 * e


_co2 = Molecule("CO2", 10.0 * 1e3, diffusivity=1.0, permeability=1.0)
_NADPH = Molecule("NADPH", 200.0 * 1e3)
_NADP = Molecule("NADP", 130.0 * 1e3)
_ATP = Molecule("ATP", 100.0 * 1e3)
_ADP = Molecule("ADP", 65.0 * 1e3)
_G3P = Molecule("G3P", 420.0 * 1e3)  # P 3C 12e !
_acetylCoA = Molecule("acetyl-CoA", 475.0 * 1e3)  # 2C 8e !
_HSCoA = Molecule("HS-CoA", 200.0 * 1e3)  # !
_pyruvate = Molecule("pyruvate", 900 * 1e3)  # 3C 10e
_X = Molecule("X", 250.0 * 1e3)  # TODO
_Y = Molecule("Y", 150.0 * 1e3)

_common_mols = [_co2, _NADPH, _NADP, _ATP, _ADP]

_common_reacts = [
    ([_NADPH], [_NADP]),  # -70
    ([_ATP], [_ADP]),  # -35
    ([_ADP, _ADP, _Y], [_ATP, _ATP]),  # -80
    ([_NADP, _Y], [_NADPH]),  # -80
    ([_G3P], [_X, _X, _X]),  # TODO
    ([_pyruvate], [_X, _X, _X]),  # TODO
    ([_acetylCoA], [_HSCoA, _X, _X]),  # TODO
]


# Calvin
_RuBP = Molecule("RuBP", 725.0 * 1e3)  # 2P 5C 20e !
_3PGA = Molecule("3PGA", 350.0 * 1e3)  # P 3C 10e !
_13BPG = Molecule("1,3BPG", 370.0 * 1e3)  # 2P 3C 10e !
_G3P = Molecule("G3P", 420.0 * 1e3)  # P 3C 12e !
_Ru5P = Molecule("Ru5P", 695.0 * 1e3)  # P 5C 20e !

_calvin_reacts = [
    ([_RuBP, _co2], [_3PGA, _3PGA]),  # -35
    ([_3PGA, _ATP], [_13BPG, _ADP]),  # -15
    ([_13BPG, _NADPH], [_G3P, _NADP]),  # -20
    ([_G3P, _G3P, _G3P, _G3P, _G3P], [_Ru5P, _Ru5P, _Ru5P]),  # -15
    ([_Ru5P, _ATP], [_RuBP, _ADP]),  # -5
]

_calvin_mols = [_RuBP, _3PGA, _13BPG, _G3P, _Ru5P]


# 3-Hydroxypropionate bicycle
_acetylCoA = Molecule("acetyl-CoA", 475.0 * 1e3)  # 2C 8e !
_HSCoA = Molecule("HS-CoA", 200.0 * 1e3)  # !
_malonylCoA = Molecule("malonyl-CoA", 495.0 * 1e3)  # 3C 8e !
_propionylCoA = Molecule("propionyl-CoA", 675.0 * 1e3)  # 3C 14e !
_methylmalonylCoA = Molecule("methylmalonyl-CoA", 685.0 * 1e3)  # 4C 14e !
_succinylCoA = Molecule("succinyl-CoA", 685.0 * 1e3)  # 4C 14e !
_succinate = Molecule("succinate", 485.0 * 1e3)  # 4C 14e !
_fumarate = Molecule("fumarate", 415.0 * 1e3)  # 4C 12e !
_malate = Molecule("malate", 415.0 * 1e3)  # 4C 12e !
_malylCoA = Molecule("malyl-CoA", 615.0 * 1e3)  # 4C 12e !
_glyoxylate = Molecule("glyoxylate", 140.0 * 1e3)  # 2C 4e !
_methylmalylCoA = Molecule("methylmalyl-CoA", 810.0 * 1e3)  # 5C 18e !
_citramalylCoA = Molecule("citramalyl-CoA", 810.0 * 1e3)  # 5C 18e !
_pyruvate = Molecule("pyruvate", 330.0 * 1e3)  # 3C 10e !

_hprop_reacts = [
    ([_acetylCoA, _co2], [_malonylCoA]),  # +10
    (
        [_malonylCoA, _NADPH, _NADPH, _NADPH],
        [_propionylCoA, _NADP, _NADP, _NADP],
    ),  # -30
    ([_propionylCoA, _co2], [_methylmalonylCoA]),  # 0
    ([_methylmalonylCoA], [_succinylCoA]),  # 0
    ([_succinylCoA], [_succinate, _HSCoA]),  # 0
    ([_succinate, _NADP], [_fumarate, _NADPH]),  # 0
    ([_fumarate], [_malate]),  # 0
    ([_malate, _HSCoA], [_malylCoA]),  # 0
    ([_malylCoA], [_acetylCoA, _glyoxylate]),  # 0
    ([_propionylCoA, _glyoxylate], [_methylmalylCoA]),  # -5
    ([_methylmalylCoA], [_citramalylCoA]),  # 0
    ([_citramalylCoA], [_acetylCoA, _pyruvate]),  # -5
]

_hprop_mols = [
    _malonylCoA,
    _propionylCoA,
    _methylmalonylCoA,
    _succinylCoA,
    _succinate,
    _fumarate,
    _malate,
    _malylCoA,
    _glyoxylate,
    _methylmalylCoA,
    _citramalylCoA,
    _pyruvate,
]

# WL
_methylFH4 = Molecule("methyl-FH4", 410.0 * 1e3)  # 1C 6e !
_methylenFH4 = Molecule("methylen-FH4", 355.0 * 1e3)  # 1C 4e !
_formylFH4 = Molecule("formyl-FH4", 295.0 * 1e3)  # 1C 2e !
_FH4 = Molecule("FH4", 200.0 * 1e3)  # !
_formate = Molecule("formate", 70.0 * 1e3)  # 1C 2e !
_co = Molecule("CO", 75 * 1e3)  # 1C 2e !
_acetylCoA = Molecule("acetyl-CoA", 475.0 * 1e3)  # 2C 8e !
_HSCoA = Molecule("HS-CoA", 200.0 * 1e3)  # !

_wl_reacts = [
    ([_co2, _NADPH], [_formate, _NADP]),  # -10
    ([_formate, _FH4], [_formylFH4]),  # -10
    ([_formylFH4, _NADPH], [_methylenFH4, _NADP]),  # -10
    ([_methylenFH4, _NADPH], [_methylFH4, _NADP]),  # -15
    ([_co2, _NADPH], [_co, _NADP]),  # -5
    ([_methylFH4, _co, _HSCoA], [_acetylCoA, _FH4]),  # -10
]

_wl_mols = [
    _methylFH4,
    _methylenFH4,
    _formylFH4,
    _FH4,
    _formate,
    _co,
]

# Reductive TCA
_acetylCoA = Molecule("acetyl-CoA", 475.0 * 1e3)  # 2C 8e !
_HSCoA = Molecule("HS-CoA", 200.0 * 1e3)  # !
_oxalacetate = Molecule("oxalacetate", 350.0 * 1e3)  # 4C 10e !
_malate = Molecule("malate", 415.0 * 1e3)  # 4C 12e !
_fumarate = Molecule("fumarate", 415.0 * 1e3)  # 4C 12e !
_succinate = Molecule("succinate", 485.0 * 1e3)  # 4C 14e !
_succinylCoA = Molecule("succinyl-CoA", 685.0 * 1e3)  # 4C 14e !
_aKetoglutarate = Molecule("alpha-ketoglutarate", 540.0 * 1e3)  # 5C 16e !
_isocitrate = Molecule("isocitrate", 600.0 * 1e3)  # 5C 18e !
_citrate = Molecule("citrate", 600.0 * 1e3)  # 5C 18e !

_rtca_reacts = [
    ([_oxalacetate, _NADPH], [_malate, _NADP]),  # -5
    ([_malate], [_fumarate]),  # 0
    ([_fumarate, _NADPH], [_succinate, _NADP]),  # 0
    ([_succinate, _HSCoA], [_succinylCoA]),  # 0
    ([_succinylCoA, _NADPH, _co2], [_aKetoglutarate, _HSCoA, _NADP]),  # -25
    ([_aKetoglutarate, _co2, _NADPH], [_isocitrate, _NADP]),  # -20
    ([_isocitrate], [_citrate]),  # 0
    ([_citrate, _HSCoA], [_oxalacetate, _acetylCoA]),  # +25
]

_rtca_mols = [
    _oxalacetate,
    _malate,
    _fumarate,
    _succinate,
    _succinylCoA,
    _aKetoglutarate,
    _isocitrate,
    _citrate,
]

# Dicarboxylate/4-Hydroxybutyrate Cycle
_acetylCoA = Molecule("acetyl-CoA", 475.0 * 1e3)  # 2C 8e !
_pyruvate = Molecule("pyruvate", 330.0 * 1e3)  # 3C 10e !
_PEP = Molecule("PEP", 350.0 * 1e3)  # 1P 3C 10e !
_oxalacetate = Molecule("oxalacetate", 350.0 * 1e3)  # 4C 10e !
_malate = Molecule("malate", 415.0 * 1e3)  # 4C 12e !
_fumarate = Molecule("fumarate", 415.0 * 1e3)  # 4C 12e !
_succinate = Molecule("succinate", 485.0 * 1e3)  # 4C 14e !
_succinylCoA = Molecule("succinyl-CoA", 685.0 * 1e3)  # 4C 14e !
_SSA = Molecule("SSA", 535.0 * 1e3)  # 4C 16e !
_GHB = Molecule("GHB", 600.0 * 1e3)  # 4C 18e !
_hydroxybutyrylCoA = Molecule("hydroxybutyryl-CoA", 830.0 * 1e3)  # 4C 18e !
_acetoacetylCoA = Molecule("acetoacetyl-CoA", 760.0 * 1e3)  # 4C 16e !

_dcarbhb_reacts = [
    ([_acetylCoA, _co2, _NADPH], [_pyruvate, _HSCoA, _NADP]),  # -25
    ([_pyruvate, _ATP], [_PEP, _ADP]),  # -15
    ([_PEP, _co2], [_oxalacetate]),  # -10
    ([_oxalacetate, _NADPH], [_malate, _NADP]),  # -5
    ([_malate], [_fumarate]),  # 0
    ([_fumarate, _NADPH], [_succinate, _NADP]),  # 0
    ([_succinate, _HSCoA], [_succinylCoA]),  # 0
    ([_succinylCoA, _NADPH], [_SSA, _HSCoA, _NADP]),  # -20
    ([_SSA, _NADPH], [_GHB, _NADP]),  # -5
    ([_GHB, _HSCoA], [_hydroxybutyrylCoA]),  # +30
    ([_hydroxybutyrylCoA, _NADP], [_acetoacetylCoA, _NADPH]),  # 0
    ([_acetoacetylCoA, _HSCoA], [_acetylCoA, _acetylCoA]),  # -10
]

_dcarbhb_mols = [
    _pyruvate,
    _PEP,
    _oxalacetate,
    _malate,
    _fumarate,
    _succinate,
    _succinylCoA,
    _SSA,
    _GHB,
    _hydroxybutyrylCoA,
    _acetoacetylCoA,
]

# 3-Hydroxypropionate/4-Hydroxybutyrate Cycle
_acetylCoA = Molecule("acetyl-CoA", 475.0 * 1e3)  # 2C 8e !
_malonylCoA = Molecule("malonyl-CoA", 495.0 * 1e3)  # 3C 8e !
_propionylCoA = Molecule("propionyl-CoA", 675.0 * 1e3)  # 3C 14e !
_methylmalonylCoA = Molecule("methylmalonyl-CoA", 685.0 * 1e3)  # 4C 14e !
_succinylCoA = Molecule("succinyl-CoA", 685.0 * 1e3)  # 4C 14e !
_SSA = Molecule("SSA", 535.0 * 1e3)  # 4C 16e !
_GHB = Molecule("GHB", 600.0 * 1e3)  # 4C 18e !
_hydroxybutyrylCoA = Molecule("hydroxybutyryl-CoA", 830.0 * 1e3)  # 4C 18e !
_acetoacetylCoA = Molecule("acetoacetyl-CoA", 760.0 * 1e3)  # 4C 16e !

_hprophbut_reacts = [
    ([_acetylCoA, _co2], [_malonylCoA]),  # +10
    (
        [_malonylCoA, _NADPH, _NADPH, _NADPH],
        [_propionylCoA, _NADP, _NADP, _NADP],
    ),  # -30
    ([_propionylCoA, _co2], [_methylmalonylCoA]),  # 0
    ([_methylmalonylCoA], [_succinylCoA]),  # 0
    ([_succinylCoA, _NADPH], [_SSA, _HSCoA, _NADP]),  # -20
    ([_SSA, _NADPH], [_GHB, _NADP]),  # -5
    ([_GHB, _HSCoA], [_hydroxybutyrylCoA]),  # +30
    ([_hydroxybutyrylCoA, _NADP], [_acetoacetylCoA, _NADPH]),  # 0
    ([_acetoacetylCoA, _HSCoA], [_acetylCoA, _acetylCoA]),  # -10
]

_hprophbut_mols = [
    _malonylCoA,
    _propionylCoA,
    _methylmalonylCoA,
    _succinylCoA,
    _SSA,
    _GHB,
    _hydroxybutyrylCoA,
    _acetoacetylCoA,
]

# combined

MOLECULES = (
    _common_mols
    + _calvin_mols
    + _wl_mols
    + _hprop_mols
    + _rtca_mols
    + _dcarbhb_mols
    + _hprophbut_mols
)

REACTIONS = (
    _common_reacts
    + _calvin_reacts
    + _wl_reacts
    + _hprop_reacts
    + _rtca_reacts
    + _dcarbhb_reacts
    + _hprophbut_reacts
)

CHEMISTRY = Chemistry(molecules=MOLECULES, reactions=REACTIONS)
