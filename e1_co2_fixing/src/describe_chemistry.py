import pandas as pd
from collections import Counter
from .chemistry import CHEMISTRY
from .util import (
    save_doc,
    write_table,
    table_to_markdown,
)


def create(_: dict):
    records = [
        {
            "index": i,
            "name": d.name,
            "energy[kJ]": int(d.energy / 1000),
            "diffusivity": d.diffusivity,
            "permeability": d.permeability,
        }
        for i, d in enumerate(CHEMISTRY.molecules)
    ]
    mols = pd.DataFrame.from_records(records)
    write_table(df=mols, name="all_molecules.csv")
    molstab = table_to_markdown(
        df=mols,
        name="1.1 Molecules",
        descr="Definition of all molecules."
        " Energy affects reaction equilibriums,"
        " diffusivity allows diffusion,"
        " permeability allows permeating cell membranes.",
    )

    records = []
    for subs, prods in CHEMISTRY.reactions:
        subs_n = Counter(str(d) for d in subs)
        prods_n = Counter([str(d) for d in prods])
        subs_s = " + ".join([f"{d} {k}" for k, d in subs_n.items()])
        prods_s = " + ".join([f"{d} {k}" for k, d in prods_n.items()])
        energy = sum(d.energy for d in prods) - sum(d.energy for d in subs)
        react = f"{subs_s} $\\rightleftharpoons$ {prods_s}"
        records.append({"reaction": react, "energy [kJ]": int(energy / 1000)})

    reacts = pd.DataFrame.from_records(records)
    write_table(df=reacts, name="all_reactions.csv")
    reactstab = table_to_markdown(
        df=reacts,
        name="1.2 Reactions",
        descr="Definition of all reactions. Energy affects reaction equilibriums.",
    )

    summary = ["\n## Chemistry\n\n"]
    summary.append("\n### Molecules\n\n")
    summary.append(molstab + "\n")
    summary.append("\n### Reactions\n\n")
    summary.append(reactstab + "\n")
    save_doc(content=summary, name="chemistry.md")
