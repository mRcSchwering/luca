from argparse import ArgumentParser

from src import chemistry
from src.util.chemistry_solver import ChemistrySolver
from src.util.misc import print_markdown_table


def _solve_chemistry() -> None:
    lines = [d for d in chemistry.__doc__.split("\n") if " -> " in d]
    solver = ChemistrySolver.from_text(lines=lines, abserr=15)
    solver.add_other_conditions(
        [
            "ATP > ADP",
            "ADP > AMP",
            "PPi > Pi",
            "NADPH > NADP",
            "FADH2 > FAD",
            "Fe-red > Fe-ox",
            "Fd-red > Fd-ox",
            "QH2 > Q",
            "acetyl-CoA > HS-CoA",
            "Ru5P > RuBP",
            "citramalyl-CoA > acetyl-CoA",
            "methyl-FH4 > FH4",
            "acetoacetyl-CoA > acetyl-CoA",
            "ATP - ADP >= 30",
            "NADPH - NADP >= 40",
            "Fd-red - Fd-ox >= 40",
            "acetyl-CoA - HS-CoA >= 30",
        ]
    )

    result = solver.solve()
    for m, e in result.items():
        print(f'{solver.n2v[m]} = Molecule("{m}", {e})')


def _show_molecules() -> None:
    records = []
    for mol in sorted(chemistry.molecules):
        records.append(
            {
                "name": mol.name,
                "energy [kJ/mol]": mol.energy,
                "half-life [days]": mol.half_life,
                "diffusivity [cm²/s]": mol.diffusivity,
                "permeability [cm/s]": mol.permeability,
            }
        )
    print_markdown_table(records=records, bold_first_col=True)


def _show_reactions() -> None:
    records = []
    for react in sorted(chemistry.reactions):
        energy = sum(d.energy for d in react.products) - sum(d.energy for d in react.substrates)
        records.append(
            {
                "reaction": str(react).replace("<->", "$\\rightleftharpoons$"),
                "energy [kJ/mol]": energy,
            }
        )
    print_markdown_table(records=records)


_subcmds = {
    "solve-chemistry": _solve_chemistry,
    "show-molecules": _show_molecules,
    "show-reactions": _show_reactions,
}


def run(**kwargs) -> None:
    subcmd = kwargs.pop("subcmd")
    _subcmds[subcmd](**kwargs)


def register(parser: ArgumentParser) -> None:
    parser.add_argument("subcmd", choices=list(_subcmds))
    parser.set_defaults(func=run)
