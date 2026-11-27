from collections.abc import Iterable

import z3  # type: ignore


class Molecule:
    def __init__(self, name: str) -> None:
        self.name = name
        self.varname = "_" + name.replace(",", "").replace("-", "").replace(":", "")


class Reaction:
    def __init__(self, substrates: Iterable[str], products: Iterable[str], energy: int) -> None:
        self.energy = energy
        self.substrates = substrates
        self.products = products
        self.molecules = set(self.substrates) | set(self.products)

    @classmethod
    def from_text(cls, descr: str) -> "Reaction":
        assert " -> " in descr
        lft, rgt = descr.split(" -> ")
        rgt, rest = rgt.split("(")
        rest = rest.split(")")[0]
        assert "kJ/mol" in rest
        rgt = rgt.strip()
        lft = lft.strip()
        rest = rest.strip()
        return cls(
            substrates=cls._extract(lft),
            products=cls._extract(rgt),
            energy=int(rest.split("kJ/mol")[0].strip()),
        )

    @classmethod
    def _extract(cls, side: str) -> list[str]:
        summands = side.split("+")
        out = []
        for summand in summands:
            parts = [d for d in summand.strip().split() if len(d) > 0]
            assert len(parts) < 3
            if len(parts) == 2:
                out.extend([parts[1].strip()] * int(parts[0]))
            else:
                out.append(parts[0].strip())
        return out


class ChemistrySolver:
    """Solve Chemistry Energies

    Define all molecule species energies and solve them by minimuzing the difference
    between expected energies of reactions vs calculated ones within a per-reaction
    absolute error limit. Uses z3. Adds conditions to solve reactions and requires every
    molecule species to have energy greater 0. Additional z3 conditions can be added.
    """

    def __init__(
        self, molecules: list[Molecule], reactions: list[Reaction], abserr: int = 15
    ) -> None:
        self.n2v: dict[str, str] = {}
        for mol in molecules:
            if mol.name not in self.n2v:
                self.n2v[mol.name] = mol.varname
            else:
                assert self.n2v[mol.name] == mol.varname

        for react in reactions:
            for molname in react.molecules:
                assert molname in self.n2v

        self.molecules = molecules
        self.reactions = reactions
        self.abserr = abserr
        self.other_conditions: list[str] = []

    def add_other_conditions(self, conditions: Iterable[str]) -> None:
        self.other_conditions.extend(
            [" ".join([self.n2v.get(dd, dd) for dd in d.split()]) for d in conditions]
        )

    @classmethod
    def from_text(cls, lines: Iterable[str], abserr: int = 15) -> "ChemistrySolver":
        reactions = [Reaction.from_text(d) for d in lines]
        molnames = {dd for d in reactions for dd in d.molecules}
        molecules = [Molecule(d) for d in molnames]
        return cls(molecules=molecules, reactions=reactions, abserr=abserr)

    def solve(self) -> dict[str, int]:
        """Solve by minimizing for absolute error

        Defines all variables as greater 0 and solves for minimizing
        the absolute difference between expected reaction energy and actual reaction energy
        min(|exp_energy - act_energy|) while not allowing the absolute difference to be
        greater than the absolute acceptable error per reaction.
        """
        opt = z3.Optimize()

        for mol in self.molecules:
            exec(f"{mol.varname} = z3.Int('{mol.name}')")  # pylint: disable=W0122 # noqa
            opt.add(eval(f"{mol.varname} > 0"))  # pylint: disable=W0123

        for cond in self.other_conditions:
            opt.add(eval(cond))  # pylint: disable=W0123

        errors = []
        for i, react in enumerate(self.reactions):
            subs = [eval(self.n2v[d]) for d in react.substrates]  # pylint: disable=W0123
            prods = [eval(self.n2v[d]) for d in react.products]  # pylint: disable=W0123
            d_i = sum(prods) - sum(subs)
            target = react.energy

            e_i = z3.Int(f"e_{i}")
            opt.add(e_i >= d_i - target)  # e_i >= |deviation| (upper side)
            opt.add(e_i >= target - d_i)  # e_i >= |deviation| (lower side)
            opt.add(e_i <= self.abserr)
            errors.append(e_i)

        total_error = z3.Sum(errors)
        opt.minimize(total_error)
        assert str(opt.check()) == "sat"
        res = opt.model()
        return {m.name: res[eval(m.varname)] for m in self.molecules}  # pylint: disable=W0123
