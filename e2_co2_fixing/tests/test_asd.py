import magicsoup as ms

from src.chemistry import chemistry, molecules


def test_molecules_are_ordered() -> None:
    orig = chemistry.molecules
    for _ in range(10):
        mols = ms.Chemistry(molecules=molecules, reactions=[]).molecules
        assert orig == mols


# TODO: check state mixin
