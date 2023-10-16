from pathlib import Path
import magicsoup as ms
from .util import init_writer
from .experiment import BatchCulture, ChemoStat


class Logger:
    """
    Tensorboard logger base class
    """

    def __init__(
        self,
        trial_dir: Path,
        hparams: dict,
    ):
        self.writer = init_writer(logdir=trial_dir, hparams=hparams)

    def close(self):
        self.writer.close()

    def log_scalars(
        self,
        step: int,
        dtime: float,
    ):
        raise NotImplementedError

    def log_imgs(self, step: int):
        raise NotImplementedError

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.writer.close()


class BatchCultureLogger(Logger):
    """
    Tensorboard logger for batch culture experiment

    Arguments:
        - trial_dir: path to runs directory
        - hparams: dict of all hyperparameters
        - exp: initialized experiment object
        - watch_mols: list of molecules to log
    """

    def __init__(
        self,
        trial_dir: Path,
        hparams: dict,
        exp: BatchCulture,
        watch_mols: list[ms.Molecule],
    ):
        super().__init__(trial_dir=trial_dir, hparams=hparams)

        mol_2_idx = {d.name: i for i, d in enumerate(exp.world.chemistry.molecules)}
        mol_idxs = [(d, mol_2_idx[d.name]) for d in watch_mols]

        self.molecules = {f"Molecules/{s}": i for s, i in mol_idxs}
        self.exp = exp

        self.log_scalars(step=0, dtime=0.0)
        self.log_imgs(step=0)

    def log_scalars(
        self,
        step: int,
        dtime: float,
    ):
        n_cells = self.exp.world.n_cells
        molecule_map = self.exp.world.molecule_map
        cell_molecules = self.exp.world.cell_molecules

        for scalar, idx in self.molecules.items():
            tag = f"{scalar}[ext]"
            self.writer.add_scalar(tag, molecule_map[idx].mean(), step)

        if n_cells > 0:
            self.writer.add_scalar("Cells/Total", n_cells, step)
            mean_surv = self.exp.world.cell_lifetimes.float().mean()
            mean_divis = self.exp.world.cell_divisions.float().mean()
            genome_lens = [len(d) for d in self.exp.world.genomes]
            self.writer.add_scalar("Cells/Survival", mean_surv, step)
            self.writer.add_scalar("Cells/Divisions", mean_divis, step)
            self.writer.add_scalar("Cells/cPD", self.exp.cpd, step)
            self.writer.add_scalar("Cells/GrowthRate", self.exp.growth_rate, step)
            self.writer.add_scalar("Cells/GenomeSize", sum(genome_lens) / n_cells, step)
            for scalar, idx in self.molecules.items():
                tag = f"{scalar}[int]"
                self.writer.add_scalar(tag, cell_molecules[:, idx].mean(), step)

        self.writer.add_scalar("Other/TimePerStep[s]", dtime, step)
        self.writer.add_scalar("Other/Split", self.exp.split_i, step)
        self.writer.add_scalar("Other/Progress", self.exp.progress, step)
        self.writer.add_scalar("Other/MutationRate", self.exp.mutation_rate, step)

    def log_imgs(self, step: int):
        self.writer.add_image(
            "Maps/Cells", self.exp.world.cell_map, step, dataformats="WH"
        )


class ChemoStatLogger(Logger):
    """
    Tensorboard logger for ChemoStat experiment

    Arguments:
        - trial_dir: path to runs directory
        - hparams: dict of all hyperparameters
        - exp: initialized experiment object
        - watch_mols: list of molecules to log
    """

    def __init__(
        self,
        trial_dir: Path,
        hparams: dict,
        exp: ChemoStat,
        watch_mols: list[ms.Molecule],
    ):
        super().__init__(trial_dir=trial_dir, hparams=hparams)

        mol_2_idx = {d.name: i for i, d in enumerate(exp.world.chemistry.molecules)}
        mol_idxs = [(d, mol_2_idx[d.name]) for d in watch_mols]

        self.molecules = {f"Molecules/{s}": i for s, i in mol_idxs}
        self.exp = exp

        self.log_scalars(step=0, dtime=0.0)
        self.log_imgs(step=0)

    def log_scalars(
        self, step: int, dtime: float, kwargs: dict[str, float] | None = None
    ):
        n_cells = self.exp.world.n_cells
        molecule_map = self.exp.world.molecule_map
        cell_molecules = self.exp.world.cell_molecules

        for scalar, idx in self.molecules.items():
            tag = f"{scalar}[ext]"
            self.writer.add_scalar(tag, molecule_map[idx].mean(), step)

        if n_cells > 0:
            self.writer.add_scalar("Cells/Total", n_cells, step)
            mean_surv = self.exp.world.cell_lifetimes.float().mean()
            mean_divis = self.exp.world.cell_divisions.float().mean()
            genome_lens = [len(d) for d in self.exp.world.genomes]
            self.writer.add_scalar("Cells/Survival", mean_surv, step)
            self.writer.add_scalar("Cells/Divisions", mean_divis, step)
            self.writer.add_scalar("Cells/GenomeSize", sum(genome_lens) / n_cells, step)
            for scalar, idx in self.molecules.items():
                tag = f"{scalar}[int]"
                self.writer.add_scalar(tag, cell_molecules[:, idx].mean(), step)

        self.writer.add_scalar("Other/TimePerStep[s]", dtime, step)
        self.writer.add_scalar("Other/Progress", self.exp.progress, step)
        self.writer.add_scalar("Other/MutationRate", self.exp.mutation_rate, step)

        if kwargs is not None:
            for key, val in kwargs.items():
                self.writer.add_scalar(key, val, step)

    def log_imgs(self, step: int):
        self.writer.add_image(
            "Maps/Cells", self.exp.world.cell_map, step, dataformats="WH"
        )
