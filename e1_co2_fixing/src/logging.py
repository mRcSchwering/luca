from pathlib import Path
import json
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams as get_summary
import magicsoup as ms
from .culture import BatchCulture, ChemoStat


class Logger:
    """Tensorboard logger base"""

    def __init__(
        self,
        trial_dir: Path,
        hparams: dict,
    ):
        metrics = {"Other/Progress": 0.0}
        self.writer = SummaryWriter(log_dir=trial_dir)
        exp, ssi, sei = get_summary(hparam_dict=hparams, metric_dict=metrics)
        self.writer.file_writer.add_summary(exp)
        self.writer.file_writer.add_summary(ssi)
        self.writer.file_writer.add_summary(sei)

        with open(trial_dir / "hparams.json", "w", encoding="utf-8") as fh:
            json.dump(hparams, fh)

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
    """Tensorboard logger for batch culture"""

    def __init__(
        self,
        trial_dir: Path,
        hparams: dict,
        cltr: BatchCulture,
        watch_mols: list[ms.Molecule],
    ):
        super().__init__(trial_dir=trial_dir, hparams=hparams)
        mol_2_idx = cltr.world.chemistry.mol_2_idx
        self.molecules = {f"Molecules/{d.name}": mol_2_idx[d] for d in watch_mols}
        self.cltr = cltr
        self.log_scalars(step=0, dtime=0.0)
        self.log_imgs(step=0)

    def log_scalars(
        self, step: int, dtime: float, kwargs: dict[str, float] | None = None
    ):
        n_cells = self.cltr.world.n_cells
        molecule_map = self.cltr.world.molecule_map
        cell_molecules = self.cltr.world.cell_molecules
        for scalar, idx in self.molecules.items():
            tag = f"{scalar}[ext]"
            self.writer.add_scalar(tag, molecule_map[idx].mean(), step)

        if n_cells > 0:
            self.writer.add_scalar("Cells/Total", n_cells, step)
            mean_surv = self.cltr.world.cell_lifetimes.float().mean()
            mean_divis = self.cltr.world.cell_divisions.float().mean()
            genome_lens = [len(d) for d in self.cltr.world.cell_genomes]
            self.writer.add_scalar("Cells/Survival", mean_surv, step)
            self.writer.add_scalar("Cells/Divisions", mean_divis, step)
            self.writer.add_scalar("Cells/cPD", self.cltr.cpd, step)
            self.writer.add_scalar("Cells/GrowthRate", self.cltr.growth_rate, step)
            self.writer.add_scalar("Cells/GenomeSize", sum(genome_lens) / n_cells, step)
            for scalar, idx in self.molecules.items():
                tag = f"{scalar}[int]"
                self.writer.add_scalar(tag, cell_molecules[:, idx].mean(), step)

        self.writer.add_scalar("Other/TimePerStep[s]", dtime, step)
        self.writer.add_scalar("Other/Split", self.cltr.split_i, step)
        self.writer.add_scalar("Other/Progress", self.cltr.progress, step)
        if kwargs is not None:
            for key, val in kwargs.items():
                self.writer.add_scalar(key, val, step)

    def log_imgs(self, step: int):
        cell_map = self.cltr.world.cell_map
        self.writer.add_image("Maps/Cells", cell_map, step, dataformats="WH")


class ChemoStatLogger(Logger):
    """Tensorboard logger for ChemoStat"""

    def __init__(
        self,
        trial_dir: Path,
        hparams: dict,
        cltr: ChemoStat,
        watch_mols: list[ms.Molecule],
    ):
        super().__init__(trial_dir=trial_dir, hparams=hparams)
        mol_2_idx = cltr.world.chemistry.mol_2_idx
        self.molecules = {f"Molecules/{d.name}": mol_2_idx[d] for d in watch_mols}
        self.cltr = cltr
        self.log_scalars(step=0, dtime=0.0)
        self.log_imgs(step=0)

    def log_scalars(
        self, step: int, dtime: float, kwargs: dict[str, float] | None = None
    ):
        n_cells = self.cltr.world.n_cells
        molecule_map = self.cltr.world.molecule_map
        cell_molecules = self.cltr.world.cell_molecules
        for scalar, idx in self.molecules.items():
            tag = f"{scalar}[ext]"
            self.writer.add_scalar(tag, molecule_map[idx].mean(), step)

        if n_cells > 0:
            self.writer.add_scalar("Cells/Total", n_cells, step)
            mean_surv = self.cltr.world.cell_lifetimes.float().mean()
            mean_divis = self.cltr.world.cell_divisions.float().mean()
            genome_lens = [len(d) for d in self.cltr.world.cell_genomes]
            self.writer.add_scalar("Cells/Survival", mean_surv, step)
            self.writer.add_scalar("Cells/Divisions", mean_divis, step)
            self.writer.add_scalar("Cells/GenomeSize", sum(genome_lens) / n_cells, step)
            for scalar, idx in self.molecules.items():
                tag = f"{scalar}[int]"
                self.writer.add_scalar(tag, cell_molecules[:, idx].mean(), step)

        self.writer.add_scalar("Other/TimePerStep[s]", dtime, step)
        self.writer.add_scalar("Other/Progress", self.cltr.progress, step)
        if kwargs is not None:
            for key, val in kwargs.items():
                self.writer.add_scalar(key, val, step)

    def log_imgs(self, step: int):
        cell_map = self.cltr.world.cell_map
        self.writer.add_image("Maps/Cells", cell_map, step, dataformats="WH")
