from pathlib import Path
import json
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams as get_summary
import magicsoup as ms
from .culture import BatchCulture, ChemoStat

# TODO: write checkpointer, so that it can save automatically at exit


class Checkpointer:
    """Checkpointing base"""

    def __init__(
        self,
        trial_dir: Path,
        hparams: dict,
        scalar_freq=5,
        img_freq=50,
        save_freq=50,
    ):
        self.scalar_freq = scalar_freq
        self.img_freq = img_freq
        self.save_freq = save_freq
        self.trial_dir = trial_dir

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

    def log_scalars(self, dtime: float):
        raise NotImplementedError

    def log_imgs(self):
        raise NotImplementedError

    def save_state(self):
        raise NotImplementedError

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.writer.close()


class BatchCultureCheckpointer(Checkpointer):
    """Checkpointing for batch culture"""

    def __init__(self, cltr: BatchCulture, watch_mols: list[ms.Molecule], **kwargs):
        super().__init__(**kwargs)
        mol_2_idx = cltr.world.chemistry.mol_2_idx
        self.molecules = {f"Molecules/{d.name}": mol_2_idx[d] for d in watch_mols}
        self.cltr = cltr
        self.log_scalars(dtime=0.0)
        self.log_imgs()
        self.save_state()

    def log_scalars(self, dtime: float, kwargs: dict[str, float] | None = None):
        step = self.cltr.step_i
        if step % self.scalar_freq != 0:
            return

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

    def log_imgs(self):
        step = self.cltr.step_i
        if step % self.img_freq != 0:
            return

        cell_map = self.cltr.world.cell_map
        self.writer.add_image("Maps/Cells", cell_map, step, dataformats="WH")

    def save_state(self):
        step = self.cltr.step_i
        if step % self.save_freq != 0:
            return

        self.cltr.world.save_state(statedir=self.trial_dir / f"step={step}")


class ChemoStatCheckpointer(Checkpointer):
    """Checkpointing for ChemoStat"""

    def __init__(self, cltr: ChemoStat, watch_mols: list[ms.Molecule], **kwargs):
        super().__init__(**kwargs)
        mol_2_idx = cltr.world.chemistry.mol_2_idx
        self.molecules = {f"Molecules/{d.name}": mol_2_idx[d] for d in watch_mols}
        self.cltr = cltr
        self.log_scalars(dtime=0.0)
        self.log_imgs()
        self.save_state()

    def log_scalars(self, dtime: float, kwargs: dict[str, float] | None = None):
        step = self.cltr.step_i
        if step % self.scalar_freq != 0:
            return

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

    def log_imgs(self):
        step = self.cltr.step_i
        if step % self.img_freq != 0:
            return

        cell_map = self.cltr.world.cell_map
        self.writer.add_image("Maps/Cells", cell_map, step, dataformats="WH")

    def save_state(self):
        step = self.cltr.step_i
        if step % self.save_freq != 0:
            return

        self.cltr.world.save_state(statedir=self.trial_dir / f"step={step}")
