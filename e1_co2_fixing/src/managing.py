from pathlib import Path
import json
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams as get_summary
import magicsoup as ms
from .culture import BatchCulture, ChemoStat


class Manager:
    """Boilerplate base"""

    def __init__(
        self,
        trial_dir: Path,
        hparams: dict,
        throttle_light_log=2,
        throttle_fat_log=5,
        throttle_saves=1000,
    ):
        self.throttle_light_log = throttle_light_log
        self.throttle_fat_log = throttle_fat_log
        self.throttle_saves = throttle_saves
        self.trial_dir = trial_dir

        metrics = {"Other/Progress": 0.0}
        self.writer = SummaryWriter(log_dir=trial_dir)
        flat_hparams = {
            k: ", ".join([str(dd) for dd in d]) if isinstance(d, (list, tuple)) else d
            for k, d in hparams.items()
        }
        exp, ssi, sei = get_summary(hparam_dict=flat_hparams, metric_dict=metrics)
        self.writer.file_writer.add_summary(exp)
        self.writer.file_writer.add_summary(ssi)
        self.writer.file_writer.add_summary(sei)

        with open(trial_dir / "hparams.json", "w", encoding="utf-8") as fh:
            json.dump(hparams, fh)

    def close(self):
        self.writer.close()

    def light_log(self, kwargs: dict[str, float] | None = None):
        raise NotImplementedError

    def throttled_light_log(self, step: int, kwargs: dict[str, float] | None = None):
        if step % self.throttle_light_log == 0:
            self.light_log(kwargs=kwargs)

    def fat_log(self):
        raise NotImplementedError

    def throttled_fat_log(self, step: int):
        if step % self.throttle_fat_log == 0:
            self.fat_log()

    def save_state(self):
        raise NotImplementedError

    def throttled_save_state(self, step: int):
        if step % self.throttle_saves == 0:
            self.save_state()

    def __enter__(self):
        self.light_log()
        self.fat_log()
        self.save_state()
        return self

    def __exit__(self, *exc):
        self.light_log()
        self.fat_log()
        self.save_state()
        self.writer.close()


class BatchCultureManager(Manager):
    """Boilerplate for batch culture"""

    def __init__(
        self, cltr: BatchCulture, watch_mols: list[ms.Molecule] | None = None, **kwargs
    ):
        super().__init__(**kwargs)
        if watch_mols is None:
            watch_mols = []
        mol_2_idx = cltr.world.chemistry.mol_2_idx
        self.molecules = {f"Molecules/{d.name}": mol_2_idx[d] for d in watch_mols}
        self.cltr = cltr

    def light_log(self, kwargs: dict[str, float] | None = None):
        step = self.cltr.step_i
        n_cells = self.cltr.world.n_cells
        molecule_map = self.cltr.world.molecule_map
        for scalar, idx in self.molecules.items():
            tag = f"{scalar}[ext]"
            self.writer.add_scalar(tag, molecule_map[idx].mean(), step)

        if n_cells > 0:
            mean_surv = self.cltr.world.cell_lifetimes.float().mean()
            mean_divis = self.cltr.world.cell_divisions.float().mean()
            genome_lens = [len(d) for d in self.cltr.world.cell_genomes]
            self.writer.add_scalar("Cells/Total", n_cells, step)
            self.writer.add_scalar("Cells/Survival", mean_surv, step)
            self.writer.add_scalar("Cells/Divisions", mean_divis, step)
            self.writer.add_scalar("Cells/cPD", self.cltr.cpd, step)
            self.writer.add_scalar("Cells/GrowthRate", self.cltr.growth_rate, step)
            self.writer.add_scalar("Cells/GenomeSize", sum(genome_lens) / n_cells, step)
            cell_molecules = self.cltr.world.cell_molecules
            for scalar, idx in self.molecules.items():
                tag = f"{scalar}[int]"
                self.writer.add_scalar(tag, cell_molecules[:, idx].mean(), step)

        self.writer.add_scalar("Other/Split", self.cltr.split_i, step)
        self.writer.add_scalar("Other/Progress", self.cltr.progress, step)
        if kwargs is not None:
            for key, val in kwargs.items():
                self.writer.add_scalar(key, val, step)

    def fat_log(self):
        step = self.cltr.step_i
        cell_map = self.cltr.world.cell_map
        self.writer.add_image("Maps/Cells", cell_map, step, dataformats="WH")

    def save_state(self):
        step = self.cltr.step_i
        self.cltr.world.save_state(statedir=self.trial_dir / f"step={step}")


class ChemoStatManager(Manager):
    """Boilerplate for ChemoStat"""

    def __init__(
        self, cltr: ChemoStat, watch_mols: list[ms.Molecule] | None = None, **kwargs
    ):
        super().__init__(**kwargs)
        if watch_mols is None:
            watch_mols = []
        mol_2_idx = cltr.world.chemistry.mol_2_idx
        self.molecules = {f"Molecules/{d.name}": mol_2_idx[d] for d in watch_mols}
        self.cltr = cltr

    def light_log(self, kwargs: dict[str, float] | None = None):
        step = self.cltr.step_i
        n_cells = self.cltr.world.n_cells
        molecule_map = self.cltr.world.molecule_map
        for scalar, idx in self.molecules.items():
            tag = f"{scalar}[ext]"
            self.writer.add_scalar(tag, molecule_map[idx].mean(), step)

        if n_cells > 0:
            mean_surv = self.cltr.world.cell_lifetimes.float().mean()
            mean_divis = self.cltr.world.cell_divisions.float().mean()
            genome_lens = [len(d) for d in self.cltr.world.cell_genomes]
            self.writer.add_scalar("Cells/Total", n_cells, step)
            self.writer.add_scalar("Cells/Survival", mean_surv, step)
            self.writer.add_scalar("Cells/Divisions", mean_divis, step)
            self.writer.add_scalar("Cells/GenomeSize", sum(genome_lens) / n_cells, step)
            cell_molecules = self.cltr.world.cell_molecules
            for scalar, idx in self.molecules.items():
                tag = f"{scalar}[int]"
                self.writer.add_scalar(tag, cell_molecules[:, idx].mean(), step)

        self.writer.add_scalar("Other/Progress", self.cltr.progress, step)
        if kwargs is not None:
            for key, val in kwargs.items():
                self.writer.add_scalar(key, val, step)

    def fat_log(self):
        step = self.cltr.step_i
        cell_map = self.cltr.world.cell_map
        self.writer.add_image("Maps/Cells", cell_map, step, dataformats="WH")

    def save_state(self):
        step = self.cltr.step_i
        self.cltr.world.save_state(statedir=self.trial_dir / f"step={step}")
