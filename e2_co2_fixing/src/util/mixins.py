import json
from pathlib import Path

_energy_mins = {d: 0.1 for d in ("ATP", "NADPH", "QH2", "Fe-red", "FADH2", "Fd-red")}


class StateMixin:
    """Adds saving/loading states"""

    def _get_file(self, statedir: Path) -> Path:
        return statedir / f"{type(self).__name__}.json"

    def _get_simple_attrs(self) -> list[str]:
        types = (int, float, str, dict, list, tuple)
        return [d for d in dir(self) if isinstance(d, types)]

    def _get_mixin_attrs(self) -> list[str]:
        return [d for d in dir(self) if isinstance(d, StateMixin)]

    def save_state(self, statedir: Path) -> None:
        for attr in self._get_mixin_attrs():
            getattr(self, attr).save_state(statedir=statedir)

        file = self._get_file(statedir)
        simple_attrs = self._get_simple_attrs()
        data = {d: getattr(self, d) for d in simple_attrs}
        file.write_text(json.dumps(data))

    def load_state(self, statedir: Path) -> None:
        for attr in self._get_mixin_attrs():
            getattr(self, attr).load_state(statedir=statedir)

        file = self._get_file(statedir)
        simple_attrs = self._get_simple_attrs()
        data = json.loads(file.read_text())
        for attr in simple_attrs:
            setattr(self, attr, data[attr])
