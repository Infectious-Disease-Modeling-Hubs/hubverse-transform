# mypy: disable-error-code="operator, arg-type"

import itertools
import json
import os
from datetime import date

from cloudpathlib import AnyPath


class HubConfig:
    """
    Represents the configuration of a Hubverse hub.

    Attributes
    ----------
    """

    def __init__(
        self,
        hub_path: os.PathLike,
        config_dir: str = "hub-config",
        admin_file: str = "admin.json",
        tasks_file: str = "tasks.json",
    ):
        """
        Parameters
        ----------
        hub_path : os.PathLike
            The location of a Hubverse hub
            (e.g., S3 bucket name, local filepath).
        config_dir : str, default="hub-config"
            Directory containing the hub's configuration files
        admin_file : str, default="admin.json"
            Filename of the hub's admin configuration file.
        tasks_file : str, default="tasks.json"
            Filename of the hub's tasks configuration file.
        """

        self.hub_config_path = AnyPath(hub_path) / config_dir
        self.admin = self._get_admin_config(admin_file)
        self.tasks = self._get_tasks_config(tasks_file)
        self.hub_name = self.admin.get("name", "unknown hub name")

    def __repr__(self):
        return f"HubConfig('{str(self.hub_config_path)}')"

    def __str__(self):
        return f"Hubverse config information for {self.hub_name}."

    def get_task_id_values(self) -> dict[str, set]:
        """
        Return a dict of hub task ids and values for a specific round.

        Returns
        -------
        model_tasks : dict[str, list]
            A mapping of tasks ids to their possible values, as
            defined in a hub's tasks.json configuration file.
        """

        tasks = self.tasks
        rounds = tasks.get("rounds", [])

        model_tasks_dict: dict[str, set] = dict()

        for r in rounds:
            for task_set in r.get("model_tasks", []):
                task_values = self._get_task_id_values(task_set)
                for key, value in task_values.items():
                    model_tasks_dict[key] = model_tasks_dict.get(key, set()) | value

        return model_tasks_dict

    def _get_admin_config(self, admin_file: str):
        """Read a Hubverse hub's admin configuration file."""
        admin_path = self.hub_config_path / admin_file

        if not admin_path.exists():
            raise FileNotFoundError(f"Hub admin config not found at {str(admin_path)}")

        with admin_path.open() as f:
            return json.loads(f.read())

    def _get_tasks_config(self, tasks_file: str):
        """Read a Hubverse hub's tasks configuration file."""
        tasks_path = self.hub_config_path / tasks_file

        if not tasks_path.exists():
            raise FileNotFoundError(f"Hub tasks config not found at {str(tasks_path)}")

        with tasks_path.open() as f:
            return json.loads(f.read())

    def _get_task_id_values(self, task_set: dict) -> dict[str, set]:
        """Return a dict of ids and values for a specific modeling task."""
        task_id_values = dict()

        # create a dictionary of all tasks ids and values for this task
        task_ids = {task_id[0]: task_id[1] for task_id in task_set.get("task_ids", {}).items()}

        # flatten the dictionary values for each task_id (i.e., combine "required" and "optional" lists)
        for task_id in task_ids.items():
            task_id_values[task_id[0]] = set(
                itertools.chain.from_iterable([value or [] for value in task_id[1].values()])
            )

        return task_id_values

    def _get_data_type(self, value: int | bool | str | date | float) -> type:
        """Return the data type of a value."""
        data_type = type(value)

        if data_type == str:
            try:
                date.fromisoformat(value)
                data_type = date
            except ValueError:
                pass

        return data_type
