import json
from datetime import date

import pytest
from cloudpathlib import AnyPath
from hubverse_transform.hub_config import HubConfig


@pytest.fixture()
def admin_config() -> dict:
    """
    Return a Hubverse admin.json config.
    """
    admin_config_dict = {
        "schema_version": "https://link_to_admin_schema.json",
        "name": "Borg Assimilation Forecast Hub",
        "maintainer": "Starfleet",
        "contact": {"name": "Katherine Janeway", "email": "janeway@starfleet.com"},
        "repository_host": "GitHub",
        "repository_url": "https://github.com/starfleet/borg-assimilation-forecast-hub",
        "file_format": ["csv"],
        "model_output_dir": "model-output",
        "cloud": {
            "enabled": True,
            "host": {"name": "aws", "storage_service": "s3", "storage_location": "borg-assimilation-forecast"},
        },
    }

    return admin_config_dict


@pytest.fixture()
def tasks_config() -> dict:
    """
    Return a Hubverse admin.json config.
    """
    task_config_dict = {
        "schema_version": "https://link_to_tasks_schema.json",
        "rounds": [
            {
                "round_id_from_variable": True,
                "round_id": "reference_date",
                "model_tasks": [
                    {
                        "task_ids": {
                            "reference_date": {"required": None, "optional": ["2024-07-13", "2024-07-21"]},
                            "target": {"required": ["borg growth rate change"]},
                            "horizon": {"required": None, "optional": [-1, 0, 1, 2, 3]},
                            "location": {"required": ["Earth", "Vulcan", "789"], "optional": ["Ryza", "123"]},
                            "target_end_date": {"required": None, "optional": ["2024-07-20", "2024-07-27"]},
                        },
                        "output_type": {
                            "pmf": {
                                "output_type_id": {"required": ["decrease", "stable", "increase"]},
                                "value": {"type": "double", "minimum": 0, "maximum": 1},
                            }
                        },
                    },
                    {
                        "task_ids": {
                            "reference_date": {"required": None, "optional": ["2024-08-13", "2024-07-21"]},
                            "target": {"required": ["wk number assimilations"]},
                            "horizon": {"required": ["one", "two", "three"]},
                            "location": {"required": ["Earth", "Bajor", "123"], "optional": []},
                            "target_end_date": {"required": ["1999-12-31"], "optional": ["2024-08-20", "2024-07-27"]},
                        },
                        "output_type": {
                            "quantile": {
                                "output_type_id": {"required": [0.25, 0.5, 0.75], "optional": None},
                                "value": {"type": "double", "minimum": 0},
                            }
                        },
                    },
                ],
                "submissions_due": {"relative_to": "reference_date", "start": -6, "end": -3},
            },
            {
                "round_name": "Round 2",
                "model_tasks": [
                    {
                        "task_ids": {
                            "age": {"required": [10, 20, 30], "optional": [40, 50]},
                            "target": {"required": ["starfleet entrance exam score"]},
                        },
                        "output_type": {
                            "mean": {
                                "output_type_id": {"required": ["NA"], "optional": None},
                                "value": {"type": "double", "minimum": 0, "maximum": 1},
                            }
                        },
                    }
                ],
                "submissions_due": {"relative_to": "reference_date", "start": -100, "end": 5},
            },
        ],
    }

    return task_config_dict


@pytest.fixture()
def hubverse_hub(tmp_path, admin_config, tasks_config):
    """
    Return a Hubverse hub with sample config files.
    """
    hub_path = tmp_path / "hubverse-hub"
    hub_path.mkdir()
    config_dir = hub_path / "hub-config"
    config_dir.mkdir()

    # write admin config
    admin_config_file = config_dir / "admin.json"
    with open(admin_config_file, "w") as f:
        json.dump(admin_config, f)

    # write tasks config
    task_config_file = config_dir / "tasks.json"
    with open(task_config_file, "w") as f:
        json.dump(tasks_config, f)

    return hub_path


def test_hub_config_new_instance(hubverse_hub, admin_config, tasks_config):
    hub_path = AnyPath(hubverse_hub)
    hc = HubConfig(hub_path)

    assert hc.hub_config_path == hub_path / "hub-config"
    assert hc.admin == admin_config
    assert hc.tasks == tasks_config
    assert hc.hub_name == "Borg Assimilation Forecast Hub"
    assert hc.hub_name in hc.__str__()
    assert str(hc.hub_config_path) in hc.__repr__()


def test_hub_missing_admin_config(hubverse_hub):
    hub_path = AnyPath(hubverse_hub)

    with pytest.raises(FileNotFoundError):
        hc = HubConfig(hub_path, "missing-config-dir")
        print(hc)


def test_hub_missing_tasks_config(hubverse_hub):
    hub_path = AnyPath(hubverse_hub)

    with pytest.raises(FileNotFoundError):
        hc = HubConfig(hub_path, tasks_file="missing-tasks.json")
        print(hc)


def test_get_task_id_values(hubverse_hub):
    hub_path = AnyPath(hubverse_hub)
    hc = HubConfig(hub_path)

    assert hc.get_task_id_values() == {
        "reference_date": {"2024-07-13", "2024-07-21", "2024-08-13"},
        "target": {"borg growth rate change", "wk number assimilations", "starfleet entrance exam score"},
        "horizon": {-1, 0, 1, 2, 3, "one", "two", "three"},
        "location": {"Earth", "Vulcan", "789", "Ryza", "123", "Bajor"},
        "target_end_date": {"2024-07-20", "2024-07-27", "1999-12-31", "2024-08-20"},
        "age": {10, 20, 30, 40, 50},
    }


@pytest.mark.parametrize(
    "value, expected_type",
    [
        ("a string", str),
        (123, int),
        (123.45, float),
        ("2024-07-13", date),
        (False, bool),
    ],
)
def test_get_data_type(hubverse_hub, value, expected_type):
    hub_path = AnyPath(hubverse_hub)
    hc = HubConfig(hub_path)

    assert hc._get_data_type(value) == expected_type
