import csv
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from hubverse_transform.model_output import ModelOutputHandler
from pyarrow import csv as pyarrow_csv
from pyarrow import fs

# note: the mocker fixture used throughout is provided by pytest-mock


@pytest.fixture()
def model_output_table() -> pa.Table:
    """
    Simple model-output representation to test functions with a PyArrow table input.
    """
    return pa.table(
        {
            "location": ["earth", "vulcan", "seti alpha"],
            "value": [11.11, 22.22, 33.33],
        }
    )


@pytest.fixture()
def model_output_data() -> list[dict[str, Any]]:
    """
    Fixture that returns a list of model-output data representing multiple output types.
    This fixture is used as input for other fixtures that generate temporary .csv and .parquest files for testing.
    """

    model_output_fieldnames = [
        "reference_date",
        "location",
        "horizon",
        "target",
        "output_type",
        "output_type_id",
        "value",
    ]
    model_output_list = [
        ["2420-01-01", "US", "1 light year", "hospitalizations", "quantile", 0.5, 62],
        ["2420-01-01", "US", "1 light year", "hospitalizations", "quantile", 0.75, 50.1],
        ["2420-01-01", "03", 3, "hospitalizations", "mean", None, 33],
        ["1999-12-31", "US", "last month", "hospitalizations", "pmf", "large_increase", 2.597827508665773e-9],
    ]

    model_output_dict_list = [
        {field: value for field, value in zip(model_output_fieldnames, row)} for row in model_output_list
    ]

    return model_output_dict_list


@pytest.fixture()
def test_csv_file(tmpdir, model_output_data) -> str:
    """
    Write a temporary csv file and return the URI for use in tests.
    """
    test_file_dir = Path(tmpdir.mkdir("raw_csv"))
    test_file_name = "2420-01-01-janeswayaddition-voyager1.csv"
    test_file_path = str(test_file_dir.joinpath(test_file_name))

    fieldnames = model_output_data[0].keys()

    with open(test_file_path, "w", newline="") as test_csv_file:
        writer = csv.DictWriter(test_csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in model_output_data:
            writer.writerow(row)

    return str(test_file_path)


@pytest.fixture()
def test_parquet_file(tmpdir, test_csv_file) -> str:
    """
    Write a temporary parquet file and return the URI for use in tests.
    """
    test_file_dir = Path(tmpdir.mkdir("raw_parquet"))
    test_file_name = "2420-01-01-janeswayaddition-voyager1.parquet"
    test_file_path = str(test_file_dir.joinpath(test_file_name))
    local = fs.LocalFileSystem()

    # read our test csv so we can write it back out as a parquet file
    model_output_table = pyarrow_csv.read_csv(test_csv_file)
    with local.open_output_stream(test_file_path) as test_parquet_file:
        pq.write_table(model_output_table, test_parquet_file)

    return str(test_file_path)


def test_new_instance():
    input_uri = "mock:bucket123/raw/prefix1/prefix2/2420-01-01-team_one-model.csv"
    output_uri = "mock:bucket123/prefix1/prefix2"
    mo = ModelOutputHandler(input_uri, output_uri)
    assert mo.input_file == "bucket123/raw/prefix1/prefix2/2420-01-01-team_one-model.csv"
    assert mo.output_path == "bucket123/prefix1/prefix2"
    assert mo.file_name == "2420-01-01-team_one-model"
    assert mo.file_type == ".csv"
    assert mo.round_id == "2420-01-01"
    assert mo.model_id == "team_one-model"


@pytest.mark.parametrize(
    "original_uri, expected_sanitized_uri",
    [
        ("s3://hub-bucket/raw/round name/model name.parquet", "s3://hub-bucket/raw/round%20name/model%20name.parquet"),
        ("mock:hub-bucket/raw/round name/model name.parquet", "mock:hub-bucket/raw/round%20name/model%20name.parquet"),
        ("/tmp/bucket 1/a local file.csv", "/tmp/bucket%201/a%20local%20file.csv"),
    ],
)
def test_sanitize_uri(original_uri, expected_sanitized_uri):
    # instantiate a ModelOutputHandler object with uris we don't care about (as a way to
    # test the sanitize_uri method against various types of URIs supported by the pyarrow filesystem
    mo = ModelOutputHandler("mock:/input-uri/2024-01-01 file.parquet", "mock:/output-uri")

    assert mo.sanitize_uri(original_uri) == expected_sanitized_uri


@pytest.mark.parametrize(
    "file_uri, expected_round_id, expected_model_id",
    [
        ("mock:raw/prefix/2420-01-01-team-model.csv", "2420-01-01", "team-model"),
        ("mock:raw/prefix/2420-01-01-----team-model.parquet", "2420-01-01", "team-model"),
        ("mock:raw/prefix/2420-01-01____teammodelallonestring.csv", "2420-01-01", "teammodelallonestring"),
        ("mock:raw/prefix/2420-01-01____look-at-all-the-hyphens-.csv", "2420-01-01", "look-at-all-the-hyphens-"),
    ],
)
def test_parse_file(file_uri, expected_round_id, expected_model_id):
    mo = ModelOutputHandler(file_uri, "mock:/output-uri")
    assert mo.round_id == expected_round_id
    assert mo.model_id == expected_model_id


@pytest.mark.parametrize(
    "input_uri, output_uri, expected_input_file, expected_output_path, expected_file_name, expected_model_id",
    [
        (
            "mock:bucket123/raw/prefix1/prefix 2/2420-01-01-team-model name with spaces.csv",
            "mock:bucket123/prefix1/prefix 2",
            "bucket123/raw/prefix1/prefix 2/2420-01-01-team-model name with spaces.csv",
            "bucket123/prefix1/prefix 2",
            "2420-01-01-team-model name with spaces",
            "team-model name with spaces",
        ),
        (
            "mock:bucket1.2.3/raw/prefix1/prefix 2/2420-01-01-team-model.name.csv",
            "mock:bucket123/prefix1/~prefix 2",
            "bucket1.2.3/raw/prefix1/prefix 2/2420-01-01-team-model.name.csv",
            "bucket123/prefix1/~prefix 2",
            "2420-01-01-team-model.name",
            "team-model.name",
        ),
        (
            "mock:raw/prefix 1/prefix2/2420-01-01-spáces at end .csv",
            "mock:prefix 1/prefix2",
            "raw/prefix 1/prefix2/2420-01-01-spáces at end.csv",
            "prefix 1/prefix2",
            "2420-01-01-spáces at end",
            "spáces at end",
        ),
        (
            "mock:a space/prefix 1/prefix2/2420-01-01 look ma no hyphens.csv",
            "mock:prefix 1/prefix 🐍",
            "a space/prefix 1/prefix2/2420-01-01 look ma no hyphens.csv",
            "prefix 1/prefix 🐍",
            "2420-01-01 look ma no hyphens",
            "look ma no hyphens",
        ),
    ],
)
def test_new_instance_special_characters(
    input_uri, output_uri, expected_input_file, expected_output_path, expected_file_name, expected_model_id
):
    # ensure spaces and other characters in directory, filename, s3 key, etc. are handled correctly

    mo = ModelOutputHandler(input_uri, output_uri)
    assert mo.input_file == expected_input_file
    assert mo.output_path == expected_output_path
    assert mo.file_name == expected_file_name
    assert mo.model_id == expected_model_id


@pytest.mark.parametrize(
    "s3_key, expected_input_uri, expected_output_uri",
    [
        (
            "raw/prefix1/prefix2/2420-01-01-team-model.csv",
            "s3://bucket123/raw/prefix1/prefix2/2420-01-01-team-model.csv",
            "s3://bucket123/prefix1/prefix2",
        ),
        (
            "raw/model-output/prefix1/prefix2/2420-01-01-team-model.parquet",
            "s3://bucket123/raw/model-output/prefix1/prefix2/2420-01-01-team-model.parquet",
            "s3://bucket123/model-output/prefix1/prefix2",
        ),
        (
            "raw/prefix1/prefix2/prefix3/prefix4/2420-01-01-team-model.csv",
            "s3://bucket123/raw/prefix1/prefix2/prefix3/prefix4/2420-01-01-team-model.csv",
            "s3://bucket123/prefix1/prefix2/prefix3/prefix4",
        ),
        ("raw/2420-01-01-team-model.csv", "s3://bucket123/raw/2420-01-01-team-model.csv", "s3://bucket123/."),
    ],
)
def test_from_s3(mocker, s3_key, expected_input_uri, expected_output_uri):
    mocker.patch("hubverse_transform.model_output.ModelOutputHandler.__init__", return_value=None)
    mo = ModelOutputHandler.from_s3("bucket123", s3_key)
    mo.__init__.assert_called_once_with(expected_input_uri, expected_output_uri)


def test_from_s3_alternate_origin_prefix(mocker):
    mocker.patch("hubverse_transform.model_output.ModelOutputHandler.__init__", return_value=None)
    mo = ModelOutputHandler.from_s3(
        "bucket123",
        "different-raw-prefix/prefix1/prefix2/2420-01-01-team-model.parquet",
        origin_prefix="different-raw-prefix",
    )
    mo.__init__.assert_called_once_with(
        "s3://bucket123/different-raw-prefix/prefix1/prefix2/2420-01-01-team-model.parquet",
        "s3://bucket123/prefix1/prefix2",
    )


def test_from_s3_missing_prefix():
    with pytest.raises(ValueError):
        ModelOutputHandler.from_s3("test-bucket", "prefix1/2420-01-01-team_name-model.csv")


@pytest.mark.parametrize(
    "file_uri, expected_error",
    [
        ("mock:raw/prefix1/prefix2/2420-01-01.csv", ValueError),
        ("mock:raw/prefix1/prefix2/round_id-team-model.csv", ValueError),
        ("mock:raw/prefix1/prefix2/01-02-2440-team-model-name.csv", ValueError),
    ],
)
def test_parse_s3_key_invalid_format(file_uri, expected_error):
    # ensure ValueError is raised for invalid model-output file name format
    with pytest.raises(expected_error):
        ModelOutputHandler(file_uri, "mock:fake-output-uri")


def test_add_columns(model_output_table):
    file_uri = "mock:raw/prefix1/prefix2/2420-01-01-team-model.csv"
    mo = ModelOutputHandler(file_uri, "mock:fake-output-uri")

    result = mo.add_columns(model_output_table)

    # transformed data should have 2 new columns: round_id and model_id
    assert result.num_columns == 4
    assert set(["round_id", "model_id"]).issubset(result.column_names)


def test_added_column_values(model_output_table):
    file_uri = "mock:raw/prefix1/prefix2/2420-01-01-janewaysaddiction-voyager1.csv"
    mo = ModelOutputHandler(file_uri, "mock:fake-output-uri")

    result = mo.add_columns(model_output_table)

    # round_id and model_id columns should each contain a single value
    # that matches round and model as derived from the file name
    assert len(result.column("round_id").unique()) == 1
    assert result.column("round_id").unique()[0].as_py() == "2420-01-01"

    assert len(result.column("model_id").unique()) == 1
    result.column("model_id").unique()[0].as_py() == "janewaysaddiction-voyager1"


def test_read_file_csv(test_csv_file, model_output_table):
    mo = ModelOutputHandler(test_csv_file, "mock:fake-output-uri")
    pyarrow_table = mo.read_file()
    assert len(pyarrow_table) == 4

    # empty output_type_id should transform to null
    output_type_id_col = pyarrow_table.column("output_type_id")
    assert str(output_type_id_col[0]) == "0.5"
    assert pa.compute.is_null(output_type_id_col[2]).as_py() is True
    assert str(output_type_id_col[3]) == "large_increase"


def test_read_file_parquet(test_parquet_file, model_output_table):
    mo = ModelOutputHandler(test_parquet_file, "mock:fake-output-uri")
    pyarrow_table = mo.read_file()
    assert len(pyarrow_table) == 4

    # output_type_id should retain the value from the .csv file, even when the value is empty or "NA"
    output_type_id_col = pyarrow_table.column("output_type_id")
    assert str(output_type_id_col[0]) == "0.5"
    assert str(output_type_id_col[2]) == ""
    assert str(output_type_id_col[3]) == "large_increase"


def test_write_parquet(tmpdir, model_output_table):
    output_dir = str(tmpdir.mkdir("model-output"))

    mo = ModelOutputHandler("mock:raw/prefix1/prefix2/2420-01-01-team-model.csv", output_dir)

    expected_output_file_path = str(Path(*[output_dir, f"{mo.file_name}.parquet"]))
    actual_output_file_path = mo.write_parquet(model_output_table)

    assert actual_output_file_path == expected_output_file_path


def test_transform_model_output_path(test_csv_file, tmpdir):
    output_dir = str(tmpdir.mkdir("model-output"))
    mo = ModelOutputHandler(test_csv_file, output_dir)
    output_uri = mo.transform_model_output()

    input_path = Path(test_csv_file)
    output_path = Path(output_uri)

    assert output_path.suffix == ".parquet"
    assert "raw" not in output_path.parts
    assert input_path.stem in output_path.stem


@pytest.mark.parametrize(
    "file_uri",
    [
        ("mock:raw/prefix1/prefix2/"),
        ("mock:raw/prefix1/prefix2/round_id-team-model.txt"),
        ("mock:photo.jpg"),
        ("mock:raw/prefix1/prefix2/01-02-2440-team-model-name"),
    ],
)
def test_invalid_file_warning(file_uri):
    # ensure ValueError is raised for invalid model-output file name format
    with pytest.raises(UserWarning):
        ModelOutputHandler(file_uri, "mock:fake-output-uri")
