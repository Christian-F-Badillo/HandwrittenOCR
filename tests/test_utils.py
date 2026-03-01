from utils.datasets import load_data
import pytest


@pytest.mark.parametrize("path", ["tests/examples/test_data.json"])
def test_load_data(path):
    data = load_data(path)

    assert data.data == {"file1": "hola", "file2": "2", "file3": "xd"}
    assert data.index == ["file1", "file2", "file3"]


def test_load_data_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_data("ruta/que/no/existe.json")
