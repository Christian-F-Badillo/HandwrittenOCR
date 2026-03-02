from utils.preprocessing import process_img
import pytest


@pytest.mark.parametrize("path", ["tests/examples/test_img.jpg"])
def test_load_data(path):
    img = process_img(path)

    assert img.width == 128
    assert img.height == 32
