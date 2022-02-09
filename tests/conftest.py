import pytest
import os

# from model_directory.config import config


@pytest.fixture()
def test_input_data():
    pass


def test_hello_world():
    x = 3
    assert(x + 2 == 5, "Passed test")
