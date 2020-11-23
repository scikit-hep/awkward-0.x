import pytest
pandas = pytest.importorskip("pandas")


@pytest.fixture
def awkward0_pandas():
    from awkward0 import pandas
    return pandas


def test_import_pandas(awkward0_pandas):
    pass
