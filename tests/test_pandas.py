import pytest
pandas = pytest.importorskip("pandas")


@pytest.fixture
def awkward_pandas():
    from awkward import pandas
    return pandas


def test_import_pandas(awkward_pandas):
    pass
