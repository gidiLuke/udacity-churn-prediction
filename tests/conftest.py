"""Common Test Functionality / Fixtures."""


from pathlib import Path

import pandas as pd
import pytest

from customer_churn_prediction.churn_library import import_data


@pytest.fixture
def sample_data() -> pd.DataFrame:  # type: ignore
    """Generate a simple synthetic dataset for testing."""
    test_path = Path("data/bank_data.csv")
    df = import_data(pth=test_path).head(n=100)
    return df
