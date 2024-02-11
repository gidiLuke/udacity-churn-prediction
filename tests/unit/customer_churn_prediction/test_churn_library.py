"""Tests for the churn_library.py."""

from pathlib import Path

import pandas as pd

from customer_churn_prediction.churn_library import (
    encoder_helper,
    import_data,
    perform_eda,
    perform_feature_engineering,
    train_models,
)
from customer_churn_prediction.utils import logger

logger = logger.get_logger()


def test_import_data():
    """Test import_data function."""
    test_path = Path("data/bank_data.csv")
    df = import_data(pth=test_path)
    assert not df.empty
    assert "Churn" in df.columns
    logger.info(msg="test_import_data passed")


def test_perform_eda(tmpdir: Path, sample_data: pd.DataFrame) -> None:
    """Test perform_eda function."""
    fig_path = Path(tmpdir)
    perform_eda(df=sample_data, fig_path=fig_path)
    assert (fig_path / "churn_history.png").exists()
    assert (fig_path / "customer_age_hist.png").exists()
    assert (fig_path / "correlations.png").exists()
    assert (fig_path / "marital_status_bar.png").exists()
    assert (fig_path / "total_trans_ct.png").exists()
    logger.info(msg="test_perform_eda passed")


def test_encoder_helper(sample_data: pd.DataFrame) -> None:
    """Test encoder_helper function."""
    cat_columns = [
        "Gender",
        "Education_Level",
        "Marital_Status",
        "Income_Category",
        "Card_Category",
    ]
    df_encoded = encoder_helper(df=sample_data, category_columns=cat_columns)
    assert "Gender_Churn" in df_encoded.columns
    # Verify the encoding correctness
    assert df_encoded["Gender_Churn"].notnull().all()
    logger.info(msg="test_encoder_helper passed")


def test_perform_feature_engineering(sample_data: pd.DataFrame) -> None:
    """Test perform_feature_engineering function."""
    X_train, X_test, y_train, y_test = perform_feature_engineering(df=sample_data)
    assert not X_train.empty and not X_test.empty
    assert not y_train.empty and not y_test.empty
    logger.info(msg="test_perform_feature_engineering passed")


def test_train_models(tmpdir, sample_data) -> None:  # type: ignore
    """Test the train_models function to ensure it runs and generates expected output."""
    X_train, X_test, y_train, y_test = perform_feature_engineering(sample_data)

    fig_path = Path(tmpdir.mkdir("figures"))
    model_path = Path(tmpdir.mkdir("models"))

    # Ensure directories are empty before the test
    assert len(list(fig_path.iterdir())) == 0
    assert len(list(model_path.iterdir())) == 0

    # Run the train_models function
    train_models(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        fig_path=fig_path,
        model_path=model_path,
    )

    # Check if the expected files (models and images) were created
    assert len(list(fig_path.iterdir())) > 0, "No figures were saved"
    assert len(list(model_path.iterdir())) > 0, "No models were saved"

    logger.info(msg="test_train_models passed")
