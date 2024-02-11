"""This is a library of functions to find customers who are likely to churn."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split

from customer_churn_prediction.utils.logger import get_logger

logger = get_logger()


def import_data(pth: Path) -> pd.DataFrame:
    """Returns dataframe for the csv found at pth.

    Args:
        pth (Path): path to the csv to be imported

    Returns:
        pd.DataFrame: imported data as pandas dataframe
    """
    logger.info("Importing data from %s", pth)
    df = pd.read_csv(filepath_or_buffer=pth)

    df["Churn"] = df["Attrition_Flag"].apply(
        func=lambda val: 0 if val == "Existing Customer" else 1  # type: ignore
    )
    return df.drop(columns=["Attrition_Flag"])


def perform_eda(df: pd.DataFrame, fig_path: Path) -> None:
    """Perform eda on df and save figures to images folder.

    Args:
        df (pd.DataFrame): DataFrame to perform the EDA on
        fig_path (Path): Path where the figures should be stored

    """
    logger.info("Performing EDA")
    churn_hist = plt.figure(figsize=(20, 10))
    df["Churn"].hist()
    churn_hist.savefig(
        fname=fig_path / "churn_history.png",
    )

    customer_age_hist = plt.figure(figsize=(20, 10))
    df["Customer_Age"].hist()
    customer_age_hist.savefig(fname=fig_path / "customer_age_hist.png")

    marital_status_bar = plt.figure(figsize=(20, 10))
    df.Marital_Status.value_counts(normalize=True).plot(kind="bar")
    marital_status_bar.savefig(fname=fig_path / "marital_status_bar.png")

    total_trans_ct_hist = plt.figure(figsize=(20, 10))
    # Show distributions of 'Total_Trans_Ct' and add a smooth curve obtained using a kernel density
    sns.histplot(
        data=df["Total_Trans_Ct"],  # type: ignore
        stat="density",
        kde=True,
    )  # type: ignore
    total_trans_ct_hist.savefig(fname=fig_path / "total_trans_ct.png")

    correlations = plt.figure(figsize=(20, 10))
    sns.heatmap(
        data=df.select_dtypes(include=["float64", "int64"]).corr(),  # Only for numeric cols
        annot=False,
        cmap="Dark2_r",
        linewidths=2,
    )
    correlations.savefig(fname=fig_path / "correlations.png")
    logger.info("EDA complete, figures saved to %s", fig_path)


def encoder_helper(df: pd.DataFrame, category_columns: list[str]) -> pd.DataFrame:
    """Turns each categorical column into a column proportional to category churn.

    Args:
        df (pd.DataFrame): input dataframe
        category_columns (list[str]): list of columns that contain categories

    Returns:
        pd.DataFrame: DataFrame with encoded categorical columns
    """
    for category in category_columns:
        # gender encoded column
        cat_lst = []
        groups: pd.DataFrame = df[[category, "Churn"]].groupby(by=category).mean()["Churn"]  # type: ignore

        for val in df[category]:
            cat_lst.append(groups.loc[val])
        df[category + "_Churn"] = cat_lst
    return df


def perform_feature_engineering(df: pd.DataFrame):  # type: ignore
    """Performs all feature engineering steps.

    Args:
        df (pd.DataFrame): Encoded input DataFrame

    Returns:
        tuple[pd.DataFrame]: X_train, X_test, y_train, y_test
    """
    logger.info("Performing feature engineering")
    cat_columns = [
        "Gender",
        "Education_Level",
        "Marital_Status",
        "Income_Category",
        "Card_Category",
    ]
    df_encoded = encoder_helper(df=df, category_columns=cat_columns)

    y = df_encoded["Churn"]
    X = pd.DataFrame()
    keep_cols = [
        "Customer_Age",
        "Dependent_count",
        "Months_on_book",
        "Total_Relationship_Count",
        "Months_Inactive_12_mon",
        "Contacts_Count_12_mon",
        "Credit_Limit",
        "Total_Revolving_Bal",
        "Avg_Open_To_Buy",
        "Total_Amt_Chng_Q4_Q1",
        "Total_Trans_Amt",
        "Total_Trans_Ct",
        "Total_Ct_Chng_Q4_Q1",
        "Avg_Utilization_Ratio",
        "Gender_Churn",
        "Education_Level_Churn",
        "Marital_Status_Churn",
        "Income_Category_Churn",
        "Card_Category_Churn",
    ]

    X[keep_cols] = df_encoded[keep_cols]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    logger.info("Feature engineering complete")
    return X_train, X_test, y_train, y_test


def classification_report_image(
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    y_train_preds_lr,  # type: ignore
    y_train_preds_rf,  # type: ignore
    y_test_preds_lr,  # type: ignore
    y_test_preds_rf,  # type: ignore
    fig_path: Path,
) -> None:
    """Produces classification report for training and testing results and stores report as image.

    Args:
        y_train (pd.DataFrame): training response values
        y_test (pd.DataFrame): test response values
        y_train_preds_lr (ArrayLike): training predictions from logistic regression
        y_train_preds_rf (ArrayLike): training predictions from random forest
        y_test_preds_lr (ArrayLike): test predictions from logistic regression
        y_test_preds_rf (ArrayLike): test predictions from random forest
        fig_path (Path): Path where the report images should be stored
    """
    logger.info("Creating classification report images")

    rf_report_plot = plt.figure("figure", figsize=(10, 5))
    # plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    plt.text(0.01, 1.25, str("Random Forest Train"), {"fontsize": 10}, fontproperties="monospace")
    plt.text(
        0.01,
        0.05,
        str(classification_report(y_test, y_test_preds_rf)),
        {"fontsize": 10},
        fontproperties="monospace",
    )  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str("Random Forest Test"), {"fontsize": 10}, fontproperties="monospace")
    plt.text(
        0.01,
        0.7,
        str(classification_report(y_train, y_train_preds_rf)),
        {"fontsize": 10},
        fontproperties="monospace",
    )  # approach improved by OP -> monospace!
    plt.axis("off")
    rf_report_plot.savefig(fname=fig_path / "random_forest_report.png")

    lr_report_plot = plt.figure("figure", figsize=(10, 5))
    plt.text(
        0.01, 1.25, str("Logistic Regression Train"), {"fontsize": 10}, fontproperties="monospace"
    )
    plt.text(
        0.01,
        0.05,
        str(classification_report(y_train, y_train_preds_lr)),
        {"fontsize": 10},
        fontproperties="monospace",
    )  # approach improved by OP -> monospace!
    plt.text(
        0.01, 0.6, str("Logistic Regression Test"), {"fontsize": 10}, fontproperties="monospace"
    )
    plt.text(
        0.01,
        0.7,
        str(classification_report(y_test, y_test_preds_lr)),
        {"fontsize": 10},
        fontproperties="monospace",
    )  # approach improved by OP -> monospace!
    plt.axis("off")

    lr_report_plot.savefig(fname=fig_path / "logistic_regression_report.png")

    logger.info(f"Classification report images saved to {fig_path}")


def feature_importance_plot(model, X_data: pd.DataFrame, output_pth: Path) -> None:  # type: ignore
    """Creates and stores the feature importances plot.

    Args:
        model: model object containing feature_importances_
        X_data (pd.DataFrame): input data
        output_pth (Path): path where the figure should be stored
    """
    logger.info("Creating feature importance plot")
    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(a=importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title(label="Feature Importance")
    plt.ylabel(ylabel="Importance")

    # Add bars
    plt.bar(range(X_data.shape[1]), height=importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(ticks=range(X_data.shape[1]), labels=names, rotation=90)

    plt.savefig(output_pth)
    logger.info(
        f"Feature importance plot saved to {output_pth}",
    )


def plot_roc_curves(
    lrc_model,  # type: ignore
    rfc_model,  # type: ignore
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    fig_path: Path,
) -> None:
    """Plots ROC curve for classifiers.

    Args:
        lrc_model: Linear Regression Classifier model
        rfc_model: Random Forest Classifier model
        X_test (pd.DataFrame): Test data
        y_test (pd.DataFrame): Test response values
        fig_path (Path): Path to store the figures
    """
    fig = plt.figure(figsize=(15, 8))
    ax = fig.gca()
    RocCurveDisplay.from_estimator(estimator=lrc_model, X=X_test, y=y_test, ax=ax, alpha=0.8)
    RocCurveDisplay.from_estimator(estimator=rfc_model, X=X_test, y=y_test, ax=ax, alpha=0.8)
    fig.savefig(fname=fig_path / "roc_curve.png")


def train_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    fig_path: Path,
    model_path: Path,
) -> None:
    """Train, store model results: images + scores, and store models.

    Args:
        X_train (pd.DataFrame): X training data
        X_test (pd.DataFrame): X testing data
        y_train (pd.DataFrame): y training data
        y_test (pd.DataFrame): y testing data
        fig_path (Path): Path where result images should be stored
        model_path (Path): Path where models should be stored
    """
    logger.info("Training models")
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver="lbfgs", max_iter=3000)

    param_grid = {
        "n_estimators": [200, 500],
        "max_features": ["auto", "sqrt"],
        "max_depth": [4, 5, 100],
        "criterion": ["gini", "entropy"],
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    classification_report_image(
        y_train=y_train,
        y_test=y_test,
        y_train_preds_lr=y_train_preds_lr,
        y_test_preds_lr=y_test_preds_lr,
        y_train_preds_rf=y_train_preds_rf,
        y_test_preds_rf=y_test_preds_rf,
        fig_path=fig_path,
    )
    feature_importance_plot(
        model=cv_rfc, X_data=pd.concat(objs=[X_train, X_test]), output_pth=fig_path
    )

    plot_roc_curves(
        lrc_model=lrc,
        rfc_model=cv_rfc.best_estimator_,
        X_test=X_test,
        y_test=y_test,
        fig_path=fig_path,
    )

    logger.info(
        msg=f"Models trained and images saved to {fig_path}",
    )

    # save best model
    joblib.dump(value=cv_rfc.best_estimator_, filename=model_path / "random_forest.pkl")
    joblib.dump(value=lrc, filename=model_path / "logistic_regression.pkl")

    logger.info(msg=f"Models saved to {model_path}")


def main() -> None:
    """Entry point for the churn library."""
    import_path = Path(r"data/bank_data.csv")
    df_import = import_data(pth=import_path)
    fig_path = Path("images")
    perform_eda(df=df_import, fig_path=fig_path)
    X_train, X_test, y_train, y_test = perform_feature_engineering(df=df_import)
    train_models(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        fig_path=fig_path,
        model_path=Path("models"),
    )


if __name__ == "__main__":
    main()
