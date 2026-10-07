import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    log_loss,
)


# ============================================================
# DEFAULT MARKET FEATURES
# ============================================================

MARKET_FEATURES = [
    "return_1d",
    "return_3d",
    "return_7d",
    "volatility_7d",
    "volatility_14d",
    "volatility_30d",
    "sma_ratio_7",
    "sma_ratio_30",
    "volume_change",
    "rsi_14",
    "macd",
    "macd_signal",
]

TRENDS_FEATURES = [
    "google_trends",
    "google_trends_change",
    "google_trends_3m_avg",
]


# ============================================================
# SVM
# ============================================================

def create_svm(C=1.0, gamma="scale"):

    model = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "svm",
            SVC(
                kernel="rbf",
                C=C,
                gamma=gamma,
                probability=False,
                random_state=42,
            )
        ),
    ])

    return model


# ============================================================
# CHRONOLOGICAL PROBABILITY CALIBRATOR
# ============================================================

class ChronologicalProbabilityCalibrator:

    def __init__(self):

        self.calibrator = LogisticRegression(
            solver="lbfgs",
            random_state=42,
        )

    def fit(self, decision_scores, y):

        decision_scores = np.asarray(
            decision_scores
        ).reshape(-1, 1)

        y = np.asarray(y).astype(int)

        if len(np.unique(y)) < 2:
            raise ValueError(
                "Calibration data must contain both classes."
            )

        self.calibrator.fit(
            decision_scores,
            y
        )

        return self

    def predict_proba(self, decision_scores):

        decision_scores = np.asarray(
            decision_scores
        ).reshape(-1, 1)

        return self.calibrator.predict_proba(
            decision_scores
        )[:, 1]


# ============================================================
# FIT SVM + CALIBRATION
# ============================================================

def fit_calibrated_model(
    train_df,
    feature_cols,
    target_col="target",
    C=1.0,
    gamma="scale",
    calibration_fraction=0.20,
):

    n = len(train_df)

    calibration_size = max(
        20,
        int(n * calibration_fraction)
    )

    svm_train_end = n - calibration_size

    if svm_train_end <= 0:
        raise ValueError(
            "Not enough observations for calibration."
        )

    svm_train = train_df.iloc[
        :svm_train_end
    ]

    calibration = train_df.iloc[
        svm_train_end:
    ]

    if svm_train[target_col].nunique() < 2:
        raise ValueError(
            "SVM training data contains only one class."
        )

    if calibration[target_col].nunique() < 2:
        raise ValueError(
            "Calibration data contains only one class."
        )

    svm = create_svm(
        C=C,
        gamma=gamma
    )

    svm.fit(
        svm_train[feature_cols],
        svm_train[target_col]
    )

    calibration_scores = svm.decision_function(
        calibration[feature_cols]
    )

    calibrator = ChronologicalProbabilityCalibrator()

    calibrator.fit(
        calibration_scores,
        calibration[target_col]
    )

    return svm, calibrator


# ============================================================
# PREDICT PROBABILITY
# ============================================================

def predict_probability(
    svm,
    calibrator,
    X,
):

    scores = svm.decision_function(X)

    probabilities = calibrator.predict_proba(
        scores
    )

    return probabilities


# ============================================================
# EXPECTED CALIBRATION ERROR
# ============================================================

def expected_calibration_error(
    y_true,
    probabilities,
    n_bins=10,
):

    y_true = np.asarray(y_true)

    probabilities = np.asarray(
        probabilities
    )

    bin_edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1
    )

    ece = 0.0

    for i in range(n_bins):

        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        if i == n_bins - 1:

            mask = (
                (probabilities >= lower)
                &
                (probabilities <= upper)
            )

        else:

            mask = (
                (probabilities >= lower)
                &
                (probabilities < upper)
            )

        if not np.any(mask):
            continue

        mean_probability = probabilities[
            mask
        ].mean()

        observed_frequency = y_true[
            mask
        ].mean()

        bin_weight = mask.mean()

        ece += (
            bin_weight
            *
            abs(
                mean_probability
                -
                observed_frequency
            )
        )

    return ece


# ============================================================
# WALK-FORWARD VALIDATION
# ============================================================

def walk_forward(
    df,
    feature_cols=None,
    target_col="target_1d",
    train_size=500,
    validation_size=100,
    test_size=30,
    step=30,
):

    # --------------------------------------------------------
    # Default features
    # --------------------------------------------------------

    if feature_cols is None:
        feature_cols = MARKET_FEATURES

    feature_cols = list(feature_cols)

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    df = (
        df
        .sort_values("timestamp")
        .reset_index(drop=True)
        .copy()
    )

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_columns = (
        feature_cols
        + [
            target_col,
            "timestamp",
            "close",
            "future_return_1d",
        ]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required columns: "
            + str(missing_columns)
        )

    # --------------------------------------------------------
    # Clean data
    # --------------------------------------------------------

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.dropna(
        subset=feature_cols + [target_col]
    ).reset_index(drop=True)

    df[target_col] = (
        df[target_col]
        .astype(int)
    )

    unique_targets = sorted(
        df[target_col].unique()
    )

    if not set(unique_targets).issubset({0, 1}):

        raise ValueError(
            "Target must contain only 0 and 1. "
            f"Found: {unique_targets}"
        )

    # --------------------------------------------------------
    # Internal target name
    # --------------------------------------------------------

    if target_col != "target":

        df = df.rename(
            columns={
                target_col: "target"
            }
        )

        target_col = "target"

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    results = []

    start = 0

    # ========================================================
    # WALK-FORWARD LOOP
    # ========================================================

    while (
        start
        + train_size
        + validation_size
        + test_size
        <= len(df)
    ):

        train_end = (
            start
            + train_size
        )

        validation_end = (
            train_end
            + validation_size
        )

        test_end = (
            validation_end
            + test_size
        )

        # ----------------------------------------------------
        # Train / validation / test
        # ----------------------------------------------------

        train = df.iloc[
            :train_end
        ]

        validation = df.iloc[
            train_end:validation_end
        ]

        test = df.iloc[
            validation_end:test_end
        ]

        # ----------------------------------------------------
        # Class checks
        # ----------------------------------------------------

        if train["target"].nunique() < 2:
            start += step
            continue

        if validation["target"].nunique() < 2:
            start += step
            continue

        if test["target"].nunique() < 2:
            start += step
            continue

        # ====================================================
        # HYPERPARAMETER SEARCH
        # ====================================================

        parameter_grid = [
            (0.1, "scale"),
            (1.0, "scale"),
            (10.0, "scale"),
            (1.0, 0.01),
            (1.0, 0.1),
        ]

        best_C = None
        best_gamma = None
        best_brier = np.inf

        for C, gamma in parameter_grid:

            try:

                svm, calibrator = fit_calibrated_model(
                    train_df=train,
                    feature_cols=feature_cols,
                    target_col="target",
                    C=C,
                    gamma=gamma,
                )

                validation_probability = (
                    predict_probability(
                        svm,
                        calibrator,
                        validation[feature_cols],
                    )
                )

                validation_brier = (
                    brier_score_loss(
                        validation["target"],
                        validation_probability,
                    )
                )

                if validation_brier < best_brier:

                    best_brier = validation_brier
                    best_C = C
                    best_gamma = gamma

            except ValueError:

                continue

        if best_C is None:

            start += step
            continue

        # ====================================================
        # FINAL DEVELOPMENT MODEL
        # ====================================================

        development = pd.concat(
            [
                train,
                validation,
            ],
            ignore_index=True,
        )

        final_svm, final_calibrator = (
            fit_calibrated_model(
                train_df=development,
                feature_cols=feature_cols,
                target_col="target",
                C=best_C,
                gamma=best_gamma,
            )
        )

        # ====================================================
        # UNSEEN TEST
        # ====================================================

        test_probability = (
            predict_probability(
                final_svm,
                final_calibrator,
                test[feature_cols],
            )
        )

        test_prediction = (
            test_probability >= 0.50
        ).astype(int)

        # ====================================================
        # STORE RESULTS
        # ====================================================

        temp = test[
            [
                "timestamp",
                "close",
                "future_return_1d",
            ]
        ].copy()

        temp["y_true"] = (
            test["target"].values
        )

        temp["prediction"] = (
            test_prediction
        )

        temp["prob_positive"] = (
            test_probability
        )

        temp["prob_negative"] = (
            1.0 -
            test_probability
        )

        temp["confidence"] = (
            np.maximum(
                test_probability,
                1.0 - test_probability,
            )
        )

        temp["selected_C"] = best_C

        temp["selected_gamma"] = (
            str(best_gamma)
        )

        temp["validation_brier"] = (
            best_brier
        )

        temp["fold_start"] = start

        temp["train_end"] = train_end

        temp["validation_end"] = (
            validation_end
        )

        temp["test_end"] = test_end

        results.append(temp)

        # ----------------------------------------------------
        # Move forward
        # ----------------------------------------------------

        start += step

    # ========================================================
    # FINAL RESULT
    # ========================================================

    if not results:

        raise ValueError(
            "Not enough rows for the "
            "walk-forward configuration."
        )

    return pd.concat(
        results,
        ignore_index=True
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate(predictions):

    y_true = (
        predictions["y_true"]
        .astype(int)
    )

    prediction = (
        predictions["prediction"]
        .astype(int)
    )

    probability = (
        predictions["prob_positive"]
        .astype(float)
    )

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        prediction
    )

    precision = precision_score(
        y_true,
        prediction,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        prediction,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        prediction,
        zero_division=0
    )

    # --------------------------------------------------------
    # ROC-AUC
    # --------------------------------------------------------

    try:

        roc_auc = roc_auc_score(
            y_true,
            probability
        )

    except ValueError:

        roc_auc = np.nan

    # --------------------------------------------------------
    # Brier
    # --------------------------------------------------------

    brier = brier_score_loss(
        y_true,
        probability
    )

    # --------------------------------------------------------
    # Log loss
    # --------------------------------------------------------

    logloss = log_loss(
        y_true,
        probability,
        labels=[0, 1]
    )

    # --------------------------------------------------------
    # ECE
    # --------------------------------------------------------

    ece = expected_calibration_error(
        y_true,
        probability
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "brier": brier,
        "log_loss": logloss,
        "ece": ece,
    }