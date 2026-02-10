from sklearn.ensemble import HistGradientBoostingClassifier


def build_hist_gradient_boosting() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.05,
        max_iter=300,
        l2_regularization=0.0,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=2026,
    )
