from typing import Tuple
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def make_holdout_split(X, y, test_size: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(X, y))
    return train_idx, test_idx
