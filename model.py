"""
NumPy House Price Regression

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impute_nan_with_mean
import numpy as np

def impute_nan_with_mean(X):
    """Replace every NaN in X with that column's nan-aware mean (all-NaN cols -> 0).

    Args:
        X: (N, F) array-like of floats, may contain NaN.

    Returns:
        (N, F) float ndarray with no NaNs.
    """
    # Convert input to a float numpy array copy
    Xc = np.array(X, dtype=float, copy=True)
    
    # Compute nan-aware mean along each column (axis=0)
    col_means = np.nanmean(Xc, axis=0)
    
    # Replace NaN means (all-NaN columns) with 0.0
    col_means = np.nan_to_num(col_means, nan=0.0)
    
    # Replace NaNs in Xc with the corresponding column mean via broadcasting
    return np.where(np.isnan(Xc), col_means, Xc)

# Step 2 - compute_iqr_bounds
def compute_iqr_bounds(X, k=1.5):
    # TODO: Compute per-column lower/upper clip bounds using the IQR rule.
    lqr = np.percentile(X, 25, axis=0)
    uqr = np.percentile(X, 75, axis=0)
    IQR = uqr-lqr
    lower = lqr-k*IQR 
    upper = uqr+k*IQR
    return lower, upper

# Step 3 - clip_columns
def clip_columns(X, lower, upper):
    # TODO: Clip every entry of a feature matrix to per-column lower/upper bounds.
    return np.clip(X, lower, upper)

# Step 4 - make_ratio_feature
def make_ratio_feature(numerator, denominator, eps=1e-8):
    # TODO: Form a derived ratio feature from two 1-D arrays with safe division.
    denominator += eps 
    return numerator/denominator

# Step 5 - append_column
def append_column(X, col):
    # TODO: Horizontally append one 1-D feature column onto a design matrix.
    return np.column_stack([X, col])

# Step 6 - one_hot_encode
def one_hot_encode(labels):
    # TODO: Convert a 1-D array of categorical labels into a dense binary one-hot matrix.
    labels = np.asarray(labels)
    uni = np.unique(labels)
    return (labels[:, None] == uni[None, :]).astype(float)

# Step 7 - fit_standardizer
def fit_standardizer(X):
    # TODO: Compute per-column mean and std used to standardize features...
    mean, std = np.mean(X, axis=0), np.std(X, axis=0)
    std = np.where(std == 0, 1.0, std)
    return mean, std

# Step 8 - apply_standardizer
def apply_standardizer(X, mean, std):
    # TODO: Return the scaled matrix (X - mean) / std via broadcasting.
    return (X-mean)/std

# Step 9 - add_bias_column
def add_bias_column(X):
    # TODO: Prepend a column of ones to a 2-D feature matrix X...
    n = X.shape[0]
    return np.hstack([np.ones((n, 1)), X])

# Step 10 - make_shuffled_indices
def make_shuffled_indices(n_samples, seed):
    """Create a reproducibly shuffled permutation of row indices."""
    rng = np.random.RandomState(seed)
    return rng.permutation(n_samples)

# Step 11 - partition_indices
def partition_indices(indices, train_ratio, val_ratio):
    # TODO: Split a shuffled index array into train, validation, and test index arrays.
    n = len(indices)
    n_train = int(n*train_ratio)
    n_val = int(n*val_ratio)
    return indices[:n_train], indices[n_train:n_val+n_train], indices[n_train+n_val:]

# Step 12 - subset_xy
def subset_xy(X, y, indices):
    # TODO: Select the rows of X and y at the given indices.
    return X[indices], y[indices]

# Step 13 - ols_fit
def ols_fit(X, y):
    # TODO: return the ordinary-least-squares weight vector for a linear model.
    a = X.T @ X 
    b = X.T @ y 
    return np.linalg.solve(a, b)

# Step 14 - ols_predict
def ols_predict(X, theta):
    # TODO: Predict continuous targets with a fitted linear model.
    return X @ theta

# Step 15 - mean_absolute_error
def mean_absolute_error(y_true, y_pred):
    # TODO: return the mean absolute error between targets and predictions
    return np.mean(abs(y_true-y_pred))

# Step 16 - root_mean_squared_error
def root_mean_squared_error(y_true, y_pred):
    """Compute root mean squared error between targets and predictions.

    Args:
        y_true (np.ndarray): Ground-truth targets, shape (N,).
        y_pred (np.ndarray): Predicted targets, shape (N,).

    Returns:
        float: RMSE value.
    """
    # TODO: return the root mean squared error as a Python float
    return np.sqrt(np.mean((y_true-y_pred)**2))

# Step 17 - r_squared
def r_squared(y_true, y_pred):
    # TODO: Compute R^2 = 1 - SS_res/SS_tot (return 0.0 if SS_tot is 0)...
    SS_res = np.sum((y_true-y_pred)**2)
    mean = np.mean(y_true)
    SS_tot = np.sum((y_true-mean)**2)
    if SS_tot == 0:
        return 0.0 
    else:
        return 1-SS_res/SS_tot

# Step 18 - residual_summary
def residual_summary(y_true, y_pred):
    # TODO: Return a compact dict summarizing prediction residuals...
    res = y_true - y_pred 
    return {"mean":np.mean(res), "std":np.std(res), "median_abs":np.median(np.abs(res))}

# Step 19 - prepare_cleaned_features
def prepare_cleaned_features(X, iqr_k=1.5):
    """Impute NaNs then IQR-clip columns to produce a clean numeric matrix.

    Args:
        X: (N, F) array-like of floats, may contain NaN.
        iqr_k: IQR multiplier passed to compute_iqr_bounds (default 1.5).

    Returns:
        (N, F) float ndarray with no NaNs, columns clipped to IQR bounds.
    """
    X_imp = impute_nan_with_mean(X)
    lower, upper = compute_iqr_bounds(X_imp, k=iqr_k)
    return clip_columns(X_imp, lower, upper)

# Step 20 - assemble_feature_matrix
def assemble_feature_matrix(X_num, ratio_num_idx, ratio_den_idx, cat_labels=None):
    # TODO: build an extended feature matrix by appending a derived ratio...
    num, denom = X_num[:, ratio_num_idx], X_num[:, ratio_den_idx]
    ratio = make_ratio_feature(num, denom)
    X_app = append_column(X_num, ratio)
    if cat_labels is not None:
        labels = one_hot_encode(cat_labels)
        X_app = np.hstack([X_app, labels])
    return X_app

# Step 21 - make_train_val_test
def make_train_val_test(X, y, train_ratio, val_ratio, seed):
    # TODO: Shuffle and materialize train/validation/test matrices from X and y...
    n = X.shape[0]
    indices = make_shuffled_indices(n, seed)
    train_idx, val_idx, test_idx = partition_indices(indices, train_ratio, val_ratio)
    out = {}
    out["X_train"], out["y_train"] = subset_xy(X, y, train_idx)
    out["X_val"], out["y_val"] = subset_xy(X, y, val_idx)
    out["X_test"], out["y_test"] = subset_xy(X, y, test_idx)
    return out

# Step 22 - standardize_and_add_bias
def standardize_and_add_bias(splits):
    # TODO: Fit standardizer on train, transform all splits, prepend bias...
    X_train = splits["X_train"]
    X_val = splits["X_val"]
    X_test = splits["X_test"]

    mean, std = fit_standardizer(X_train)
    X_train_std = apply_standardizer(X_train, mean, std)
    X_test_std = apply_standardizer(X_test, mean, std)
    X_val_std = apply_standardizer(X_val, mean, std)

    out = splits.copy()
    out["X_train"] = add_bias_column(X_train_std)
    out["X_test"] = add_bias_column(X_test_std)
    out["X_val"] = add_bias_column(X_val_std)
    return out, mean, std

# Step 23 - evaluate_predictions
def evaluate_predictions(y_true, y_pred):
    # TODO: Bundle MAE, RMSE, R^2, and residual summary into one metrics dict.
    out = {}
    out["mae"] = mean_absolute_error(y_true, y_pred)
    out["rmse"] = root_mean_squared_error(y_true, y_pred)
    out["r2"] = r_squared(y_true, y_pred)
    out["residual_summary"] = residual_summary(y_true, y_pred)
    return out

# Step 24 - house_price_pipeline
def house_price_pipeline(X, y, ratio_num_idx, ratio_den_idx, cat_labels=None, train_ratio=0.7, val_ratio=0.15, seed=42, iqr_k=1.5):
    """Run full clean -> featurize -> split -> standardize -> OLS -> evaluate pipeline."""
    
    # 1. Clean missing values and cap outliers
    X_clean = prepare_cleaned_features(X, iqr_k=iqr_k)
    
    # 2. Assemble feature matrix (ratios + one-hot encodings)
    X_feat = assemble_feature_matrix(X_clean, ratio_num_idx, ratio_den_idx, cat_labels=cat_labels)
    
    # 3. Partition data into reproducible train/val/test splits
    splits = make_train_val_test(X_feat, y, train_ratio, val_ratio, seed)
    
    # 4. Fit standardizer on train, transform all splits, and prepend bias column
    std_splits, _, _ = standardize_and_add_bias(splits)
    
    # 5. Fit OLS model on standardized training set
    theta = ols_fit(std_splits["X_train"], std_splits["y_train"])
    
    # 6. Predict on validation and test sets using fitted theta
    y_val_pred = ols_predict(std_splits["X_val"], theta)
    y_test_pred = ols_predict(std_splits["X_test"], theta)
    
    # 7. Evaluate validation and test predictions
    val_metrics = evaluate_predictions(std_splits["y_val"], y_val_pred)
    test_metrics = evaluate_predictions(std_splits["y_test"], y_test_pred)
    
    # 8. Return final outputs dict with exact required keys
    return {
        "theta": theta,
        "y_test": std_splits["y_test"],
        "y_test_pred": y_test_pred,
        "test_metrics": test_metrics,
        "val_metrics": val_metrics
    }

