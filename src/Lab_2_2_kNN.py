import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()
import numpy as np
from collections import Counter


def minkowski_distance(a, b, p=2):
    """
    Compute the Minkowski distance between two arrays.

    Args:
        a (np.ndarray): First array.
        b (np.ndarray): Second array.
        p (int, optional): The degree of the Minkowski distance. Defaults to 2 (Euclidean distance).

    Returns:
        float: Minkowski distance between arrays a and b.
    """
    return np.sum(np.abs(a - b) ** p) ** (1 / p)


# k-Nearest Neighbors Model
class knn:
    def __init__(self):
        self.k = None
        self.p = None
        self.x_train = None
        self.y_train = None
        self.classes_ = None  # Store unique classes found in y_train

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, k: int = 5, p: int = 2):
        """
        Fit the model using X as training data and y as target values.

        You should check that all the arguments shall have valid values:
            X and y have the same number of rows.
            k and p are positive integers.

        Args:
            X_train (np.ndarray): Training data.
            y_train (np.ndarray): Target values.
            k (int, optional): Number of neighbors to use. Defaults to 5.
            p (int, optional): The degree of the Minkowski distance. Defaults to 2.
        """
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("Length of X_train and y_train must be equal.")
        if not (isinstance(k, int) and k > 0 and isinstance(p, int) and p > 0):
            raise ValueError("k and p must be positive integers.")

        self.k = k
        self.p = p
        self.x_train = X_train
        self.y_train = y_train
        self.classes_ = np.unique(y_train)

    def compute_distances(self, point: np.ndarray) -> np.ndarray:
        """Compute distance from a point to every point in the training dataset

        Args:
            point (np.ndarray): data sample.

        Returns:
            np.ndarray: distance from point to each point in the training dataset.
        """
        # Vectorized computation of Minkowski distance:
        distances = np.sum(np.abs(self.x_train - point) ** self.p, axis=1) ** (1 / self.p)
        return distances

    def get_k_nearest_neighbors(self, distances: np.ndarray) -> np.ndarray:
        """Get the k nearest neighbors indices given the distances vector from a point.

        Args:
            distances (np.ndarray): distances vector from a point whose neighbors want to be identified.

        Returns:
            np.ndarray: row indices from the k nearest neighbors.
        """
        return np.argsort(distances)[: self.k]

    def most_common_label(self, knn_labels: np.ndarray) -> int:
        """Obtain the most common label from the labels of the k nearest neighbors

        Args:
            knn_labels (np.ndarray): labels from the k nearest neighbors

        Returns:
            int: most common label
        """
        counter = Counter(knn_labels)
        # most_common returns a list of (label, count) tuples; pick the label with the highest count.
        most_common = counter.most_common(1)[0][0]
        return most_common

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the class labels for the provided data.

        Args:
            X (np.ndarray): data samples to predict their labels.

        Returns:
            np.ndarray: Predicted class labels.
        """
        predictions = []
        for point in X:
            distances = self.compute_distances(point)
            neighbor_indices = self.get_k_nearest_neighbors(distances)
            neighbor_labels = self.y_train[neighbor_indices]
            label = self.most_common_label(neighbor_labels)
            predictions.append(label)
        return np.array(predictions)

    def predict_proba(self, X):
        """
        Predict the class probabilities for the provided data.

        Each class probability is the amount of each label from the k nearest neighbors
        divided by k.

        Args:
            X (np.ndarray): data samples to predict their labels.

        Returns:
            np.ndarray: Predicted class probabilities. The columns are in the order of self.classes_.
        """
        proba_list = []
        for point in X:
            distances = self.compute_distances(point)
            neighbor_indices = self.get_k_nearest_neighbors(distances)
            neighbor_labels = self.y_train[neighbor_indices]
            # Count occurrences for each class (as stored in self.classes_)
            counts = {label: 0 for label in self.classes_}
            for label in neighbor_labels:
                counts[label] += 1
            # Compute probabilities in the same order as self.classes_
            proba = [counts[label] / self.k for label in self.classes_]
            proba_list.append(proba)
        return np.array(proba_list)

    def __str__(self):
        """
        String representation of the kNN model.
        """
        return f"kNN model (k={self.k}, p={self.p})"


def plot_2Dmodel_predictions(X, y, model, grid_points_n):
    """
    Plot the classification results and predicted probabilities of a model on a 2D grid.
    """
    unique_labels = np.unique(y)
    num_to_label = {i: label for i, label in enumerate(unique_labels)}

    preds = model.predict(X)

    tp = (y == unique_labels[1]) & (preds == unique_labels[1])
    fp = (y == unique_labels[0]) & (preds == unique_labels[1])
    fn = (y == unique_labels[1]) & (preds == unique_labels[0])
    tn = (y == unique_labels[0]) & (preds == unique_labels[0])

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].scatter(X[tp, 0], X[tp, 1], color="green", label=f"True {num_to_label[1]}")
    ax[0].scatter(X[fp, 0], X[fp, 1], color="red", label=f"False {num_to_label[1]}")
    ax[0].scatter(X[fn, 0], X[fn, 1], color="blue", label=f"False {num_to_label[0]}")
    ax[0].scatter(X[tn, 0], X[tn, 1], color="orange", label=f"True {num_to_label[0]}")
    ax[0].set_title("Classification Results")
    ax[0].legend()

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_points_n),
        np.linspace(y_min, y_max, grid_points_n),
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = model.predict_proba(grid)[:, 1].reshape(xx.shape)

    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette="Set1", ax=ax[1])
    ax[1].set_title("Classes and Estimated Probability Contour Lines")

    cnt = ax[1].contour(xx, yy, probs, levels=np.arange(0, 1.1, 0.1), colors="black")
    ax[1].clabel(cnt, inline=True, fontsize=8)

    plt.tight_layout()
    plt.show()


def evaluate_classification_metrics(y_true, y_pred, positive_label):
    """
    Calculate various evaluation metrics for a classification model.
    """
    y_true_mapped = np.array([1 if label == positive_label else 0 for label in y_true])
    y_pred_mapped = np.array([1 if label == positive_label else 0 for label in y_pred])

    tp = np.sum((y_true_mapped == 1) & (y_pred_mapped == 1))
    tn = np.sum((y_true_mapped == 0) & (y_pred_mapped == 0))
    fp = np.sum((y_true_mapped == 0) & (y_pred_mapped == 1))
    fn = np.sum((y_true_mapped == 1) & (y_pred_mapped == 0))

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "Confusion Matrix": [tn, fp, fn, tp],
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "F1 Score": f1,
    }


def plot_calibration_curve(y_true, y_probs, positive_label, n_bins=10):
    """
    Plot a calibration curve to evaluate the accuracy of predicted probabilities.
    """
    y_true_mapped = np.array([1 if label == positive_label else 0 for label in y_true])
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    true_proportions = []

    for i in range(n_bins):
        bin_lower = bins[i]
        bin_upper = bins[i + 1]
        bin_center = (bin_lower + bin_upper) / 2
        indices = np.where((y_probs >= bin_lower) & (y_probs < bin_upper))[0]
        if len(indices) > 0:
            true_frac = np.mean(y_true_mapped[indices])
        else:
            true_frac = np.nan
        bin_centers.append(bin_center)
        true_proportions.append(true_frac)

    plt.figure(figsize=(8, 6))
    plt.plot(bin_centers, true_proportions, marker='o', linestyle='-', label='Calibration Curve')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

    return {"bin_centers": np.array(bin_centers), "true_proportions": np.array(true_proportions)}


def plot_probability_histograms(y_true, y_probs, positive_label, n_bins=10):
    """
    Plot probability histograms for the positive and negative classes separately.
    """
    y_true_mapped = np.array([1 if label == positive_label else 0 for label in y_true])
    pos_probs = y_probs[y_true_mapped == 1]
    neg_probs = y_probs[y_true_mapped == 0]

    plt.figure(figsize=(8, 6))
    plt.hist(pos_probs, bins=n_bins, range=(0, 1), alpha=0.7, label='Positive Class', color='green')
    plt.hist(neg_probs, bins=n_bins, range=(0, 1), alpha=0.7, label='Negative Class', color='red')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.title('Probability Histograms by Class')
    plt.legend()
    plt.grid(True)
    plt.show()

    return {
        "array_passed_to_histogram_of_positive_class": pos_probs,
        "array_passed_to_histogram_of_negative_class": neg_probs,
    }


def plot_roc_curve(y_true, y_probs, positive_label):
    """
    Plot the Receiver Operating Characteristic (ROC) curve.
    """
    y_true_mapped = np.array([1 if label == positive_label else 0 for label in y_true])
    thresholds = np.linspace(0, 1, 11)  # Se generan 11 umbrales
    tpr_list = []
    fpr_list = []

    for thresh in thresholds:
        y_pred = (y_probs >= thresh).astype(int)
        tp = np.sum((y_true_mapped == 1) & (y_pred == 1))
        tn = np.sum((y_true_mapped == 0) & (y_pred == 0))
        fp = np.sum((y_true_mapped == 0) & (y_pred == 1))
        fn = np.sum((y_true_mapped == 1) & (y_pred == 0))

        tpr_val = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0

        tpr_list.append(tpr_val)
        fpr_list.append(fpr_val)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_list, tpr_list, marker='.', label='ROC Curve')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

    return {"fpr": np.array(fpr_list), "tpr": np.array(tpr_list)}
