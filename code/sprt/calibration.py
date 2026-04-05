"""
Calibration Utilities for SPRT
Temperature Scaling and Isotonic Regression
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.isotonic import IsotonicRegression
from typing import Tuple, Optional


class TemperatureScaling(nn.Module):
    """
    Temperature scaling for probability calibration.
    Implements Guo et al. (2017) "On Calibration of Modern Neural Networks"
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Calibrate logits with temperature scaling.

        Args:
            logits: Raw model logits

        Returns:
            Calibrated logits
        """
        return logits / self.temperature

    def fit(self, val_logits: torch.Tensor, val_labels: torch.Tensor,
            lr: float = 0.01, max_iter: int = 50) -> float:
        """
        Learn optimal temperature on validation set.

        Args:
            val_logits: Validation set logits
            val_labels: Validation set labels
            lr: Learning rate
            max_iter: Maximum iterations

        Returns:
            Final loss value
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval_loss():
            optimizer.zero_grad()
            loss = criterion(self.forward(val_logits), val_labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        return eval_loss().item()

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """
        Calibrate probabilities.

        Args:
            probs: Raw probabilities

        Returns:
            Calibrated probabilities
        """
        logits = np.log(probs / (1 - probs + 1e-10))
        calibrated_logits = logits / self.temperature.item()
        return 1 / (1 + np.exp(-calibrated_logits))


class IsotonicCalibration:
    """
    Isotonic regression for probability calibration.
    Non-parametric calibration method.
    """

    def __init__(self):
        self.regressor = IsotonicRegression(out_of_bounds='clip')
        self.fitted = False

    def fit(self, probs: np.ndarray, labels: np.ndarray):
        """
        Fit isotonic regression on validation set.

        Args:
            probs: Validation set probabilities
            labels: Validation set labels
        """
        self.regressor.fit(probs, labels)
        self.fitted = True

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """
        Calibrate probabilities.

        Args:
            probs: Raw probabilities

        Returns:
            Calibrated probabilities
        """
        if not self.fitted:
            raise ValueError("IsotonicCalibration must be fitted before calibration")
        return self.regressor.transform(probs)


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE).

    Args:
        probs: Predicted probabilities
        labels: True labels
        n_bins: Number of bins for ECE computation

    Returns:
        ECE value
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (probs > bin_lower) & (probs <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = labels[in_bin].mean()
            avg_confidence_in_bin = probs[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece


def calibrate_classifier(model, val_loader, device: str = "cuda") -> Tuple[float, float]:
    """
    Calibrate classifier and return ECE before and after.

    Args:
        model: Classifier model
        val_loader: Validation data loader
        device: Device for computation

    Returns:
        ece_before: ECE before calibration
        ece_after: ECE after temperature scaling
    """
    model.eval()
    all_logits = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in val_loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)

            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[:, 1]

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    all_probs = torch.cat(all_probs).numpy()

    # ECE before calibration
    ece_before = compute_ece(all_probs, all_labels.numpy())

    # Temperature scaling
    temp_scaling = TemperatureScaling()
    temp_scaling.fit(all_logits, all_labels)

    # Calibrate and compute ECE after
    calibrated_probs = temp_scaling.calibrate(all_probs)
    ece_after = compute_ece(calibrated_probs, all_labels.numpy())

    return ece_before, ece_after, temp_scaling.temperature.item()
