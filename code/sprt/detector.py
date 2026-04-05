"""
Wald-Threshold Early Exit Detector

Uses Wald's SPRT threshold formula to derive classifier confidence thresholds
from target error rate parameters (alpha, beta). Provides principled threshold
selection for confidence-based early exit in streaming safety detection.

NOTE: Classical SPRT error rate guarantees do NOT hold when using neural
classifier outputs as evidence scores. This framework provides interpretable
threshold selection, not theoretical error bounds.
"""

import torch
import numpy as np
from typing import Tuple, Optional, Callable


class WaldThresholdDetector:
    """
    Wald-Threshold early exit detector for streaming LLM safety detection.

    Uses Wald's threshold formula A = ln((1-beta)/alpha) to derive confidence
    thresholds from target error rate parameters. Prior adjustment is available
    for imbalanced distributions but invalidates theoretical bounds.
    """

    def __init__(
        self,
        classifier,
        alpha: float = 0.05,
        beta: float = 0.10,
        t_min: int = 5,
        prior: float = 0.5,
        device: str = "cuda"
    ):
        """
        Initialize SPRT detector.

        Args:
            classifier: Safety classifier (e.g., fine-tuned RoBERTa)
            alpha: Type I error rate (false positive rate bound)
            beta: Type II error rate (false negative rate bound)
            t_min: Minimum tokens before allowing decision
            prior: Prior probability P(H_1) of toxicity
            device: Device for inference
        """
        self.classifier = classifier
        self.alpha = alpha
        self.beta = beta
        self.t_min = t_min
        self.prior = prior
        self.device = device

        # Compute Wald thresholds with prior adjustment
        self.A = np.log((1 - beta) / alpha) + np.log(prior / (1 - prior))
        self.B = np.log(beta / (1 - alpha)) + np.log(prior / (1 - prior))

        self.epsilon = 1e-10

    def compute_evidence(self, p_t: float) -> float:
        """
        Compute cumulative evidence S_t from classifier probability.

        For a well-calibrated classifier, the logit directly estimates
        the cumulative log-likelihood ratio:
            S_t = log(p_t / (1 - p_t))

        Args:
            p_t: Classifier probability of toxicity

        Returns:
            Cumulative evidence S_t
        """
        # Symmetric epsilon for numerical stability in both directions
        return np.log((p_t + self.epsilon) / (1 - p_t + self.epsilon))

    def predict_streaming(
        self,
        text: str,
        tokenizer,
        max_length: int = 512
    ) -> Tuple[str, int]:
        """
        Perform streaming safety detection.

        Args:
            text: Input text to analyze
            tokenizer: Tokenizer for the classifier
            max_length: Maximum sequence length

        Returns:
            decision: "SAFE", "UNSAFE", or "CONTINUE"
            stopping_time: Token position where decision was made
        """
        self.classifier.eval()

        # Tokenize
        tokens = tokenizer.encode(text, add_special_tokens=True)

        for t in range(self.t_min, min(len(tokens), max_length)):
            # Get prefix
            prefix_tokens = tokens[:t+1]
            prefix_text = tokenizer.decode(prefix_tokens)

            # Classify prefix
            inputs = tokenizer(
                prefix_text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length
            ).to(self.device)

            with torch.no_grad():
                outputs = self.classifier(**inputs)
                p_t = torch.softmax(outputs.logits, dim=-1)[0, 1].item()

            # Compute evidence
            S_t = self.compute_evidence(p_t)

            # Check thresholds
            if S_t >= self.A:
                return "UNSAFE", t + 1
            elif S_t <= self.B:
                return "SAFE", t + 1

        # Reached max length without decision
        return "SAFE", min(len(tokens), max_length)

    def batch_predict(
        self,
        texts: list,
        tokenizer,
        max_length: int = 512
    ) -> Tuple[list, list]:
        """
        Batch prediction for multiple texts.

        Args:
            texts: List of texts to analyze
            tokenizer: Tokenizer for the classifier
            max_length: Maximum sequence length

        Returns:
            decisions: List of decisions
            stopping_times: List of stopping times
        """
        decisions = []
        stopping_times = []

        for text in texts:
            decision, st = self.predict_streaming(text, tokenizer, max_length)
            decisions.append(decision)
            stopping_times.append(st)

        return decisions, stopping_times


class FixedWindowDetector:
    """
    Fixed-window baseline for comparison.
    Checks at fixed intervals rather than continuously.
    """

    def __init__(
        self,
        classifier,
        window_size: int = 10,
        threshold: float = 0.5,
        device: str = "cuda"
    ):
        self.classifier = classifier
        self.window_size = window_size
        self.threshold = threshold
        self.device = device

    def predict_streaming(
        self,
        text: str,
        tokenizer,
        max_length: int = 512
    ) -> Tuple[str, int]:
        tokens = tokenizer.encode(text, add_special_tokens=True)

        for t in range(self.window_size, min(len(tokens), max_length), self.window_size):
            prefix_tokens = tokens[:t+1]
            prefix_text = tokenizer.decode(prefix_tokens)

            inputs = tokenizer(
                prefix_text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length
            ).to(self.device)

            with torch.no_grad():
                outputs = self.classifier(**inputs)
                p_t = torch.softmax(outputs.logits, dim=-1)[0, 1].item()

            if p_t >= self.threshold:
                return "UNSAFE", t + 1

        return "SAFE", min(len(tokens), max_length)


class ConfidenceDetector:
    """
    Confidence-based early stopping baseline.
    Stops when classifier confidence exceeds threshold.
    """

    def __init__(
        self,
        classifier,
        confidence_threshold: float = 0.95,
        t_min: int = 5,
        device: str = "cuda"
    ):
        self.classifier = classifier
        self.confidence_threshold = confidence_threshold
        self.t_min = t_min
        self.device = device

    def predict_streaming(
        self,
        text: str,
        tokenizer,
        max_length: int = 512
    ) -> Tuple[str, int]:
        tokens = tokenizer.encode(text, add_special_tokens=True)

        for t in range(self.t_min, min(len(tokens), max_length)):
            prefix_tokens = tokens[:t+1]
            prefix_text = tokenizer.decode(prefix_tokens)

            inputs = tokenizer(
                prefix_text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length
            ).to(self.device)

            with torch.no_grad():
                outputs = self.classifier(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)[0]
                p_t = probs[1].item()
                confidence = probs.max().item()

            if confidence >= self.confidence_threshold:
                decision = "UNSAFE" if p_t >= 0.5 else "SAFE"
                return decision, t + 1

        return "SAFE", min(len(tokens), max_length)
