"""
DETECTOR AI — Stage 2: Fine-Grained Species Classifier
Uses an EfficientNetV2-S backbone (via ``timm``) to classify animal crops
into fine-grained species labels.

The classifier is designed to be *optional* — when no trained weights are
available it disables itself gracefully and every call to
:meth:`SpeciesClassifier.classify` returns ``None``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from detector_ai.config import (
    ClassifierConfig,
    SPECIES_CONF_THRESHOLD,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# ImageNet normalisation stats (RGB order)
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Species labels — extend this list as more classes are added to the
# training data.  Index 0 must match class 0 in the training set.
_SPECIES_NAMES: List[str] = ["bengal_tiger"]


class SpeciesClassifier:
    """EfficientNetV2-S species classifier.

    Parameters:
        weights_path:
            Absolute or relative path to a ``.pth`` checkpoint.  If
            ``None`` or if the file does not exist the classifier is
            disabled (all calls to :meth:`classify` return ``None``).
        conf_threshold:
            Minimum softmax / sigmoid confidence to accept a
            classification result.
        device:
            PyTorch device string (``'cuda'``, ``'cpu'``, ``'cuda:0'``…).
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        conf_threshold: float = SPECIES_CONF_THRESHOLD,
        device: str = ClassifierConfig.device,
    ) -> None:
        self.conf_threshold = conf_threshold
        self._cfg = ClassifierConfig()
        self._model: Optional[torch.nn.Module] = None
        self._device = torch.device(
            device if torch.cuda.is_available() or device == "cpu" else "cpu"
        )

        # Attempt to load the model + weights --------------------------------
        if weights_path is None:
            logger.info("Species classifier disabled — no weights path given.")
            return

        weights_file = Path(weights_path)
        if not weights_file.is_file():
            logger.warning(
                "Species classifier disabled — weights not found: %s",
                weights_file,
            )
            return

        self._load_model(weights_file)

    # -----------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------

    @property
    def is_available(self) -> bool:
        """``True`` if the model has been loaded and is ready for inference."""
        return self._model is not None

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def classify(self, crop: np.ndarray) -> Optional[Tuple[str, float]]:
        """Classify an animal crop into a species.

        Parameters:
            crop:
                BGR image crop (NumPy ``uint8`` array, any size).

        Returns:
            ``(species_name, confidence)`` tuple if the classification
            confidence meets the threshold, otherwise ``None``.
        """
        if not self.is_available:
            return None

        tensor = self._preprocess(crop)

        with torch.no_grad():
            logits: torch.Tensor = self._model(tensor)  # (1, num_classes)

        # For single-class (binary) mode, use sigmoid; else softmax.
        if self._cfg.num_classes == 1:
            prob = torch.sigmoid(logits).item()
            # prob > 0.5 → positive class (index 0 = bengal_tiger)
            if prob >= self.conf_threshold:
                return (_SPECIES_NAMES[0], float(prob))
            return None

        probs = F.softmax(logits, dim=1).squeeze(0)  # (num_classes,)
        max_prob, max_idx = probs.max(dim=0)
        max_prob = float(max_prob)
        max_idx = int(max_idx)

        if max_prob < self.conf_threshold:
            return None

        species = (
            _SPECIES_NAMES[max_idx]
            if max_idx < len(_SPECIES_NAMES)
            else f"species_{max_idx}"
        )
        return (species, max_prob)

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _load_model(self, weights_file: Path) -> None:
        """Instantiate the EfficientNetV2-S backbone and load weights."""
        try:
            import timm  # late import to keep startup fast when disabled

            self._model = timm.create_model(
                self._cfg.model_name,
                pretrained=False,
                num_classes=self._cfg.num_classes,
            )

            state_dict = torch.load(
                str(weights_file),
                map_location=self._device,
                weights_only=True,
            )
            self._model.load_state_dict(state_dict)
            self._model.to(self._device)
            self._model.eval()

            logger.info(
                "Species classifier loaded from '%s' on %s "
                "(num_classes=%d).",
                weights_file,
                self._device,
                self._cfg.num_classes,
            )
        except Exception:
            logger.exception("Failed to load species classifier — disabling.")
            self._model = None

    def _preprocess(self, crop: np.ndarray) -> torch.Tensor:
        """Resize, normalise, and convert a BGR crop to a batched tensor.

        Steps:
            1. Convert BGR → RGB.
            2. Resize to ``(img_size, img_size)``.
            3. Scale to ``[0, 1]``.
            4. Normalise with ImageNet mean / std.
            5. Transpose to ``(C, H, W)`` and add batch dimension.

        Returns:
            ``torch.Tensor`` of shape ``(1, 3, img_size, img_size)``
            on ``self._device``.
        """
        img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        img = cv2.resize(
            img,
            (self._cfg.img_size, self._cfg.img_size),
            interpolation=cv2.INTER_LINEAR,
        )
        img = img.astype(np.float32) / 255.0
        img = (img - _IMAGENET_MEAN) / _IMAGENET_STD  # (H, W, 3)
        img = np.transpose(img, (2, 0, 1))  # (3, H, W)
        tensor = torch.from_numpy(img).unsqueeze(0)  # (1, 3, H, W)
        return tensor.to(self._device)
