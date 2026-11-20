"""Opt-in encoder trainability for same-recipe paired image studies.

Keep the original module tree, even for the frozen arm: Split-PU identifies
intermediate encoder features by their registered module paths. A template
policy tag survives estimator cloning without changing constructor signatures
or the tensor-only checkpoint format. Apply it to each fit-owned copy, not to
the user's template parameters or buffers.
"""

from __future__ import annotations


def set_encoder_training_policy(encoder, *, trainable: bool):
    """Tag the template for an explicit paired-study arm and return it."""
    from torch.nn import Module

    if not isinstance(encoder, Module):
        raise TypeError("paired encoder must be a torch.nn.Module")
    if type(trainable) is not bool:
        raise TypeError("encoder trainable policy must be a boolean")
    encoder._pu_toolbox_encoder_trainable = trainable
    return encoder


def is_frozen_encoder(encoder) -> bool:
    """An untagged (legacy) encoder is not frozen by this policy."""
    return getattr(encoder, "_pu_toolbox_encoder_trainable", True) is False


def is_policy_encoder(encoder) -> bool:
    """Identify both paired-arm encoder roots, not their untagged children."""
    return type(getattr(encoder, "_pu_toolbox_encoder_trainable", None)) is bool


def prepare_encoder_copy(encoder):
    """Apply gradient policy to an owned copy and start eval-only probing."""
    return encoder.requires_grad_(not is_frozen_encoder(encoder)).eval()


def keep_frozen_encoder_eval(encoder) -> None:
    """Undo a parent train-mode switch for frozen encoder buffers and dropout."""
    if is_frozen_encoder(encoder):
        encoder.eval()
