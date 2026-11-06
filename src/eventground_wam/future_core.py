"""Decoder-free view of the unchanged pixel/JEPA future core for Week 8."""
from __future__ import annotations

from torch import nn

from eventground_wam.pixel_predictor import PixelCondition, PixelPredictor

TRAINING_OUTPUTS = frozenset({'image_decoder', 'residual_head', 'output_projection'})


class FutureTokenCore(PixelPredictor):
    """Share existing core tensors, without registering or executing output heads.

    In training the full predictor remains available for teacher-forced losses.
    Optimizers must include it once, not add this shared view a second time.
    The inherited forward runs verbatim; only its final decode hook is an identity.
    Thus there is no second implementation of the future-token computation.
    """

    def __init__(self, predictor: PixelPredictor):
        nn.Module.__init__(self)
        if not isinstance(predictor, PixelPredictor):
            raise TypeError('future core requires the accepted pixel or JEPA predictor')
        self.config = (predictor.config.pixel_config() if hasattr(predictor.config, 'pixel_config')
                       else predictor.config)
        for name, module in predictor.named_children():
            if name not in TRAINING_OUTPUTS:
                self.add_module(name, module)
        for name, parameter in predictor.named_parameters(recurse=False):
            self.register_parameter(name, parameter)
        for name, buffer in predictor.named_buffers(recurse=False):
            self.register_buffer(name, buffer)
        self.train(predictor.training)

    def decode(self, future_tokens, current_images):
        return future_tokens

    def forward(self, condition: PixelCondition):
        return PixelPredictor.forward(self, condition).future_tokens

    def parameter_budget(self):
        return {'future_core': sum(p.numel() for p in self.parameters()),
                'future_tokens': 128, 'future_token_dimension': self.config.dimension,
                'training_output_parameters': 0}
