# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Implementation of VolSDF.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Type

import torch
from torch import nn

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.ray_samplers import ErrorBoundedSampler
from nerfstudio.models.base_surface_model import SurfaceModel, SurfaceModelConfig


class LaplaceDensity(nn.Module):  # alpha * Laplace(loc=0, scale=beta).cdf(-sdf)
    """Laplace density from VolSDF"""

    def __init__(self, init_val, beta_min=0.0001):
        super().__init__()
        self.register_parameter("beta_min", nn.Parameter(beta_min * torch.ones(1), requires_grad=False))
        self.register_parameter("beta", nn.Parameter(init_val * torch.ones(1), requires_grad=True))

        if torch.cuda.is_available():
            self.beta_min = nn.Parameter(self.beta_min.cuda())
            self.beta = nn.Parameter(self.beta.cuda())

    def forward(
        self, sdf: TensorType["bs":...], beta: Union[TensorType["bs":...], None] = None
    ) -> TensorType["bs":...]:
        """convert sdf value to density value with beta, if beta is missing, then use learable beta"""

        if beta is None:
            beta = self.get_beta()

        alpha = 1.0 / beta
        return alpha * (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() / beta))

    def get_beta(self):
        """return current beta value"""
        beta = self.beta.abs() + self.beta_min
        return beta


@dataclass
class VolSDFModelConfig(SurfaceModelConfig):
    """VolSDF Model Config"""

    _target: Type = field(default_factory=lambda: VolSDFModel)
    num_samples: int = 64
    """Number of samples after error bounded sampling"""
    num_samples_eval: int = 128
    """Number of samples per iteration used in error bounded sampling"""
    num_samples_extra: int = 32
    """Number of uniformly sampled points for training"""
    beta_init: float = 0.01
    """Initial valur for the beta coefficient"""


class VolSDFModel(SurfaceModel):
    """VolSDF model

    Args:
        config: VolSDF configuration to instantiate model
    """

    config: VolSDFModelConfig

    def __init__(self, config, scene_box, num_train_data, **kwargs):
        super().__init__(config, scene_box, num_train_data, **kwargs)
        self.density_fn = LaplaceDensity(self.config.beta_init)

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        self.sampler = ErrorBoundedSampler(
            num_samples=self.config.num_samples,
            num_samples_eval=self.config.num_samples_eval,
            num_samples_extra=self.config.num_samples_extra,
        )

    def sample_and_forward_field(self, ray_bundle: RayBundle) -> Dict:
        ray_samples, eik_points = self.sampler(
            ray_bundle, density_fn=self.density_fn, sdf_fn=self.field.get_sdf
        )
        deltas = ray_samples.deltas.squeeze(-1)
        field_outputs = self.field(ray_samples)

        sdf = field_outputs[FieldHeadNames.SDF]
        beta = self.density_fn.get_beta()
        density = self.density_fn(sdf.reshape(ray_samples.shape), beta=beta.unsqueeze(-1))
        delta_density = deltas * density
        alphas = 1 - torch.exp(-delta_density)

        weights, transmittance = ray_samples.get_weights_and_transmittance_from_alphas(alphas.unsqueeze(-1))
        bg_transmittance = transmittance[:, -1, :]

        samples_and_field_outputs = {
            "ray_samples": ray_samples,
            "eik_points": eik_points,
            "field_outputs": field_outputs,
            "weights": weights,
            "bg_transmittance": bg_transmittance,
        }
        return samples_and_field_outputs

    def get_metrics_dict(self, outputs, batch) -> Dict:
        metrics_dict = super().get_metrics_dict(outputs, batch)
        if self.training:
            # training statics
            #metrics_dict["beta"] = self.field.laplace_density.get_beta().item()
            #metrics_dict["beta"] = self.field.laplace_density.get_beta().item()
            metrics_dict["alpha"] = 1.0 / self.density_fn.get_beta().item()
            metrics_dict["alpha"] = 1.0 / self.density_fn.get_beta().item()

        return metrics_dict
