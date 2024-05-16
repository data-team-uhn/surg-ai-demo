# Significant portions of this file were copied from Hugging
# Face Transformers library, version 4.37.1, as installed
# from conda-forge, from file
# transformers/models/segformer/modeling_segformer.py
#
# The most significant change is combination of the original
# SegformerForSemanticSegmentation and SegformerForImageClassification
# classes to a new class
# SegformerForMultiTaskSemanticSegmentationAndImageClassification
# that is capable of solving multiple tasks simultaneously
#
# Original copyright notice and license from this file are
# included below:

# START ORIGINAL COPYRIGHT NOTICE & LICENSE

# Copyright 2021 NVIDIA The HuggingFace Inc. team. All rights reserved.
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

# END ORIGINAL COPYRIGHT NOTICE & LICENSE

from torch import nn
from transformers.models.segformer.modeling_segformer import (
    SegformerConfig,
    SegformerDecodeHead,
    SegformerPreTrainedModel,
    SegformerModel,
)


SEGFORMER_MODEL_VARIANT_CONFIG_OVERRIDES = {
    "MiT-b0": {
        "depths": [2, 2, 2, 2],
        "hidden_sizes": [32, 64, 160, 256],
        "decoder_hidden_size": 256,
    },
    "MiT-b1": {
        "depths": [2, 2, 2, 2],
        "hidden_sizes": [64, 128, 320, 512],
        "decoder_hidden_size": 256,
    },
    "MiT-b2": {
        "depths": [3, 4, 6, 3],
        "hidden_sizes": [64, 128, 320, 512],
        "decoder_hidden_size": 768,
    },
}


class SegformerForMultiTaskSemanticSegmentationAndImageClassification(
    SegformerPreTrainedModel
):
    def __init__(self, config, task_types, task_config_overrides):
        if len(task_types) != len(task_config_overrides):
            raise ValueError(f"{len(task_types)=} != {len(task_config_overrides)=}")
        super().__init__(config)
        self.segformer = SegformerModel(config)
        self.task_types = task_types
        self.num_labelss = []
        decode_heads = []
        classifiers = []
        for task_type, task_config_override in zip(
            self.task_types, task_config_overrides
        ):
            task_config = SegformerConfig(**config.to_dict())
            for k, v in task_config_override.items():
                setattr(task_config, k, v)
            self.num_labelss.append(task_config.num_labels)
            if task_type == "segmentation":
                decode_heads.append(
                    nn.Sequential(
                        SegformerDecodeHead(task_config),
                        nn.UpsamplingBilinear2d(scale_factor=4),
                    )
                )
            elif task_type == "classification":
                classifiers.append(
                    nn.Linear(task_config.hidden_sizes[-1], task_config.num_labels)
                )
            else:
                raise ValueError(f"Unknown {task_type=}")
        self.decode_heads = nn.ModuleList(decode_heads)
        self.classifiers = nn.ModuleList(classifiers)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values,
    ):
        outputs = self.segformer(
            pixel_values,
            output_attentions=False,
            output_hidden_states=True,  # we need the intermediate hidden states
            return_dict=False,
        )

        logitss = []
        if self.decode_heads:
            encoder_hidden_states = outputs[1]
            for decode_head in self.decode_heads:
                logitss.append(decode_head(encoder_hidden_states))
        if self.classifiers:
            sequence_output = outputs[0]
            # convert last hidden states to (batch_size, height*width, hidden_size)
            batch_size = sequence_output.shape[0]
            if self.config.reshape_last_stage:
                # (batch_size, num_channels, height, width) -> (batch_size, height, width, num_channels)
                sequence_output = sequence_output.permute(0, 2, 3, 1)
            sequence_output = sequence_output.reshape(
                batch_size, -1, self.config.hidden_sizes[-1]
            )
            # global average pooling
            sequence_output = sequence_output.mean(dim=1)
            for classifier in self.classifiers:
                logitss.append(classifier(sequence_output))
        return logitss
