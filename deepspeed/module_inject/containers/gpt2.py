# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .base import *
from .features.meta_tensor import MetaTensorContainer
from .features.hybrid_engine import HybridEngineContainer
from deepspeed.model_implementations.transformers.ds_gpt import DeepSpeedGPTInference
from ..policy import TransformerPolicy, maybe_get_lora, maybe_copy
from ..policy import transformer_param_names


class DS_GPT2Container(MetaTensorContainer, HybridEngineContainer, BaseTransformerContainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # All model specific things should be defined here instead of the base class.

    def create_module(self, config=None):
        _config = config if config is not None else self.ds_model_config
        self.module = DeepSpeedGPTInference(_config, mp_group=self.mp_group)
        self.module.config.scale_attention = self.scale_attention
        return self.module

    def set_lora_params(self):
        """
        Necessary to implement for `HybridEngineContainer`
        """
        self.lora_params = [
            maybe_get_lora(p) for p in [
                self.policy.client_module.mlp.c_fc, self.policy.client_module.mlp.c_proj,
                self.policy.client_module.attn.c_attn, self.policy.client_module.attn.c_proj
            ]
        ]

    def attention_qkv_mp(self, mp_replace, reversed_dim=False):
        self.module.attention.attn_qkvw = mp_replace.copy(self.module.attention.attn_qkvw, self.qkvw)
        self.module.attention.attn_qkvb = mp_replace.copy(self.module.attention.attn_qkvb, self.qkvb)

    def get_lora_matched_pair(self):
        fc1_lora, fc2_lora, qkv_lora, out_lora = self.get_lora_params()
        ret = [(fc1_lora, self._h4h_w), (fc2_lora, self._4hh_w), (qkv_lora, self.qkvw), (out_lora, self.dense_w)]
        return ret

    def load_params(self, module, sd, weight_quantizer, mp_replace, prefix):
        param_names = (
            'attn.c_attn.weight', \
            'attn.c_attn.bias', \
            'attn.c_proj.weight', \
            'attn.c_proj.bias', \
            'mlp.c_fc.weight', \
            'mlp.c_fc.bias', \
            'mlp.c_proj.weight', \
            'mlp.c_proj.bias', \
            'ln_2.weight', \
            'ln_2.bias', \
            'ln_1.weight', \
            'ln_1.bias'
        )
        for i in range(0, 2):
            maybe_copy(module.attention,
                       sd,
                       weight_quantizer,
                       mp_replace,
                       transformer_param_names[i],
                       prefix + param_names[i],
                       qkv=True,
                       megatron_v2=self.policy.is_megatron_v2,
                       split_qkv=self.policy.split_qkv)
        for i in range(2, 4):
            maybe_copy(module.attention, sd, weight_quantizer, mp_replace, transformer_param_names[i],
                       prefix + param_names[i])
        for i in range(4, 10):
            maybe_copy(module.mlp, sd, weight_quantizer, mp_replace, transformer_param_names[i],
                       prefix + param_names[i])
        for i in range(10, 12):
            maybe_copy(module, sd, weight_quantizer, mp_replace, transformer_param_names[i], prefix + param_names[i])


class HFGPT2LayerPolicy(TransformerPolicy):
    _orig_layer_class = None

    def __init__(self, client_module, inference=True):
        # HuggingFace GPT2 uses convolutional layer instead of linear layer
        super().__init__(inference, linear_layer=False)
        self.client_module = client_module
        try:
            import transformers
            HFGPT2LayerPolicy._orig_layer_class = transformers.models.gpt2.modeling_gpt2.GPT2Block
        except:
            HFGPT2LayerPolicy._orig_layer_class = None

    def get_hidden_heads(self):
        return self.client_module.attn.embed_dim, \
                self.client_module.attn.num_heads, \
                self.client_module.ln_1.eps, \
                DEFAULT_INTERMEDIATE_SIZE

    def attention(self, enable_training=False):
        return  self.client_module.attn.c_attn.weight, \
                self.client_module.attn.c_attn.bias, \
                self.client_module.attn.c_proj.weight, \
                self.client_module.attn.c_proj.bias

    def mlp(self, enable_training=False):
        return self.client_module.mlp.c_fc.weight, \
               self.client_module.mlp.c_fc.bias, \
               self.client_module.mlp.c_proj.weight, \
               self.client_module.mlp.c_proj.bias

    def layernorm(self):
        return self.client_module.ln_2.weight, \
               self.client_module.ln_2.bias, \
               self.client_module.ln_1.weight, \
               self.client_module.ln_1.bias
