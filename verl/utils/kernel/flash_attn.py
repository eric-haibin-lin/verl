# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

from verl.utils.device import is_cuda_available, is_npu_available

if is_cuda_available:
    try:
        from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
    except ImportError:
        raise ImportError("flash_attn is required when CUDA is available but not installed")
elif is_npu_available:
    try:
        from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input
    except ImportError:
        raise ImportError("transformers with NPU flash attention support is required when NPU is available")
else:
    raise RuntimeError("Neither CUDA nor NPU is available. Flash attention functions require either CUDA or NPU.")
