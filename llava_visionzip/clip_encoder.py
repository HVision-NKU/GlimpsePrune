#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
# ------------------------------------------------------------------------
# Modified from LLaVA (https://github.com/haotian-liu/LLaVA)
# Copyright 2024 Senqiao Yang
# ------------------------------------------------------------------------
import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from transformers.models.clip.modeling_clip import CLIPEncoderLayer, CLIPAttention, CLIPEncoder

from .utils import CLIPAttention_forward, CLIP_EncoderLayer_forward



class CLIPVisionTower_VisionZip(nn.Module):


    @torch.no_grad()
    def forward(self, images):
        
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True, output_attentions=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True, output_attentions=True)
            attn_weights  = image_forward_outs.attentions[-2]
            hidden_states = image_forward_outs.hidden_states[-2]
            metric = self.vision_tower.vision_model.encoder.layers[-2].metric
            dominant_num =  self.vision_tower._info["dominant"]
            contextual_num = self.vision_tower._info["contextual"]

            ## Dominant Visual Tokens
            cls_idx = 0
            cls_attention = attn_weights[:, :, cls_idx, cls_idx+1:]  
            cls_attention_sum = cls_attention.sum(dim=1)  
            topk_indices = cls_attention_sum.topk(dominant_num, dim=1).indices + 1
            all_indices = torch.cat([torch.zeros((hidden_states.shape[0], 1), dtype=topk_indices.dtype, device=topk_indices.device), topk_indices], dim=1)
            
            mask = torch.ones_like(hidden_states[:, :, 0], dtype=torch.bool, device=metric.device).scatter_(1, all_indices, False)
            dominant_tokens = hidden_states.masked_select(~mask.unsqueeze(-1)).view(hidden_states.shape[0], dominant_num + 1, hidden_states.shape[2])
            
            ### Filter
            metric_filtered = metric[mask].view(hidden_states.shape[0], hidden_states.shape[1] - (dominant_num + 1), metric.shape[2])

            hidden_states_filtered = hidden_states.masked_select(mask.unsqueeze(-1)).view(hidden_states.shape[0], hidden_states.shape[1] - (dominant_num +1), hidden_states.shape[2])  
            
            metric_normalized = metric_filtered / metric_filtered.norm(dim=-1, keepdim=True) 

            ## Contextual Visual Tokens
            step = max(1, metric_normalized.shape[1] // contextual_num)
            target_indices = torch.arange(0, metric_normalized.shape[1], step, device=metric_normalized.device)[:contextual_num]
            target_tokens = metric_normalized[:, target_indices, :]

            tokens_to_merge = metric_normalized[:, ~torch.isin(torch.arange(metric_normalized.shape[1], device=metric_normalized.device), target_indices), :]
            similarity = torch.bmm(tokens_to_merge, target_tokens.transpose(1, 2))
            assign_one_hot = torch.zeros(tokens_to_merge.shape[0], tokens_to_merge.shape[1], contextual_num, dtype=hidden_states_filtered.dtype, device=metric_normalized.device)
            assign_one_hot.scatter_(2, similarity.argmax(dim=2).unsqueeze(-1), 1)
            counts = assign_one_hot.sum(dim=1).clamp(min=1).unsqueeze(-1)
            hidden_to_merge = hidden_states_filtered[:, ~torch.isin(torch.arange(hidden_states_filtered.shape[1], device=hidden_states_filtered.device), target_indices), :]
            aggregated_hidden = torch.bmm(assign_one_hot.transpose(1, 2), hidden_to_merge) / counts
            target_hidden = hidden_states_filtered[:, target_indices, :]  
            
            contextual_tokens = target_hidden + aggregated_hidden

            # Merge with target hidden states and concatenate
            hidden_states_save = torch.cat([dominant_tokens, contextual_tokens], dim=1).to(images.dtype)

        return hidden_states_save, all_indices

        # return hidden_states_save, hidden_states, all_indices





