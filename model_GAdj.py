import torch
import pickle
import torch.nn as nn
import numpy as np
import torch_geometric
from transformers import GPT2Model, GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
from typing import Optional, Tuple, Union


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def load_graph_data(pkl_filename):
    adj_mx = load_pickle(pkl_filename)
    return adj_mx
    
graph_pkl_filename = 'data/adj_mx.pkl'
adjacency_matrix = load_graph_data(graph_pkl_filename)
adjacency_matrix = torch.tensor(adjacency_matrix)

class TemporalEmbedding(nn.Module):
    def __init__(self, time, features):
        super(TemporalEmbedding, self).__init__()

        self.time = time
        # temporal embeddings
        self.time_day = nn.Parameter(torch.empty(time, features))
        nn.init.xavier_uniform_(self.time_day)

        self.time_week = nn.Parameter(torch.empty(7, features))
        nn.init.xavier_uniform_(self.time_week)

    def forward(self, x):
        day_emb = x[..., 1]
        time_day = self.time_day[
            (day_emb[:, -1, :] * self.time).type(torch.LongTensor)
        ]
        time_day = time_day.transpose(1, 2).unsqueeze(-1)

        week_emb = x[..., 2]
        time_week = self.time_week[
            (week_emb[:, -1, :]).type(torch.LongTensor)
        ]
        time_week = time_week.transpose(1, 2).unsqueeze(-1)

        # temporal embeddings
        tem_emb = time_day + time_week
        return tem_emb

from dataclasses import dataclass

@dataclass
class BaseModelOutputWithPastAndCrossAttentions:
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

class PFA(nn.Module):
    def __init__(self, device="cuda:0", gpt_layers=6, U=1):
        super(PFA, self).__init__()
        self.gpt2 = GPT2Model.from_pretrained("gpt2", attn_implementation="sdpa",
                                              output_attentions=True, output_hidden_states=True)  #attn_implementation="sdpa" OR "eager"
        
        # Adjust GPT-2 layers and freezing/unfreezing parameters
        self.gpt2.h = self.gpt2.h[:gpt_layers]
        self.U = U
        self.device = device

        for layer_index, layer in enumerate(self.gpt2.h):
            for name, param in layer.named_parameters():
                if layer_index < gpt_layers - self.U:
                    if "ln" in name or "wpe" in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                else:
                    if "mlp" in name:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True

    # Define a custom forward function where the attention_mask.view() step is skipped
    def custom_forward(self,
                       input_ids: Optional[torch.LongTensor] = None,
                       past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
                       attention_mask: Optional[torch.FloatTensor] = None,
                       token_type_ids: Optional[torch.LongTensor] = None,
                       position_ids: Optional[torch.LongTensor] = None,
                       head_mask: Optional[torch.FloatTensor] = None,
                       inputs_embeds: Optional[torch.FloatTensor] = None,
                       encoder_hidden_states: Optional[torch.Tensor] = None,
                       encoder_attention_mask: Optional[torch.FloatTensor] = None,
                       use_cache: Optional[bool] = None,
                       output_attentions: Optional[bool] = None,
                       output_hidden_states: Optional[bool] = None,
                       return_dict: Optional[bool] = None) -> Union[Tuple, dict]:
    
        output_attentions = output_attentions if output_attentions is not None else self.gpt2.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.gpt2.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.gpt2.config.use_cache
        return_dict = return_dict if return_dict is not None else self.gpt2.config.use_return_dict
    
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
    
        device = input_ids.device if input_ids is not None else inputs_embeds.device
    
        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.gpt2.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)
    
        if inputs_embeds is None:
            inputs_embeds = self.gpt2.wte(input_ids)
        position_embeds = self.gpt2.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
    
        # Attention mask.
        _use_sdpa = self.gpt2.config._attn_implementation == "sdpa" and output_attentions is False and head_mask is None
        attention_mask = attention_mask if attention_mask is not None else None #attention_mask.view(batch_size, -1) skipped this
        if self.gpt2.config._attn_implementation == "flash_attention_2":
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif _use_sdpa:
            attention_mask = attention_mask
            # attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
            #     attention_mask=attention_mask,
            #     input_shape=(batch_size, input_shape[-1]),
            #     inputs_embeds=inputs_embeds,
            #     past_key_values_length=past_length,
            # )
        else:
            if attention_mask is not None:
                attention_mask = attention_mask
                # # We create a 3D attention mask from a 2D tensor mask.
                # # Sizes are [batch_size, 1, 1, to_seq_length]
                # # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
                # # this attention mask is more simple than the triangular masking of causal attention
                # # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
                # attention_mask = attention_mask[:, None, None, :]

                # # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
                # # masked positions, this operation will create a tensor which is 0.0 for
                # # positions we want to attend and the dtype's smallest value for masked positions.
                # # Since we are adding it to the raw scores before the softmax, this is
                # # effectively the same as removing these entirely.
                # attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
                # attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.gpt2.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            if _use_sdpa:
                encoder_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    mask=encoder_attention_mask, dtype=inputs_embeds.dtype, tgt_len=input_shape[-1]
                )
            elif not self._attn_implementation == "flash_attention_2":
                encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.gpt2.get_head_mask(head_mask, self.gpt2.config.n_layer)

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds
    
        # Remaining code follows the original logic...
        # You can also skip other operations if needed.
        hidden_states = self.gpt2.drop(hidden_states)
        
        # Process through the rest of the layers
        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)
    
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.gpt2.config.add_cross_attention else None

        all_hidden_states = () if output_hidden_states else None
        presents = () if use_cache else None
    
        for i, (block, layer_past) in enumerate(zip(self.gpt2.h, past_key_values)):
            if attention_mask is not None:
                attention_mask = attention_mask.to(hidden_states.device)
            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask[i] if head_mask is not None else None,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = outputs[0]
    
            if use_cache:
                presents = presents + (outputs[1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2],)
    
        hidden_states = self.gpt2.ln_f(hidden_states)
    
        hidden_states = hidden_states.view(output_shape)

        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


    def forward(self, x, adjacency_matrix):
        """
        Args:
            x: input embeddings [batch_size, sequence_length, hidden_dim]
            adjacency_matrix: adjacency matrix used as an attention mask
                              [batch_size, sequence_length, sequence_length]
        """
        batch_size =  x.shape[0]
        num_heads =  self.gpt2.config.n_head
        adjacency_matrix = adjacency_matrix.unsqueeze(0).repeat(batch_size, 1, 1)
        adjacency_matrix = adjacency_matrix.unsqueeze(1).repeat(1, num_heads, 1, 1)

        attention_mask = adjacency_matrix.to(self.device).float() #[64,12,250,250]
        # print(attention_mask.shape)

        # Use GPT-2 with attention mask
        output = self.custom_forward(
            inputs_embeds=x,
            attention_mask=attention_mask
        ).last_hidden_state
        
        return output

class GAdj(nn.Module):
    def __init__(
        self,
        device,
        adj_mx,
        input_dim=3,
        channels=64,
        num_nodes=170,
        input_len=12,
        output_len=12,
        dropout=0.1,
    ):
        super().__init__()

        # attributes
        self.device = device
        self.adj_mx = adj_mx
        self.num_nodes = num_nodes
        self.node_dim = channels
        self.input_len = input_len
        self.input_dim = input_dim
        self.output_len = output_len
        self.llm_layer = 6
        self.U = 1
        
        if num_nodes == 170 or num_nodes == 207:
            time = 288
        elif num_nodes == 250 or num_nodes == 266:
            time = 48


        gpt_channel = 256
        to_gpt_channel = 768
            
        self.start_conv = nn.Conv2d(
            self.input_dim * self.input_len, gpt_channel, kernel_size=(1, 1)
        )

        self.Temb = TemporalEmbedding(time, gpt_channel)
        self.node_emb = nn.Parameter(torch.empty(self.num_nodes, gpt_channel))
        nn.init.xavier_uniform_(self.node_emb)

        self.in_layer = nn.Conv2d(gpt_channel*3, to_gpt_channel, kernel_size=(1, 1))        

        # regression
        self.regression_layer = nn.Conv2d(to_gpt_channel, self.output_len, kernel_size=(1, 1)) 

        #GPT2
        self.gpt = PFA(device=self.device, gpt_layers=self.llm_layer, U=self.U)
                 
    # return the total parameters of model
    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])

    def forward(self, history_data):

        data = history_data.permute(0, 3, 2, 1)
        B, T, S, F = data.shape
        # print(data.shape) #[64, 12, 250, 3]
        
        #Temporal Embedding
        tem_emb = self.Temb(data)
        # print(tem_emb.shape) #[64, 256, 250, 1]

        node_emb = []
        node_emb.append(
            self.node_emb.unsqueeze(0)
            .expand(B, -1, -1)
            .transpose(1, 2)
            .unsqueeze(-1)
        )
        
        input_data = data.permute(0,3,2,1) #[32, 2, 207, 12]
        input_data = input_data.transpose(1, 2).contiguous() #[32, 207, 2, 12]
        input_data = (input_data.view(B, S, -1).transpose(1, 2).unsqueeze(-1))
        # print(input_data.shape) #[64, 36, 250, 1]

        input_data = self.start_conv(input_data)
        # print(input_data.shape)#[64, 36, 250, 1]

        data_st = torch.cat([input_data] + [tem_emb] + node_emb, dim=1)
        data_st = self.in_layer(data_st)

        data_st = data_st.permute(0, 2, 1, 3).squeeze(-1) 
        # print(data_st.shape) #[64, 250, 768]
        
        outputs = self.gpt(data_st, adjacency_matrix)
            
        outputs = outputs.permute(0, 2, 1).unsqueeze(-1)
#         print(outputs.shape) #[64, 768, 250, 1]       

        # regression
        outputs = self.regression_layer(outputs)  
        # print(outputs.shape) #[64, 12, 250, 1]

        return outputs
