import torch
import torch.nn as nn
import torch_geometric

from transformers import GPT2Model, GPT2Tokenizer #, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from torch_geometric.nn import GCNConv, GATConv

class GNNRetriever(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, nheads, gnn_type='gcn'):
        super(GNNRetriever, self).__init__()
        if gnn_type == 'gcn':
            self.gnn1 = GCNConv(input_dim, hidden_dim)
            self.gnn2 = GCNConv(hidden_dim, output_dim)
        elif gnn_type == 'gat':
            self.gnn1 = GATConv(input_dim, hidden_dim, heads = nheads)
            self.gnn2 = GATConv(hidden_dim*nheads, output_dim)
        else:
            raise ValueError("Unsupported GNN type: choose 'gcn' or 'gat'")
        
    def forward(self, x, edge_index):
        x = self.gnn1(x, edge_index)
        x = torch.relu(x)
        x = self.gnn2(x, edge_index)
        return x
    

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


# ########## ORGINAL #############
# class PFA(nn.Module):
#     def __init__(self, device="cuda:0", gpt_layers=6, U=1):
#         super(PFA, self).__init__()
#         self.gpt2 = GPT2Model.from_pretrained(
#             "gpt2", output_attentions=True, output_hidden_states=True
#         )
#         self.gpt2.h = self.gpt2.h[:gpt_layers]
#         self.U = U

#         for layer_index, layer in enumerate(self.gpt2.h):
#             for name, param in layer.named_parameters():
#                 if layer_index < gpt_layers - self.U:
#                     if "ln" in name or "wpe" in name:
#                         param.requires_grad = True
#                     else:
#                         param.requires_grad = False
#                 else:
#                     if "mlp" in name:
#                         param.requires_grad = False
#                     else:
#                         param.requires_grad = True

#     def forward(self, x):
#         return self.gpt2(inputs_embeds=x).last_hidden_state

########### LORA NEW (STDLLM) ##############
class PFA(nn.Module):
    def __init__(self, device="cuda:0", gpt_layers=6, U=1):
        super(PFA, self).__init__()
        self.gpt2 = GPT2Model.from_pretrained(
            "gpt2", output_attentions=True, output_hidden_states=True
        )
        self.gpt2.h = self.gpt2.h[:gpt_layers]
        self.U = U
        self.lora_rank = 16 #4,8

        # Configure LoRA
        self.lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=16,  # or any other hyperparameter specific to LoRA
            lora_dropout=0.,  # if you want to add dropout to LoRA #0.05
            target_modules=['q_attn','c_attn'],   # Apply LoRA to the attention layers only
            bias="none"  # specify whether to train bias parameters
        )
        self.gpt2 = get_peft_model(self.gpt2, self.lora_config)

        # Adjust parameter training requirements
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

    def forward(self, x):
        return self.gpt2(inputs_embeds=x).last_hidden_state

    def trainable_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                yield param

# ########### LORA ##############
# class PFA(nn.Module):
#     def __init__(self, device="cuda:0", gpt_layers=6, U=1):
#         super(PFA, self).__init__()
#         self.gpt2 = GPT2Model.from_pretrained(
#             "gpt2", output_attentions=True, output_hidden_states=True
#         )
#         self.gpt2.h = self.gpt2.h[:gpt_layers]
#         self.U = U
#         self.lora_rank = 4 #8

#         # Configure LoRA
#         self.lora_config = LoraConfig(
#             r=self.lora_rank,
#             lora_alpha=16,  # or any other hyperparameter specific to LoRA
#             lora_dropout=0.1,  # if you want to add dropout to LoRA #0.05
#             target_modules=["attn.c_attn"],   # Apply LoRA to the attention layers only
#             bias="none"  # specify whether to train bias parameters
#         )
#         self.gpt2 = get_peft_model(self.gpt2, self.lora_config)

#         # Adjust parameter training requirements
#         for layer_index, layer in enumerate(self.gpt2.h):
#             for name, param in layer.named_parameters():
#                 if layer_index < gpt_layers - self.U:
#                     if "ln" in name or "wpe" in name:
#                         param.requires_grad = True
#                     else:
#                         param.requires_grad = False
#                 else:
#                     if "mlp" in name:
#                         param.requires_grad = False
#                     else:
#                         param.requires_grad = True

#     def forward(self, x):
#         return self.gpt2(inputs_embeds=x).last_hidden_state

#     def trainable_parameters(self):
#         for name, param in self.named_parameters():
#             if param.requires_grad:
#                 yield param

# ########### LORA 2 : Freeze all multihead attention layers and apply LORA last U attention layers ##############
# class PFA(nn.Module):
#     def __init__(self, device="cuda:0", gpt_layers=6, U=1):
#         super(PFA, self).__init__()
#         self.gpt2 = GPT2Model.from_pretrained(
#             "gpt2", output_attentions=True, output_hidden_states=True
#         )
#         self.gpt2.h = self.gpt2.h[:gpt_layers]
#         self.U = U
#         self.lora_rank = 4

#         # Adjust parameter training requirements
#         for layer_index, layer in enumerate(self.gpt2.h):
#             for name, param in layer.named_parameters():
#                 if layer_index < gpt_layers - self.U:
#                     if "ln" in name or "wpe" in name:
#                         param.requires_grad = True
#                     else:
#                         param.requires_grad = False
#                 else:
#                     if "mlp" in name or "attn" in name:
#                         param.requires_grad = False
#                     else:
#                         param.requires_grad = True

#         # Apply LoRA to the frozen attention layers in the last U layers
#         self.apply_lora_to_frozen_attention_layers()

#     def apply_lora_to_frozen_attention_layers(self):
#         self.lora_config = LoraConfig(
#             r=self.lora_rank,
#             lora_alpha=16,
#             lora_dropout=0.1,
#             target_modules=["attn.c_attn"],
#             bias="none"
#         )

#         # Apply LoRA only to the frozen attention layers in the last U layers
#         for layer_index in range(len(self.gpt2.h) - self.U, len(self.gpt2.h)):
#             layer = self.gpt2.h[layer_index]
#             for name, param in layer.named_parameters():
#                 if "attn.c_attn" in name:
#                     layer = get_peft_model(layer, self.lora_config)

#     def forward(self, x):
#         return self.gpt2(inputs_embeds=x).last_hidden_state

#     def trainable_parameters(self):
#         for name, param in self.named_parameters():
#             if param.requires_grad:
#                 yield param

# ########### LORA 3 : Freeze all multihead attention layers and apply LORA to all attention layers ##############
# class PFA(nn.Module):
#     def __init__(self, device="cuda:0", gpt_layers=6, U=1):
#         super(PFA, self).__init__()
#         self.gpt2 = GPT2Model.from_pretrained(
#             "gpt2", output_attentions=True, output_hidden_states=True
#         )
#         self.gpt2.h = self.gpt2.h[:gpt_layers]
#         self.U = U
#         self.lora_rank = 4

#         # Adjust parameter training requirements
#         for layer_index, layer in enumerate(self.gpt2.h):
#             for name, param in layer.named_parameters():
#                 if layer_index < gpt_layers - self.U:
#                     if "ln" in name or "wpe" in name:
#                         param.requires_grad = True
#                     else:
#                         param.requires_grad = False
#                 else:
#                     if "mlp" in name or "attn" in name:
#                         param.requires_grad = False
#                     else:
#                         param.requires_grad = True

#         # Apply LoRA to the frozen attention layers in the last U layers
#         self.apply_lora_to_frozen_attention_layers()

#     def apply_lora_to_frozen_attention_layers(self):
#         self.lora_config = LoraConfig(
#             r=self.lora_rank,
#             lora_alpha=16,
#             lora_dropout=0.1,
#             target_modules=["attn.c_attn"],
#             bias="none"
#         )

#         # Apply LoRA only to the frozen attention layers
#         for layer_index, layer in enumerate(self.gpt2.h):
#             for name, param in layer.named_parameters():
#                 if "attn.c_attn" in name and not param.requires_grad:
#                     layer = get_peft_model(layer, self.lora_config)

#     def forward(self, x):
#         return self.gpt2(inputs_embeds=x).last_hidden_state

#     def trainable_parameters(self):
#         for name, param in self.named_parameters():
#             if param.requires_grad:
#                 yield param

# ########## LORA 4 : Apply LORA to first few frozen attention layers only  (First F attention layers are frozen and the last U attention layers are unfrozen similar to the original STLLM setting) ##############
# class PFA(nn.Module):
#     def __init__(self, device="cuda:0", gpt_layers=6, U=1):
#         super(PFA, self).__init__()
#         self.gpt2 = GPT2Model.from_pretrained(
#             "gpt2", output_attentions=True, output_hidden_states=True
#         )
#         self.gpt2.h = self.gpt2.h[:gpt_layers]
#         self.U = U
#         self.lora_rank = 4

#         for layer_index, layer in enumerate(self.gpt2.h):
#             for name, param in layer.named_parameters():
#                 if layer_index < gpt_layers - self.U:
#                     if "ln" in name or "wpe" in name:
#                         param.requires_grad = True
#                     else:
#                         param.requires_grad = False
#                 else:
#                     if "mlp" in name:
#                         param.requires_grad = False
#                     else:
#                         param.requires_grad = True

#         # Apply LoRA to the frozen attention layers in the last U layers
#         self.apply_lora_to_frozen_attention_layers()

#     def apply_lora_to_frozen_attention_layers(self):
#         self.lora_config = LoraConfig(
#             r=self.lora_rank,
#             lora_alpha=16,
#             lora_dropout=0.1,
#             target_modules=["attn.c_attn"],
#             bias="none"
#         )

#         # Apply LoRA only to the frozen attention layers
#         for layer_index, layer in enumerate(self.gpt2.h):
#             for name, param in layer.named_parameters():
#                 if "attn.c_attn" in name and not param.requires_grad:
#                     layer = get_peft_model(layer, self.lora_config)

#     def forward(self, x):
#         return self.gpt2(inputs_embeds=x).last_hidden_state

#     def trainable_parameters(self):
#         for name, param in self.named_parameters():
#             if param.requires_grad:
#                 yield param

class GPT4ST(nn.Module):
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
        self.U = 1
        
        if num_nodes == 170 or num_nodes == 207:
            time = 288
        elif num_nodes == 250 or num_nodes == 266:
            time = 48


        gpt_channel = 192
        to_gpt_channel = 768
            
        self.start_conv = nn.Conv2d(
            self.input_dim * self.input_len, gpt_channel, kernel_size=(1, 1)
        )
        self.trans_gat = nn.Conv2d(self.input_len, gpt_channel, kernel_size=(1, 1))

        self.Temb = TemporalEmbedding(time, gpt_channel)
        self.node_emb = nn.Parameter(torch.empty(self.num_nodes, gpt_channel))
        nn.init.xavier_uniform_(self.node_emb)

        self.gnn_retriever = GNNRetriever(input_dim = self.input_dim, hidden_dim=64, output_dim=1, nheads=8, gnn_type='gat')
        self.in_layer = nn.Conv2d(gpt_channel*4, to_gpt_channel, kernel_size=(1, 1))        

        # regression
        self.regression_layer = nn.Conv2d(to_gpt_channel, self.output_len, kernel_size=(1, 1))   

        self.gpt = PFA(device=self.device, gpt_layers=6, U=self.U)
                 
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

        edge_index , edge_weight = torch_geometric.utils.dense_to_sparse(torch.tensor(self.adj_mx))
        edge_index = edge_index.to(self.device)
        
        # Reshape data for GNN
        data_flat = data.view(B * T * S, F)  # Flatten for GNN input
        
        # Apply GNN retriever
        scores = self.gnn_retriever(data_flat, edge_index)
        scores = scores.view(B, T, S, 1)
        scores = torch.softmax(scores, dim=2)
        
        # Calculate weighted features using the scores
        neighbor_features = data
        weighted_features = torch.einsum('btij,btjk->btik', scores, neighbor_features) #[32, 12, 207, 2]
        weighted_features = weighted_features.permute(0,2,3,1).contiguous() #[32, 207, 2, 12]
        weighted_features = (weighted_features.view(B, S, -1).transpose(1, 2).unsqueeze(-1))
        # print(weighted_features.shape) #[64, 36, 250, 1]

        weighted_features = self.start_conv(weighted_features)
        # print(weighted_features.shape) #[64, 256, 250, 1]
                
        # Combine Temporal Embeddings, Input Embeddings, and Spatial Embeddings
        data_st = torch.cat([input_data] + node_emb + [tem_emb]+ [weighted_features] , dim=1) #  
        # print(data_st.shape) #[64, 768, 250, 1]        
 
        data_st = self.in_layer(data_st)  # Apply linear layer to get embeddings for GPT-2
        data_st = data_st.permute(0, 2, 1, 3).squeeze(-1) 
        # print(data_st.shape) #[64, 250, 768]
        
        outputs = self.gpt(data_st)
            
        outputs = outputs.permute(0, 2, 1).unsqueeze(-1)
#         print(outputs.shape) #[64, 768, 250, 1]       

        # regression
        outputs = self.regression_layer(outputs)  
        #print(outputs.shape) #[64, 12, 250, 1]

        return outputs
