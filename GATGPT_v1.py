import torch
import torch.nn as nn
import torch_geometric
import torch.nn.functional as F
from transformers import GPT2Model
from torch_geometric.nn import GATConv

class GAT(nn.Module):
    def __init__(self, in_channels, out_channels, heads=8, dropout=0.1):
        super(GAT, self).__init__()
        self.gat1 = GATConv(in_channels, out_channels // heads, heads=heads, dropout=dropout)
        self.gat2 = GATConv(out_channels, out_channels // heads, heads=heads, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)
        return x

class TemporalEmbedding(nn.Module):
    def __init__(self, time, features):
        super(TemporalEmbedding, self).__init__()
        self.time = time

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

        tem_emb = time_day + time_week
        return tem_emb

class PFA(nn.Module):
    def __init__(self, device="cuda:0", gpt_layers=6, U=1):
        super(PFA, self).__init__()
        self.gpt2 = GPT2Model.from_pretrained(
            "gpt2", output_attentions=True, output_hidden_states=True
        )
        self.gpt2.h = self.gpt2.h[:gpt_layers]
        self.U = U

        for layer_index, layer in enumerate(self.gpt2.h):
            for name, param in layer.named_parameters():
                if layer_index < gpt_layers - self.U:
                    if "ln" in name or "wpe" in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                else:
                    # if "mlp" in name:
                    #     param.requires_grad = False
                    # else:
                    param.requires_grad = True

    def forward(self, x):
        return self.gpt2(inputs_embeds=x).last_hidden_state

class GPT4ST(nn.Module):
    def __init__(
        self,
        device,
        adj_mx,
        input_dim=3,
        num_nodes=170,
        input_len=12,
        output_len=12,
        llm_layer=1,
        U=1,
        channel=128,
        head = 4    
        ):
        super().__init__()

        self.device = device
        self.adj_mx = torch.tensor(adj_mx, dtype=torch.float32).to(self.device)
        self.num_nodes = num_nodes
        self.input_len = input_len
        self.input_dim = input_dim
        self.output_len = output_len
        self.llm_layer = llm_layer
        self.U = U
        self.channel = channel
        self.head = head

        if num_nodes == 170 or num_nodes == 307:
            time = 288
        elif num_nodes == 250 or num_nodes == 266:
            time = 48

        gpt_channel = 192
        to_gpt_channel = 768
            
        self.start_conv = nn.Conv2d(
            self.input_dim * self.input_len, gpt_channel, kernel_size=(1, 1)
        )

        self.Temb = TemporalEmbedding(time, gpt_channel)
        self.node_emb = nn.Parameter(torch.empty(self.num_nodes, gpt_channel))
        nn.init.xavier_uniform_(self.node_emb)

        self.feature_fusion = nn.Conv2d(
            gpt_channel * 4, to_gpt_channel, kernel_size=(1, 1)
        )
        
        # LayerNorm, FFN, and regression layers
        self.layer_norm1 = nn.LayerNorm(gpt_channel)
        self.layer_norm2 = nn.LayerNorm(gpt_channel)
        self.ffn = nn.Sequential(
            nn.Linear(gpt_channel, gpt_channel * 4),
            nn.ReLU(),
            nn.Linear(gpt_channel * 4, gpt_channel)
        )
        
        self.gat = GAT(gpt_channel, gpt_channel, heads=self.head, dropout=0.1)
        self.gpt = PFA(device=self.device, gpt_layers=self.llm_layer, U=self.U)
        self.regression_layer = nn.Conv2d(to_gpt_channel, self.output_len, kernel_size=(1, 1))   
                 
    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])

    def forward(self, history_data):
        data = history_data.permute(0, 3, 2, 1)
        B, T, S, F = data.shape
        
        tem_emb = self.Temb(data).squeeze(-1)

        node_emb = []
        node_emb.append(
            self.node_emb.unsqueeze(0)
            .expand(B, -1, -1)
            .transpose(1, 2)
        )

        input_data = data.permute(0,3,2,1)
        input_data = input_data.transpose(1, 2).contiguous()
        input_data = (input_data.view(B, S, -1).transpose(1, 2).unsqueeze(-1))
        token_emb = self.start_conv(input_data).squeeze(-1) # B E S

        # Layer norm 1
        ln1_in = token_emb.permute(0, 2, 1) # B S E
        ln2_out = self.layer_norm1(ln1_in)
        ln2_out = ln2_out.permute(0, 2, 1)  # B E S

        # GAT forward pass
        edge_index = torch_geometric.utils.dense_to_sparse(self.adj_mx)[0].to(self.device)
        gat_out = []
        for i in range(B):
            gat_out.append(self.gat(ln2_out[i].permute(1, 0), edge_index))  # Apply GAT for each sample in batch
        gat_out = torch.stack(gat_out, dim=0)

        # Res connection
        gat_out_add = gat_out + ln1_in  # B S E
        
        # Layer norm 2
        skip_output_2 = self.layer_norm2(gat_out_add)
        
        # Feed-forward network
        ffn_out = self.ffn(skip_output_2) + gat_out_add
        ffn_out = ffn_out.permute(0, 2, 1)
        # print(input_data.shape)
        # print(ffn_out.shape)
        # Fusion
        data_st = torch.cat([token_emb] + node_emb + [tem_emb] + [ffn_out], dim=1)
        data_st = data_st.unsqueeze(-1) # B E S 1
        data_st = self.feature_fusion(data_st)
        data_st = data_st.squeeze(-1)
        data_st = data_st.permute(0, 2, 1)
       
        # GPT
        gpt_out = self.gpt(data_st)
        gpt_out = gpt_out.permute(0, 2, 1).unsqueeze(-1) 

        # print("Shape of gpt_out:",gpt_out.shape) 64, 768, 266, 1
        rec_out = self.regression_layer(gpt_out)
        # print("Shape of rec_out:",rec_out.shape) 64, 12, 266, 1

        return rec_out