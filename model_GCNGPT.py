import torch
import torch.nn as nn
import torch_geometric

from transformers import GPT2Model, GPT2Tokenizer
from torch_geometric.nn import GCNConv, GATConv

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
                    if "mlp" in name:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True

    def forward(self, x):
        return self.gpt2(inputs_embeds=x).last_hidden_state

class GCNGPT(nn.Module):
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
        self.U = 2
        
        if num_nodes == 170 or num_nodes == 207 or num_nodes == 307:
            time = 288
        elif num_nodes == 250 or num_nodes == 266:
            time = 48


        gpt_channel = 768
            
        self.start_conv = nn.Conv2d(
            self.input_dim * self.input_len, gpt_channel, kernel_size=(1, 1)
        )

        self.gat = GCNConv(in_channels = gpt_channel, out_channels = gpt_channel)        

        # regression
        self.regression_layer = nn.Conv2d(gpt_channel, self.output_len, kernel_size=(1, 1))

        self.gpt = PFA(device=self.device, gpt_layers=6, U=self.U)
                 
    # return the total parameters of model
    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])

    def forward(self, history_data):

        data = history_data.permute(0, 3, 2, 1)
        B, T, S, F = data.shape
        # print(data.shape) #[64, 12, 250, 3]
        
        input_data = data.transpose(1, 2).contiguous()
        input_data = (input_data.view(B, S, -1).transpose(1, 2).unsqueeze(-1))
        # print(input_data.shape) #[64, 36, 307, 1]

        data_st = self.start_conv(input_data)
        # print(data_st.shape)#[64, 768, 307, 1]

        # Reshape data for GNN
        data_flat = data_st.view(B*S, -1)  # Flatten for GNN input
        # print(data_flat.shape) #[19648, 768]

        edge_index , edge_weight = torch_geometric.utils.dense_to_sparse(torch.tensor(self.adj_mx))
        edge_index = edge_index.to(self.device)
        
        data_st = self.gat(data_flat, edge_index) + data_flat
        data_st = data_st.view(B,S,-1)        
        outputs = self.gpt(data_st)
            
        outputs = outputs.permute(0, 2, 1).unsqueeze(-1)
        # print(outputs.shape) #[64, 768, 250, 1]       

        # regression
        outputs = self.regression_layer(outputs)  
        # print(outputs.shape) #[64, 12, 250, 1]

        return outputs
