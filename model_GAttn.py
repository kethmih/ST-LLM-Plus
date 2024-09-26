import torch
import pickle
import torch.nn as nn
import numpy as np
import torch_geometric
from transformers import GPT2Model, GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention

from peft import LoraConfig, get_peft_model
from torch_geometric.nn import GCNConv, GATConv

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


class GraphAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(GraphAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, hidden_states, adjacency_matrix):
        batch_size, seq_len, embed_dim = hidden_states.size()

        # Project Q, K, V matrices
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Perform graph-based attention: mask with adjacency matrix
        attn_scores = torch.matmul(q, k.transpose(-2, -1))  # [batch, heads, seq_len, seq_len]
        attn_scores = attn_scores / np.sqrt(self.head_dim)

        # print(attn_scores.shape) #[64, 12, 250, 250]
        # print(adjacency_matrix.shape) #[250, 250]

        #Repeat number of batches  
        adjacency_matrix = adjacency_matrix.unsqueeze(0).repeat(batch_size, 1, 1).to(hidden_states.device)

        # Apply adjacency matrix as weights (proximity-based weighting)
        # adjacency_matrix is [batch, seq_len, seq_len], add a dimension for heads
        attn_scores = attn_scores * adjacency_matrix.unsqueeze(1)    
        
        # print(attn_scores.shape)#[64, 12, 250, 250]

        attn_scores = self.leaky_relu(attn_scores)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # Compute attention output
        attn_output = torch.matmul(attn_weights.float(), v.float())  # [batch, heads, seq_len, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)

        # Apply output projection
        attn_output = self.out_proj(attn_output)

        return attn_output

### Custom Attention with Graph Multihead Attention
# class CustomGPT2Attention(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.graph_attn = GraphAttention(config.hidden_size, config.num_attention_heads)

#     def forward(self, hidden_states,  layer_past=None, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, use_cache=False, output_attentions=False, adjacency_matrix=None):
#         # Ensure adjacency_matrix is provided since we only use graph attention
#         if adjacency_matrix is not None:
#             attn_output = self.graph_attn(hidden_states, adjacency_matrix)
#         else:
#             raise ValueError("Adjacency matrix is required for graph attention.")

#         # Return the output (you no longer need to return `presents` for caching)
#         return (attn_output,None)

### Custom Attention with Graph Attention Convolution
class CustomGPT2Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.graph_attn = GATConv(config.hidden_size, config.hidden_size // config.num_attention_heads, heads=config.num_attention_heads)

    def forward(self, hidden_states,  layer_past=None, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, use_cache=False, output_attentions=False, adjacency_matrix=None):
        # Ensure adjacency_matrix is provided since we only use graph attention
        if adjacency_matrix is not None:
            edge_index , edge_weight = torch_geometric.utils.dense_to_sparse(torch.tensor(adjacency_matrix))
            edge_index = edge_index.to(hidden_states.device) 
            
            out = []
            for b in range(hidden_states.shape[0]):
                attn_output = self.graph_attn(hidden_states[b], edge_index)
                out.append(attn_output)
            out = torch.stack(out, dim=0)
        else:
            raise ValueError("Adjacency matrix is required for graph attention.")

        # Return the output (you no longer need to return `presents` for caching)
        return (out, None)


from transformers.models.gpt2.modeling_gpt2 import GPT2Model, GPT2Block

class CustomGPT2Model(GPT2Model):
    def __init__(self, config):
        super().__init__(config)
        # Replace GPT-2 blocks with custom blocks that use graph attention
        for i, block in enumerate(self.h):
            self.h[i] = CustomGPT2Block(config)  # Replace each block with the custom block


class CustomGPT2Block(GPT2Block):
    def __init__(self, config):
        super().__init__(config)
        self.attn = CustomGPT2Attention(config)  # Replace attention with custom attention

    def forward(self, hidden_states, layer_past=None, attention_mask=None, head_mask=None, use_cache=False, output_attentions=False, adjacency_matrix=adjacency_matrix, **kwargs):
        # Override to pass adjacency_matrix to the attention mechanism
        # **kwargs to ignore unexpected arguments (e.g., encoder_hidden_states)
        
        residual = hidden_states

        # 1. Attention Layer (Custom Attention with Graph)
        attn_outputs = self.attn(
            hidden_states, 
            layer_past=layer_past, 
            attention_mask=attention_mask, 
            head_mask=head_mask, 
            use_cache=use_cache, 
            output_attentions=output_attentions, 
            adjacency_matrix=adjacency_matrix  # Pass adjacency matrix here
        )
        attn_output = attn_outputs[0]  # Attention output
        outputs = attn_outputs[1:]  # Extract other outputs (if any)

        # Residual connection
        hidden_states = attn_output + residual

        # 2. LayerNorm + MLP
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,)

        if output_attentions:
            outputs = outputs + (attn_outputs[0],)

        return outputs  # Return the hidden states and other optional outputs (e.g., attention)


class PFA(nn.Module):
    def __init__(self, adj, device="cuda:0", gpt_layers=6, U=1):
        super(PFA, self).__init__()
        # Load the GPT2Config
        # config = GPT2Config.from_pretrained('gpt2')
        config = GPT2Config(n_layer=6)  # Set n_layer to 6

        # Initialize the custom model with the config
        custom_model = CustomGPT2Model(config)

        # Load pretrained weights from GPT-2 into the custom model
        pretrained_model = GPT2Model.from_pretrained('gpt2')
        pretrained_model.h = pretrained_model.h[:gpt_layers]
        custom_model.load_state_dict(pretrained_model.state_dict(), strict=False)

        self.gpt2 = custom_model
        self.U = U

        for layer_index, layer in enumerate(self.gpt2.h):
            for name, param in layer.named_parameters():
                if layer_index < gpt_layers - self.U:
                    if "ln" in name or "wpe" in name : #or "attn" in name
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

        
class GAttn(nn.Module):
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

        # edge_index , edge_weight = torch_geometric.utils.dense_to_sparse(torch.tensor(self.adj_mx))
        # edge_index = edge_index.to(self.device)
        adj = torch.tensor(self.adj_mx)
        self.gpt = PFA(adj, device=self.device, gpt_layers=6, U=self.U)
                 
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
        
        outputs = self.gpt(data_st)
            
        outputs = outputs.permute(0, 2, 1).unsqueeze(-1)
#         print(outputs.shape) #[64, 768, 250, 1]       

        # regression
        outputs = self.regression_layer(outputs)  
        # print(outputs.shape) #[64, 12, 250, 1]

        return outputs
