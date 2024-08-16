import torch
import torch.nn as nn
from einops import rearrange
from transformers.models.gpt2.modeling_gpt2 import GPT2Model

class OFA(nn.Module):
    def __init__(self, device="cuda:0", gpt_layers=6):
        super(OFA, self).__init__()
        self.gpt2 = GPT2Model.from_pretrained(
            "gpt2", output_attentions=True, output_hidden_states=True
        )
        self.gpt2.h = self.gpt2.h[:gpt_layers]

        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            if 'ln' in name or 'wpe' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def forward(self, x):
        return self.gpt2(inputs_embeds=x).last_hidden_state

class GPT4TS(nn.Module):
    def __init__(
        self,
        device,
        input_dim=3,
        channels=64,
        num_nodes=170,
        input_len=12,
        output_len=12,
        dropout=0.1,
        U=1
    ):
        super().__init__()

        # attributes
        self.device = device
        self.num_nodes = num_nodes
        self.node_dim = channels
        self.input_len = input_len
        self.input_dim = input_dim
        self.output_len = output_len

        self.patch_size = 16
        self.stride = 8
        self.d_model = 768
        self.patch_num = (self.input_len - self.patch_size) // self.stride + 1
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_num += 1

        self.in_layer = nn.Linear(self.patch_size, self.d_model)
        self.gpt = OFA(device=self.device)
        self.out_layer = nn.Linear(self.d_model * self.patch_num, self.output_len)

    # return the total parameters of model
    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])

    def forward(self, history_data):
        input_data = history_data
        # print(history_data.shape) [64, 3, 307, 12]
        batch_size, _, num_nodes, _ = input_data.shape
        history_data = history_data.permute(0, 3, 2, 1)
        B, T, S, F = history_data.shape

        x = history_data.reshape(B, T, S*F)
        B, L, M = x.shape

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False)+ 1e-5).detach() 
        x /= stdev

        x = rearrange(x, 'b l m -> b m l')

        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x = rearrange(x, 'b m n p -> (b m) n p')

        outputs = self.in_layer(x)
        print(outputs.shape)
        
        outputs = self.gpt(outputs)
        print(outputs.shape)

        outputs = self.out_layer(outputs.reshape(B*M, -1))
        outputs = rearrange(outputs, '(b m) l -> b l m', b=B)

        outputs = outputs * stdev
        outputs = outputs + means
        print(outputs.shape)
        

        return prediction
