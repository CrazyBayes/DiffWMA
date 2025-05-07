import torch.nn as nn
import torch
import torch.nn.functional as F
from modules.diffusion.DWNdiffusion import GaussianDiffusion
from modules.diffusion.denoiser import Denoiser
from utils.th_utils import orthogonal_init_
from torch.nn import LayerNorm

class DWN_RNN_Agent(nn.Module):
    def __init__(self, input_shape, args):
        super(DWN_RNN_Agent, self).__init__()
        self.args = args
        denoiser = Denoiser(args)
        self.Diffusion_WorldModel = GaussianDiffusion(args, denoiser)
        self.fc1 = nn.Linear(args.state_shape+input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        if getattr(args, "use_layer_norm", False):
            self.layer_norm = LayerNorm(args.rnn_hidden_dim)

        if getattr(args, "use_orthogonal", False):
            orthogonal_init_(self.fc1)
            orthogonal_init_(self.fc2, gain=args.gain)

    def train_DWN(self, x, cond, DWM_W):
        return self.Diffusion_WorldModel.train_DWN(x, cond, DWM_W)

    def perform_DWN(self, cond):
        with torch.no_grad():
            return  self.Diffusion_WorldModel(cond)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        b, a, _ = inputs.size()
        DWM_h = hidden_state.reshape(b,a, -1)

        cond = torch.cat([inputs, DWM_h], dim=-1)
        pred_stat = self.Diffusion_WorldModel(cond).detach()[..., :self.args.state_shape]

        ####
        an_mask = torch.isnan(pred_stat)

        # 输出 NaN 的布尔值

        # 检查 tensor 中是否有任何 NaN 值
        contains_nan = torch.any(an_mask)

        # 输出是否包含 NaN
        if contains_nan:
            print(f"Tensor contains NaN: {contains_nan.item()}")
        ####
        inputs = torch.cat([pred_stat, inputs], dim=-1)
        _, _, e = inputs.size()

        
        x = F.relu(self.fc1(inputs.view(-1, e)), inplace=True)
        if hidden_state is not None:
            hidden_state = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        hh = self.rnn(x, hidden_state)
        #DWN_h = h.reshape(b,a, -1)
        #cond = torch.cat([inputs,DWN_h], dim=-1)
        #pred_stat = self.Diffusion_WorldModel(cond)
        #final_input = torch.cat([pred_stat, input, DWN_h], dim=-1)
        #print(final_input.shape)
        #q = self.fc2(h)

        if getattr(self.args, "use_layer_norm", False):
            q = self.fc2(self.layer_norm(hh))
        else:
            q = self.fc2(hh)
        return q.view(b, a, -1), hh.view(b, a, -1)