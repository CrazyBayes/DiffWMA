import torch as th
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter
class Merger(nn.Module):
    def __init__(self, head, fea_dim):
        super(Merger, self).__init__()
        self.head = head
        if head > 1:
            self.weight = Parameter(th.Tensor(1, head, fea_dim).fill_(1.))
            self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        :param x: [bs, n_head, fea_dim]
        :return: [bs, fea_dim]
        """
        if self.head > 1:
            return th.sum(self.softmax(self.weight) * x, dim=1, keepdim=False)
        else:
            return th.squeeze(x, dim=1)
class PTDE_RNNAgent_state(nn.Module):
    def __init__(self, input_shape, args):
        super(PTDE_RNNAgent_state, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_allies = args.n_allies
        self.n_enemies = args.n_enemies
        self.n_actions = args.n_actions
        self.n_heads = args.hpn_head_num
        self.rnn_hidden_dim = args.rnn_hidden_dim
        # self.att_dim = args.rnn_hidden_dim // 4
        self.att_dim = 16
        # self.batch_size_run = args.batch_size_run

        self.obs_att_state = args.obs_att_state
        self.Hypernetworks = args.Hypernetworks

        # [4 + 1, (6, 5), (4, 5)]
        self.own_feats_dim, self.enemy_feats_dim, self.ally_feats_dim = input_shape
        self.enemy_feats_dim = self.enemy_feats_dim[-1]  # [n_enemies, feat_dim]
        self.ally_feats_dim = self.ally_feats_dim[-1]  # [n_allies, feat_dim]
        self.new_columns = max(self.own_feats_dim,self.ally_feats_dim)
        ########################################################################################
        #################################  attention过程  ##############################
        ########################################################################################

        self.GIS = nn.Sequential(
            nn.Linear(self.rnn_hidden_dim, args.hpn_hyper_dim),
            nn.ReLU(inplace=True),
            nn.Linear(args.hpn_hyper_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, args.hpn_hyper_dim),
            nn.ReLU(inplace=True),
            nn.Linear(args.hpn_hyper_dim, ((self.args.state_shape + 1) * self.rnn_hidden_dim) * self.n_heads)
        )  # output shape: (state_enemy_feats_size * rnn_hidden_dim + rnn_hidden_dim + 1) * self.n_heads
        self.unify_heads = Merger(self.n_heads, self.rnn_hidden_dim)

        self.yingshe_state = nn.Sequential(
            nn.Linear(self.rnn_hidden_dim, args.hpn_hyper_dim),
            nn.ReLU(inplace=True),
            nn.Linear(args.hpn_hyper_dim, args.hpn_hyper_dim*2),
            nn.LeakyReLU(),
            nn.Linear(args.hpn_hyper_dim*2, args.hpn_hyper_dim),
            nn.ReLU(inplace=True),
            nn.Linear(args.hpn_hyper_dim, self.rnn_hidden_dim)
        )  # output shape: (state_enemy_feats_size * rnn_hidden_dim + rnn_hidden_dim + 1) * self.n_heads

        self.MLP = nn.Sequential(
                nn.Linear(self.rnn_hidden_dim + self.att_dim, args.hpn_hyper_dim),
                nn.ReLU(inplace=True),
                nn.Linear(args.hpn_hyper_dim, args.hpn_hyper_dim*2),
                nn.LeakyReLU(),
                nn.Linear(args.hpn_hyper_dim*2, args.hpn_hyper_dim),
                nn.ReLU(inplace=True),
                nn.Linear(args.hpn_hyper_dim, self.n_actions)
            )
        self.Student_Neiwork = nn.Sequential(
                nn.Linear(self.rnn_hidden_dim, args.hpn_hyper_dim),
                nn.ReLU(inplace=True),
                nn.Linear(args.hpn_hyper_dim, args.hpn_hyper_dim*2),
                nn.LeakyReLU(),
                nn.Linear(args.hpn_hyper_dim*2, args.hpn_hyper_dim),
                nn.ReLU(inplace=True),
                nn.Linear(args.hpn_hyper_dim, self.att_dim)
            )


        if self.args.obs_agent_id:
            # embedding table for agent_id
            # Embedding将输入的整数序列转换为密集向量表示,将每个agents的id表示成一个向量，从而方便进行下一步的计算和处理
            self.agent_id_embedding = th.nn.Embedding(self.n_agents, self.rnn_hidden_dim)

        # if self.args.obs_last_action :
            # embedding table for action id
            # 将每个动作的id表示成一个向量，从而方便进行下一步的计算和处理
        self.action_id_embedding = th.nn.Embedding(self.n_actions, self.rnn_hidden_dim)

        # Unique Features (do not need hyper net，不需要超网络)
        self.fc1_own = nn.Linear(self.own_feats_dim, self.rnn_hidden_dim, bias=True)  # only one bias is OK

        self.rnn = nn.GRUCell(self.rnn_hidden_dim, self.rnn_hidden_dim)
        self.fc2_normal_actions = nn.Linear(self.rnn_hidden_dim, args.output_normal_actions)  # (no_op, stop, up, down, right, left)
        self.zuhe_obs = nn.Linear(self.args.obs_shape, self.args.rnn_hidden_dim)
        self.loss_MSE = th.nn.MSELoss()
        self.fc_mu = th.nn.Linear(self.args.rnn_hidden_dim, self.att_dim)
        self.fc_std = nn.Linear(self.args.rnn_hidden_dim, int(self.att_dim * (self.att_dim + 1) / 2))
        # self.fc_std = th.nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def distribution(self, x, actions=None):
        # 均值向量
        mu = th.tanh(self.fc_mu(x))
        # 下三角矩阵参数
        tri_outputs = self.fc_std(x)
        # 创建一个批量大小的协方巧矩阵
        L = th.zeros(x.size(0), self.att_dim, self.att_dim, device=x.device)
        indices = th.tril_indices(self.att_dim, self.att_dim)
        # 填充下三角
        L[:, indices[0], indices[1]] = tri_outputs
        # 对角线元素应用 exp 来保证正定
        L[:, range(self.att_dim), range(self.att_dim)].exp_()
        # 计算 LL^T
        reg = 1e-6 * th.eye(self.att_dim, device=x.device)
        std_matrix = th.matmul(L, L.transpose(-1, -2)) + reg
        # std_matrix = th.bmm(L, L.transpose(1, 2))

        # std = F.softplus(self.fc_std(x))
        return {'mu': mu, 'std': std_matrix}

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1_own.weight.new(1, self.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state, state_inputs=None, test_mode=False):
        # [bs*n_agents,own_dim],[bs*n_agents*n_enemies,e_fea_dim],[bs*n_agents*n_allies,a_fea_dim]
        bs, own_feats_t, enemy_feats_t, ally_feats_t, embedding_indices = inputs
        h_in = hidden_state.reshape(-1, self.rnn_hidden_dim)

        ########################################################################################
        #################################  组合obs_embeding过程  ##############################
        ########################################################################################
        di_ally_feats_t = ally_feats_t.reshape(bs*self.n_agents, -1)
        di_enemy_feats_t = enemy_feats_t.reshape(bs*self.n_agents, -1)
        di_input = th.cat((own_feats_t, di_ally_feats_t, di_enemy_feats_t), dim=-1)
        embedding_obs = self.zuhe_obs(di_input)  # [bs * n_agents, rnn_hidden_dim]

        # (1) ID embeddings
        # Embedding将输入的整数序列转换为密集向量表示,将每个agents的id表示成一个向量，从而方便进行下一步的计算和处理
        if self.args.obs_agent_id:
            # [bs , n_agents]
            agent_indices = embedding_indices[0]
            # [bs * n_agents, rnn_hidden_dim]
            embedding_obs = embedding_obs + self.agent_id_embedding(agent_indices).view(-1, self.rnn_hidden_dim)
        if self.args.obs_last_action:
            last_action_indices = embedding_indices[-1]
            if last_action_indices is not None:  # t != 0
                # [bs * n_agents, rnn_hidden_dim]
                embedding_obs = embedding_obs + self.action_id_embedding(last_action_indices).view(
                    -1, self.rnn_hidden_dim)

        if not test_mode:
            GIS_out = self.GIS(embedding_obs)
            fc_w = GIS_out[:, :-(self.rnn_hidden_dim) * self.n_heads].reshape(
                 -1, self.args.state_shape, self.rnn_hidden_dim * self.n_heads)
            fc_b = GIS_out[:, -(self.rnn_hidden_dim) * self.n_heads:].reshape(bs * self.n_agents,-1)
            ally_state_t, enemy_state_t = state_inputs
            ally_state_feats=ally_state_t.unsqueeze(1).repeat(1, self.n_agents, 1, 1)
            enemy_state_feats=enemy_state_t.unsqueeze(1).repeat(1, self.n_agents, 1, 1)
            ally_state_feats=ally_state_feats.reshape(bs*self.n_agents, (self.n_agents * (self.args.state_ally_feats_size + self.n_actions)))
            enemy_state_feats=enemy_state_feats.reshape(bs*self.n_agents, (self.n_enemies * self.args.state_enemy_feats_size))
            
            state = th.cat((ally_state_feats, enemy_state_feats), dim=-1)

            embedding_state = th.matmul(state.unsqueeze(1), fc_w)
            
            embedding_state = (embedding_state.squeeze(1)+fc_b).view(bs * self.n_agents, self.n_heads, self.rnn_hidden_dim) 
                # [bs * n_agents, n_enemies, n_heads, rnn_hidden_dim]
                # sum(dim=1)按照维度1加和，加起来。
                # keepdim就似乎size里面是否还保留这个压缩起来的（因为注定是1了，其实没有信息量）
            # embedding_state = embedding_state.sum(dim=1, keepdim=False)  # [bs * n_agents, n_heads, rnn_hidden_dim]
            embedding_state = self.unify_heads(embedding_state)
            embedding_state = self.yingshe_state(embedding_state)

            mu = self.distribution(embedding_state)['mu']
            sigma = self.distribution(embedding_state)['std']

            zt = th.distributions.MultivariateNormal(mu, sigma).sample()

            di_zt = self.Student_Neiwork(embedding_obs)
            di_loss = self.loss_MSE(di_zt, zt.detach())
        else:
            zt = self.Student_Neiwork(embedding_obs)
        
        hh = self.rnn(embedding_obs, h_in)
        q = self.MLP(th.cat((hh, zt), dim=-1))


        
        
        if not test_mode:
            # [bs, n_agents, 6 + n_enemies]
            return q.view(bs, self.n_agents, -1), hh.view(bs, self.n_agents, -1), di_loss
        else:
            return q.view(bs, self.n_agents, -1), hh.view(bs, self.n_agents, -1), None




#         hyper_ally_out = self.hyper_ally(ally_feats.view(bs * self.n_agents * self.n_agents, -1))#(enemy_feats_dim * rnn_hidden_dim + rnn_hidden_dim + 1) * self.n_heads
#             # [bs * n_agents * n_enemies, state_enemy_feats_size, rnn_hidden_dim * n_heads]
#         # fc_w_ally = hyper_ally_out[:, :-(self.rnn_hidden_dim + 1) * self.n_heads].reshape(
#         #          -1, self.args.state_ally_feats_size + self.n_actions, self.rnn_hidden_dim * self.n_heads)
#         # fc_b_ally = hyper_ally_out[:, -(self.rnn_hidden_dim + 1) * self.n_heads:].reshape(bs * self.n_agents*self.n_agents,-1)
#         fc_w_ally = hyper_ally_out[:, :-(self.rnn_hidden_dim) * self.n_heads].reshape(
#                  -1, self.args.state_ally_feats_size + self.n_actions, self.rnn_hidden_dim * self.n_heads)
#         fc_b_ally = hyper_ally_out[:, -(self.rnn_hidden_dim) * self.n_heads:].reshape(bs * self.n_agents*self.n_agents,-1)



#         agent_indices = self.agent_id_embedding(embedding_indices[0]).view(-1, self.rnn_hidden_dim)
#         if self.args.obs_agent_id:
#             # [bs , n_agents]
#             agent_indices = embedding_indices[0]
#             # [bs * n_agents, rnn_hidden_dim]
#             agent_indices = self.agent_id_embedding(agent_indices).view(-1, self.rnn_hidden_dim)
#             di_input = th.cat((di_input,agent_indices), dim = -1)
#         if self.args.obs_last_action:
#             last_action_indices = embedding_indices[-1]
#             if last_action_indices is not None:  # t != 0
#                 # [bs * n_agents, rnn_hidden_dim]
#                 last_action_indices = self.action_id_embedding(last_action_indices).view(-1, self.rnn_hidden_dim)
#                 di_input = th.cat((di_input,last_action_indices), dim = -1)

#         if not test_mode:
#             # [bs, n_agents_s,ally_state_dim], [bs, n_enemies_s,e_state_dim],[bs, n_agents_s,sction_state_dim]
#             ally_state_t, enemy_state_t = state_inputs
#             try:
#                 # 不应该是bs*n_agents，而应该是bs,n_agents ->扩充-> bs*n_agents,n_agents
#                 ally_state_feats=ally_state_t.unsqueeze(1).repeat(1, self.n_agents, 1, 1)
#                 enemy_state_feats=enemy_state_t.unsqueeze(1).repeat(1, self.n_agents, 1, 1)
                
#                 ally_state_feats=ally_state_feats.reshape(bs*self.n_agents, (self.n_agents * (self.args.state_ally_feats_size + self.n_actions)))
#                 enemy_state_feats=enemy_state_feats.reshape(bs*self.n_agents, (self.n_enemies * self.args.state_enemy_feats_size))
                
#                 state = th.cat((ally_state_feats, enemy_state_feats), dim=-1)
#             except:
#                 assert False

#             di_loss = self.diffusion.loss(state, th.cat((di_input, h_in), dim=-1))   

#         di_state=self.diffusion.forward(th.cat((di_input, h_in), dim=-1))

#         di_state_ally_feats = di_state[:, :(self.n_agents * (self.args.state_ally_feats_size + self.n_actions))]
#         di_state_enemy_feats = di_state[:, -(self.n_enemies * self.args.state_enemy_feats_size):]
#         ally_feats = ally_feats_t.cpu().numpy()
#         ally_feats = np.pad(ally_feats, ((0, 0), (0, self.new_columns - ally_feats.shape[1])), mode='constant')
#         own_feats = own_feats_t.cpu().numpy()
#         # own_feats = np.pad(own_feats, ((0, 0), (0, self.new_columns - own_feats.shape[1])), mode='constant')
#         # own_feats    bs*n_agent*8
#         # ally_feats    bs*n_agent*(n_agent-1)*10
#         # obs   bs*n_agent*n_agent*10
#         # state_ally_feats    bs*n_agent*n_agent*12

#         for i in range(own_feats_t.shape[0]):
#             ally_feats = np.insert(ally_feats, i * self.n_agents + i % self.n_agents, own_feats[i], axis=0)
#         ally_feats = th.from_numpy(ally_feats).to(own_feats_t.device)
#         ###############################################################################
#         ########################       超网络提取信息       #############################
#         ###############################################################################

#         hyper_ally_out = self.hyper_ally(ally_feats.view(bs * self.n_agents * self.n_agents, -1))#(enemy_feats_dim * rnn_hidden_dim + rnn_hidden_dim + 1) * self.n_heads
#             # [bs * n_agents * n_enemies, state_enemy_feats_size, rnn_hidden_dim * n_heads]
#         # fc_w_ally = hyper_ally_out[:, :-(self.rnn_hidden_dim + 1) * self.n_heads].reshape(
#         #          -1, self.args.state_ally_feats_size + self.n_actions, self.rnn_hidden_dim * self.n_heads)
#         # fc_b_ally = hyper_ally_out[:, -(self.rnn_hidden_dim + 1) * self.n_heads:].reshape(bs * self.n_agents*self.n_agents,-1)
#         fc_w_ally = hyper_ally_out[:, :-(self.rnn_hidden_dim) * self.n_heads].reshape(
#                  -1, self.args.state_ally_feats_size + self.n_actions, self.rnn_hidden_dim * self.n_heads)
#         fc_b_ally = hyper_ally_out[:, -(self.rnn_hidden_dim) * self.n_heads:].reshape(bs * self.n_agents*self.n_agents,-1)
        
#              # torch.matmul两个张量矩阵相乘
#             # [bs * n_agents * n_enemies, 1, state_enemy_feats_size] * [bs * n_agents * n_enemies, state_enemy_feats_size, rnn_hidden_dim * n_heads] = [bs * n_agents * n_enemies, 1, rnn_hidden_dim]
#         embedding_allies = th.matmul(di_state_ally_feats.view(bs, self.n_agents, self.n_agents, -1)
#                                        .reshape(bs*self.n_agents*self.n_agents, -1).unsqueeze(1), fc_w_ally)
        
#         embedding_allies = (embedding_allies.squeeze(1)+fc_b_ally).view(bs * self.n_agents* self.n_agents, self.n_heads, self.rnn_hidden_dim) 
#          # [bs * n_agents, n_enemies, n_heads, rnn_hidden_dim]
#             # sum(dim=1)按照维度1加和，加起来。
#             # keepdim就似乎size里面是否还保留这个压缩起来的（因为注定是1了，其实没有信息量）
#         embedding_allies = embedding_allies.sum(dim=1, keepdim=False)  # [bs * n_agents, n_heads, rnn_hidden_dim]
# ################################################################################
# ###############################  此处没有加merge   ##############################
# ###############################################################################
#         hyper_enemy_out = self.hyper_enemy(enemy_feats_t.view(bs * self.n_agents * self.n_enemies, -1))#(enemy_feats_dim * rnn_hidden_dim + rnn_hidden_dim + 1) * self.n_heads
#             # [bs * n_agents * n_enemies, state_enemy_feats_size, rnn_hidden_dim * n_heads]
#         fc_w_enemy = hyper_enemy_out[:, :-(self.rnn_hidden_dim) * self.n_heads].reshape(
#                  -1, self.args.state_enemy_feats_size, self.rnn_hidden_dim * self.n_heads)
#         fc_b_enemy = hyper_enemy_out[:, -(self.rnn_hidden_dim) * self.n_heads:].reshape(bs * self.n_agents*self.n_enemies,-1)
#              # torch.matmul两个张量矩阵相乘
#             # [bs * n_agents * n_enemies, 1, state_enemy_feats_size] * [bs * n_agents * n_enemies, state_enemy_feats_size, rnn_hidden_dim * n_heads] = [bs * n_agents * n_enemies, 1, rnn_hidden_dim]

#         embedding_enemies = th.matmul(di_state_enemy_feats.view(bs, self.n_agents, self.n_enemies, -1)
#                                        .reshape(bs*self.n_agents*self.n_enemies, -1).unsqueeze(1), fc_w_enemy)
        
#         embedding_enemies = (embedding_enemies.squeeze(1)+fc_b_enemy).view(bs * self.n_agents* self.n_enemies, self.n_heads, self.rnn_hidden_dim) 
       
#         embedding_enemies = embedding_enemies.sum(dim=1, keepdim=False)  # [bs * n_agents, n_heads, rnn_hidden_dim]


#         ########################################################################################
#         #######################################  PI过程  #######################################
#         ########################################################################################
#         # (1) Own feature，自特征，不需要超网络，输出维度rnn_hidden_dim即64
#         embedding_own = self.fc1_own(own_feats_t)  # [bs * n_agents, rnn_hidden_dim]

#         # (2) ID embeddings
#         # Embedding将输入的整数序列转换为密集向量表示,将每个agents的id表示成一个向量，从而方便进行下一步的计算和处理
#         if self.args.obs_agent_id:
#             # [bs , n_agents]
#             agent_indices = embedding_indices[0]
#             # [bs * n_agents, rnn_hidden_dim]
#             embedding_own = embedding_own + self.agent_id_embedding(agent_indices).view(-1, self.rnn_hidden_dim)
#         if self.args.obs_last_action:
#             last_action_indices = embedding_indices[-1]
#             if last_action_indices is not None:  # t != 0
#                 # [bs * n_agents, rnn_hidden_dim]
#                 embedding_own = embedding_own + self.action_id_embedding(last_action_indices).view(
#                     -1, self.rnn_hidden_dim)

#             # (2) Ally att features
#             # [bs*n_agents, ally_feats_dim * rnn_hidden_dim * n_heads]
#         att_hyper_ally_out = self.att_hyper_ally(embedding_allies)
#         if self.args.map_type == "MMM":
#             # [bs * n_agents, ally_fea_dim, rnn_hidden_dim * head]
#                 fc1_att_w_ally = att_hyper_ally_out[:, :-(self.rnn_hidden_dim + 1) * self.n_heads].reshape(
#                     -1, self.rnn_hidden_dim, self.rnn_hidden_dim * self.n_heads)
#         else:
#             # [bs * n_agents, ally_fea_dim, rnn_hidden_dim * head]
#                 fc1_att_w_ally = att_hyper_ally_out.view(-1, self.rnn_hidden_dim, self.rnn_hidden_dim * self.n_heads)
#             # [bs * n_agents, 1, ally_fea_dim + self.n_actions] * [bs * n_agents, ally_fea_dim + self.n_actions, n_heads* rnn_hidden_dim] = [bs * n_agents, 1, n_heads*rnn_hidden_dim]
#         embedding_att_allies = th.matmul(embedding_allies.unsqueeze(1), fc1_att_w_ally).view(
#                 bs * self.n_agents, self.n_agents, self.n_heads, self.rnn_hidden_dim)  # [bs * n_agents, n_agents, head, rnn_hidden_dim]
#         embedding_att_allies = embedding_att_allies.sum(dim=1, keepdim=False)  # [bs * n_agents, head, rnn_hidden_dim]

#         # (3) Enemy att feature  (enemy_feats_dim * rnn_hidden_dim + rnn_hidden_dim + 1) * self.n_heads
#         # enemy_feats_t:[bs*n_agents*n_enemies,e_fea_dim]
#         att_hyper_enemy_out = self.att_hyper_enemy(embedding_enemies)#(enemy_feats_dim * rnn_hidden_dim + rnn_hidden_dim + 1) * self.n_heads
#             # [bs * n_agents * n_enemies, state_enemy_feats_size, rnn_hidden_dim * n_heads]
#         fc1_att_w_enemy = att_hyper_enemy_out[:, :-(self.rnn_hidden_dim + 1) * self.n_heads].reshape(
#                  -1, self.rnn_hidden_dim, self.rnn_hidden_dim * self.n_heads)
        
#              # torch.matmul两个张量矩阵相乘
#             # [bs * n_agents * n_enemies, 1, state_enemy_feats_size] * [bs * n_agents * n_enemies, state_enemy_feats_size, rnn_hidden_dim * n_heads] = [bs * n_agents * n_enemies, 1, rnn_hidden_dim]
#         embedding_att_enemies = th.matmul(embedding_enemies.unsqueeze(1), fc1_att_w_enemy).view(
#                 bs * self.n_agents, self.n_enemies, self.n_heads, self.rnn_hidden_dim)  # [bs * n_agents, n_enemies, n_heads, rnn_hidden_dim]
#             # sum(dim=1)按照维度1加和，加起来。
#             # keepdim就似乎size里面是否还保留这个压缩起来的（因为注定是1了，其实没有信息量）
#         embedding_att_enemies = embedding_att_enemies.sum(dim=1, keepdim=False)  # [bs * n_agents, n_heads, rnn_hidden_dim]

#             # Final embedding
#         embedding_att =self.unify_att_heads(embedding_att_allies + embedding_att_enemies)  # [bs * n_agents, head, rnn_hidden_dim]
#             # embedding_state:[bs * n_agent, rnn_hidden_dim], y:[bs * n_agent, rnn_hidden_dim * 2]
            
#         # Final embedding
#         final_embedding = embedding_own + embedding_att

#         ########################################################################################
#         ###################################  模块B与C  #################################
#         ########################################################################################
#         # 把di_embedding_state整合进embedding中作为输入
#         x = F.relu(final_embedding, inplace=True)
#         ###########################
#         hh = self.rnn(x, h_in)  # [bs * n_agents, rnn_hidden_dim]


#         # Q-values of normal actions
#         q_normal = self.fc2_normal_actions(hh).view(bs, self.n_agents, -1)  # [bs, n_agents, 6]

#         ########################################################################################
#         ########################################  PE过程  ######################################
#         ########################################################################################

#         # Q-values of attack actions: [bs * n_agents * n_enemies, rnn_hidden_dim * n_heads]
#         fc2_w_attack = att_hyper_enemy_out[:, -(self.rnn_hidden_dim + 1) * self.n_heads: -self.n_heads].reshape(
#             bs * self.n_agents, self.n_enemies, self.rnn_hidden_dim, self.n_heads
#         ).transpose(1, 2).reshape(  # -> [bs * n_agents, rnn_hidden_dim, n_enemies, n_heads]
#             bs * self.n_agents, self.rnn_hidden_dim, self.n_enemies * self.n_heads
#         )  # [bs * n_agents, rnn_hidden_dim, n_enemies * heads]
#         fc2_b_attack = att_hyper_enemy_out[:, -self.n_heads:].reshape(bs * self.n_agents, self.n_enemies * self.n_heads)

#         # [bs*n_agents, 1, rnn_hidden_dim] * [bs*n_agents, rnn_hidden_dim, n_enemies*head] -> [bs*n_agents, 1, n_enemies*head]
#         q_attacks = (th.matmul(hh.unsqueeze(1), fc2_w_attack).squeeze(1) + fc2_b_attack).view(
#             bs * self.n_agents * self.n_enemies, self.n_heads, 1
#         )  # [bs * n_agents, n_enemies*head] -> [bs * n_agents * n_enemies, head, 1]

#         # Merge multiple heads into one.
#         q_attack = self.unify_output_heads(q_attacks).view(  # [bs * n_agents * n_enemies, 1]
#             bs, self.n_agents, self.n_enemies
#         )  # [bs, n_agents, n_enemies]

#         # %%%%%%%%%%%%%%% 'rescue' actions for map_type == "MMM" %%%%%%%%%%%%%%%
#         if self.args.map_type == "MMM":
#             fc2_w_rescue = att_hyper_ally_out[:, -(self.rnn_hidden_dim + 1) * self.n_heads: -self.n_heads].reshape(
#                 bs * self.n_agents, self.n_agents, self.rnn_hidden_dim, self.n_heads
#             ).transpose(1, 2).reshape(  # -> [bs * n_agents, rnn_hidden_dim, n_allies, n_heads]
#                 bs * self.n_agents, self.rnn_hidden_dim, self.n_agents * self.n_heads
#             )  # [bs * n_agents, rnn_hidden_dim, n_allies * heads]
#             fc2_b_rescue = att_hyper_ally_out[:, -self.n_heads:].reshape(bs * self.n_agents, self.n_agents * self.n_heads)
#             # [bs*n_agents, 1, rnn_hidden_dim] * [bs*n_agents, rnn_hidden_dim, n_allies*head] -> [bs*n_agents, 1, n_allies*head]
#             q_rescues = (th.matmul(hh.unsqueeze(1), fc2_w_rescue).squeeze(1) + fc2_b_rescue).view(
#                 bs * self.n_agents * self.n_agents, self.n_heads, 1
#             )  # [bs * n_agents, n_allies*head] -> [bs * n_agents * n_allies, head, 1]
#             # Merge multiple heads into one.
#             q_rescue = self.unify_output_heads_rescue(q_rescues).view(  # [bs * n_agents * n_allies, 1]
#                 bs, self.n_agents, self.n_agents
#             )  # [bs, n_agents, n_allies]

#             # For the reason that medivac is the last indexed agent, so the rescue action idx -> [0, n_allies-1]
#             right_padding = th.ones_like(q_attack[:, -1:, self.n_agents:], requires_grad=False) * (-9999999)
#             modified_q_attack_of_medivac = th.cat([q_rescue[:, -1:, :], right_padding], dim=-1)
#             q_attack = th.cat([q_attack[:, :-1], modified_q_attack_of_medivac], dim=1)

#         # Concat 2 types of Q-values
#         q = th.cat((q_normal, q_attack), dim=-1)  # [bs, n_agents, 6 + n_enemies]
        
        
#         if not test_mode:
#             # [bs, n_agents, 6 + n_enemies]
#             return q.view(bs, self.n_agents, -1), hh.view(bs, self.n_agents, -1), di_loss
#         else:
#             return q.view(bs, self.n_agents, -1), hh.view(bs, self.n_agents, -1), None