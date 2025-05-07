import copy

import torch

from components.episode_buffer import EpisodeBatch
from modules.mixers.nmix import Mixer
from modules.mixers.vdn import VDNMixer
from modules.mixers.qatten import QattenMixer
from utils.rl_utils import build_td_lambda_targets, build_q_lambda_targets
import torch as th
from torch.optim import RMSprop, Adam
import numpy as np
from utils.th_utils import get_parameters_num
from components.transforms import OneHot
#检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class DWNQLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.onehot = OneHot(out_dim=args.n_actions)
        self.epsilon_flag = False
        
        self.last_target_update_episode = 0
        self.device = th.device('cuda' if args.use_cuda  else 'cpu')
        self.params = list(mac.parameters())

        if args.mixer == "qatten":
            self.mixer = QattenMixer(args)
        elif args.mixer == "vdn":
            self.mixer = VDNMixer()
        elif args.mixer == "qmix":
            self.mixer = Mixer(args)
        else:
            raise "mixer error"
        self.target_mixer = copy.deepcopy(self.mixer)
        self.params += list(self.mixer.parameters())

        self.global_state_pred = torch.nn.Sequential(
            torch.nn.Linear(self.args.n_agents * self.args.state_shape, self.args.DWN_state_pred_h),
            torch.nn.ReLU(),
            torch.nn.Linear(self.args.DWN_state_pred_h, self.args.DWN_state_pred_h),
            torch.nn.ReLU(),
            torch.nn.Linear(self.args.DWN_state_pred_h, self.args.state_shape)
        ).to(device)
        self.global_reward_pred = torch.nn.Sequential(
            torch.nn.Linear(self.args.n_agents, self.args.DWN_reward_pred_h),
            torch.nn.ReLU(),
            torch.nn.Linear(self.args.DWN_reward_pred_h, self.args.DWN_reward_pred_h),
            torch.nn.ReLU(),
            torch.nn.Linear(self.args.DWN_reward_pred_h,  1)
        ).to(device)
        self.params_sta_pre = list(self.global_reward_pred.parameters())
        self.params_re_pre = list(self.global_reward_pred.parameters())



        print('Mixer Size: ')
        print(get_parameters_num(self.mixer.parameters()))

        if self.args.optimizer == 'adam':
            self.optimiser = Adam(params=self.params,  lr=args.lr, weight_decay=getattr(args, "weight_decay", 0))
            self.optimiser_stat_pre = Adam(params=self.params_sta_pre,  lr=args.lr, weight_decay=getattr(args, "weight_decay", 0))
            self.optimiser_re_pre = Adam(params=self.params_re_pre,  lr=args.lr, weight_decay=getattr(args, "weight_decay", 0))
        else:
            self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)
        self.log_stats_t = -self.args.learner_log_interval - 1
        self.train_t = 0

        # priority replay
        self.use_per = getattr(self.args, 'use_per', False)
        self.return_priority = getattr(self.args, "return_priority", False)
        if self.use_per:
            self.priority_max = float('-inf')
            self.priority_min = float('inf')

    def train_traj(self, state, actions, observation, action_onehot, action_before_onehot,
                   agent_id, t_env, hidden):

        X_state = state
        X_actions = actions
        X_observations = observation
        X_action_onehot = action_onehot
        X_action_before_onehot = action_before_onehot
        self.mac.agent.train()
        self.mac.init_hidden(X_state.shape[0])
        x_hidden = self.mac.get_hidden_states()
        X_inputs = torch.cat([X_observations, X_action_before_onehot, agent_id], dim=-1)
        agent_outs, _ = self.mac.agent(X_inputs, x_hidden)
        #agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(agent_outs, dim=2, index=X_actions)
        #state, action是第一步的轨迹
        with th.no_grad():
            self.target_mac.agent.train()
            self.target_mac.init_hidden(state.shape[0])
            #hidden = self.target_mac.get_hidden_states()
            current_obs = observation
            avail_actions_onehot = th.ones(*(action_onehot.shape), device=hidden.device)
            cond_start = torch.cat([current_obs, action_before_onehot, agent_id, hidden], dim=-1)
            rewards_tj = torch.zeros(*(action_onehot.shape[0], 1),
                                     device= hidden.device)
            for i_t_h in range(self.args.DWN_T_Episode):
                samples_tr = self.mac.agent.perform_DWN(cond_start)
                stat_tr = samples_tr[..., : self.args.state_shape]
                rewards_tj += self.args.gamma**(i_t_h) * \
                              self.global_reward_pred(samples_tr[..., self.args.state_shape: self.args.state_shape + 1].reshape(
                                  samples_tr.shape[0], -1
                              ))
                next_obs_th = samples_tr[..., self.args.state_shape + 1:]
                # 这里需要创建actor的输入
                inputs = torch.cat([current_obs, action_before_onehot, agent_id], dim=-1)
                action_befor_q, hidden = self.target_mac.agent(inputs, hidden)
                #action_befor_q = th.nn.functional.softmax(action_befor_q, dim=-1)
                chosen_action_before = self.target_mac.action_selector. \
                    select_action(action_befor_q, avail_actions_onehot, t_env, test_mode=False)
                # action_befor需要one-hot形式
                chosen_action_before = chosen_action_before.unsqueeze(dim=-1)
                chosen_action_before_onehot = self.onehot.transform(chosen_action_before)
                current_obs = next_obs_th
                action_before_onehot = chosen_action_before_onehot
                cond_start = torch.cat([current_obs, action_before_onehot,
                                        agent_id, hidden], dim=-1)
        #计算Q^-_(s,h)
        with th.no_grad():
            self.target_mac.agent.train()
            #self.target_mac.init_hidden(state.shape[0])
            target_inputs = th.cat([current_obs, action_before_onehot, agent_id], dim=-1)
            target_hidden = self.target_mac.get_hidden_states()
            target_agent_outs, _ = self.target_mac.agent(target_inputs, target_hidden)
            #target_agent_outs = th.nn.functional.softmax(target_agent_outs, dim=-1)
            target_mac_outs_detach = target_agent_outs.clone().detach()
            target_cur_max_actions = target_mac_outs_detach.max(dim=2, keepdim=True)[1]
            target_max_qvals = th.gather(target_agent_outs, 2, target_cur_max_actions).squeeze(2)

        #print(chosen_action_qvals.shape)
        chosen_action_qvals = chosen_action_qvals.squeeze(dim=-1).unsqueeze(dim=1)
        target_max_qvals = target_max_qvals.squeeze(dim=-1).unsqueeze(dim=1)
        stat_tr = self.global_state_pred(stat_tr.reshape(
            stat_tr.shape[0], -1
        ))
        start_q = self.mixer(chosen_action_qvals, X_state)
        target_q = self.target_mixer(target_max_qvals, stat_tr)
        target_q = rewards_tj + self.args.gamma**(self.args.DWN_T_Episode) * target_q
        target_q = target_q.detach()

        td_error = (start_q - target_q)
        td_loss2 = 0.5*(td_error**2)
        if self.mac.action_selector.epsilon <= 0.051:
            self.epsilon_flag = True
        if self.epsilon_flag == False:
            td_loss2 *= (1-self.mac.action_selector.epsilon)
        loss = td_loss2.mean()

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        return loss

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, per_weight=None):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        
        # Calculate estimated Q-Values
        self.mac.agent.train()
        mac_out = []
        DWN_used_hidden_s = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            DWN_used_hidden_s.append(self.mac.get_hidden_states())
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        DWM_action_qvals = mac_out


        #1/0

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        chosen_action_qvals_ = chosen_action_qvals


        # Calculate the Q-Values necessary for the target
        DWN_used_hidden_s_target = []
        with th.no_grad():
            self.target_mac.agent.train()
            target_mac_out = []
            self.target_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                DWN_used_hidden_s_target.append(self.target_mac.get_hidden_states())
                target_agent_outs = self.target_mac.forward(batch, t=t)
                target_mac_out.append(target_agent_outs)

            # We don't need the first timesteps Q-Value estimate for calculating targets
            target_mac_out = th.stack(target_mac_out, dim=1)  # Concat across time

            # Max over target Q-Values/ Double q learning
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
            
            # Calculate n-step Q-Learning targets
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"])

            if getattr(self.args, 'q_lambda', False):
                qvals = th.gather(target_mac_out, 3, batch["actions"]).squeeze(3)
                qvals = self.target_mixer(qvals, batch["state"])

                targets = build_q_lambda_targets(rewards, terminated, mask, target_max_qvals, qvals,
                                    self.args.gamma, self.args.td_lambda)
            else:
                targets = build_td_lambda_targets(rewards, terminated, mask, target_max_qvals, 
                                                    self.args.n_agents, self.args.gamma, self.args.td_lambda)

        # Mixer
        chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])

        with th.no_grad():
            DWN_chose_action_qvals = chosen_action_qvals
            DWM_action_qvals_mix = []
            for index_D in range(DWM_action_qvals.shape[-1]):
                #print(DWM_action_qvals[:,:-1,:,index_D].shape)
                #1/0
                qval = self.mixer(DWM_action_qvals[:,:-1,:,index_D], batch["state"][:, :-1])
                DWM_action_qvals_mix.append(qval)
            DWM_action_qvals = torch.stack(DWM_action_qvals_mix, dim=-1)
            DWM_action_qvals = DWM_action_qvals.squeeze(dim=2)
            DWM_action_qvals_mean = DWM_action_qvals.mean(dim=-1).unsqueeze(dim=-1)
            DWM_W = DWN_chose_action_qvals - DWM_action_qvals_mean
            DWM_W = torch.clamp(DWM_W, min=0)[:,1:-1]

        td_error = (chosen_action_qvals - targets.detach())
        td_error2 = 0.5 * td_error.pow(2)

        mask = mask.expand_as(td_error2)
        masked_td_error = td_error2 * mask

        # important sampling for PER
        if self.use_per:
            per_weight = th.from_numpy(per_weight).unsqueeze(-1).to(device=self.device)
            masked_td_error = masked_td_error.sum(1) * per_weight

        loss = L_td = masked_td_error.sum() / mask.sum()


        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()
        ####训练World Model
        ###隐状态
        DWN_used_hidden_s = th.stack(DWN_used_hidden_s, dim=1)[:, 1:-2]
        DWN_used_hidden_s_target = th.stack(DWN_used_hidden_s_target, dim=1)[:, 1:-2]
        # 拼接condition
        DWN_used_hidden_s = copy.deepcopy(DWN_used_hidden_s.detach())
        # DWN_used_hidden_s = DWN_used_hidden_s.detach()
        DWN_observations = batch["obs"][:, 1:-2]
        DWN_actions = batch["actions_onehot"][:, :-3]
        DWN_s = batch["state"][:, 1:-2]
        DWN_s_loss = DWN_s
        DWN_r = batch["reward"][:, 1:-2]
        DWN_r_loss = DWN_r
        DWN_next_obs = batch["obs"][:, 2:-1]

        DWN_s = DWN_s.unsqueeze(2).expand(-1, -1, self.args.n_agents, -1)
        DWN_r = DWN_r.unsqueeze(2).expand(-1, -1, self.args.n_agents, -1)

        agents_id = th.eye(self.args.n_agents, device=batch.device)  # .unsqueeze(0).expand(bs, -1, -1)
        agents_id = agents_id.unsqueeze(0).expand(DWN_observations.shape[0], -1, -1)
        agents_id = agents_id.unsqueeze(1).expand(-1, DWN_observations.shape[1], -1, -1)

        DWN_condition = th.cat([DWN_observations, DWN_actions, agents_id, DWN_used_hidden_s], dim=-1)

        DWN_trajectory = th.cat([DWN_s, DWN_r, DWN_next_obs], dim=-1)
        loss_DWN, constru_tra = self.mac.train_DWN(DWN_trajectory, DWN_condition, DWM_W)
        construct_state = constru_tra[...,:self.args.state_shape].reshape(
            constru_tra.shape[0], constru_tra.shape[1],-1
        ).detach().cuda()
        construct_reward = constru_tra[..., self.args.state_shape: self.args.state_shape+1].reshape(
            constru_tra.shape[0], constru_tra.shape[1],-1
        ).detach().cuda()

        pred_state_cons = self.global_state_pred(construct_state)
        loss_state_pre = ((DWN_s_loss - pred_state_cons)**2).mean()
        self.optimiser_stat_pre.zero_grad()
        loss_state_pre.backward()
        #grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser_stat_pre.step()

        pred_reward_cons  = self.global_reward_pred(construct_reward)
        loss_reward_pre = ((DWN_r_loss-pred_reward_cons)**2).mean()
        self.optimiser_re_pre.zero_grad()
        loss_reward_pre.backward()
        # grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser_re_pre.step()
        ###在World Model中进行训练
        loss_tj = self.train_traj(batch["state"][:,1], batch["actions"][:,1],
                        batch["obs"][:,1], batch["actions_onehot"][:,1],
                        batch["actions_onehot"][:,0],
                   agents_id[:,0], t_env, DWN_used_hidden_s_target[:,1])

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss_td", L_td.item(), t_env)
            self.logger.log_stat("loss_tj", loss_tj.item(), t_env)
            self.logger.log_stat("loss_DWM", loss_DWN.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            self.logger.log_stat("predict_global_s", loss_state_pre.item(), t_env)
            self.logger.log_stat("predict_global_r", loss_reward_pre.item(), t_env )
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("DWM_increase_epsilon", 1-self.mac.action_selector.epsilon,  t_env)
            self.logger.log_stat("DWM_increase_flag", self.epsilon_flag,  t_env)
            self.log_stats_t = t_env
            
            # print estimated matrix

        # return info
        info = {}
        # calculate priority
        if self.use_per:
            if self.return_priority:
                info["td_errors_abs"] = rewards.sum(1).detach().to('cpu')
                # normalize to [0, 1]
                self.priority_max = max(th.max(info["td_errors_abs"]).item(), self.priority_max)
                self.priority_min = min(th.min(info["td_errors_abs"]).item(), self.priority_min)
                info["td_errors_abs"] = (info["td_errors_abs"] - self.priority_min) \
                                / (self.priority_max - self.priority_min + 1e-5)
            else:
                info["td_errors_abs"] = ((td_error.abs() * mask).sum(1) \
                                / th.sqrt(mask.sum(1))).detach().to('cpu')
        return info

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()
            
    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
