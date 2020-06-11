import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer, QMixerQuantilie
import torch as th
from torch.optim import RMSprop
import torch.nn.functional as F
import numpy as np
from utils.rl_utils import build_td_lambda_targets


class SLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.n_actions_levin = args.n_actions

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())

            if not args.SubAVG_Mixer_flag:
                self.target_mixer = copy.deepcopy(self.mixer)

            elif args.mixer == "qmix":
                self.target_mixer_list = []
                for i in range(self.args.SubAVG_Mixer_K):
                    self.target_mixer_list.append(copy.deepcopy(self.mixer))
                self.levin_iter_target_mixer_update = 0


        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        if not self.args.SubAVG_Agent_flag:
            self.target_mac = copy.deepcopy(mac)
        else:
            self.target_mac_list = []
            for i in range(self.args.SubAVG_Agent_K):
                self.target_mac_list.append(copy.deepcopy(mac))
            self.levin_iter_target_update = 0

        self.log_stats_t = -self.args.learner_log_interval - 1

        # ====== levin =====
        self.number = 0

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, epsilon_levin=None):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):                           
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time         

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  
                                                                        
        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        if not self.args.SubAVG_Agent_flag:
            self.target_mac.init_hidden(batch.batch_size)
        else:
            for i in range(self.args.SubAVG_Agent_K):
                self.target_mac_list[i].init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            if not self.args.SubAVG_Agent_flag:
                target_agent_outs = self.target_mac.forward(batch, t=t)   
            # exp:使用 average DQN的target_mac
            else:
                target_agent_outs = 0
                
                self.target_agent_out_list = []
                for i in range(self.args.SubAVG_Agent_K):
                    target_agent_out = self.target_mac_list[i].forward(batch, t=t)
                    target_agent_outs = target_agent_outs + target_agent_out
                    if self.args.SubAVG_Agent_flag_select:
                        self.target_agent_out_list.append(target_agent_out)
                target_agent_outs = target_agent_outs / self.args.SubAVG_Agent_K
                if self.args.SubAVG_Agent_flag_select:
                    if self.args.SubAVG_Agent_name_select_replacement == 'mean':    
                        target_out_select_sum = 0
                        for i in range(self.args.SubAVG_Agent_K):
                            if self.args.SubAVG_Agent_flag_select > 0:
                                target_out_select = th.where(self.target_agent_out_list[i] < target_agent_outs, target_agent_outs, self.target_agent_out_list[i])
                            else:
                                target_out_select = th.where(self.target_agent_out_list[i] > target_agent_outs, target_agent_outs, self.target_agent_out_list[i])
                            target_out_select_sum = target_out_select_sum + target_out_select
                        target_agent_outs = target_out_select_sum / self.args.SubAVG_Agent_K
                    elif self.args.SubAVG_Agent_name_select_replacement == 'zero':    
                        target_out_select_sum = 0
                        target_select_bool_sum = 0
                        for i in range(self.args.SubAVG_Agent_K):
                            if self.args.SubAVG_Agent_flag_select > 0:
                                target_select_bool = (self.target_agent_out_list[i] > target_agent_outs).float()
                                target_out_select = th.where(self.target_agent_out_list[i] > target_agent_outs, self.target_agent_out_list[i], th.full_like(target_agent_outs,0))
                            else:
                                target_select_bool = (self.target_agent_out_list[i] < target_agent_outs).float()
                                target_out_select = th.where(self.target_agent_out_list[i] < target_agent_outs, self.target_agent_out_list[i], th.full_like(target_agent_outs,0))
                            target_select_bool_sum = target_select_bool_sum + target_select_bool
                            target_out_select_sum = target_out_select_sum + target_out_select
                        if self.levin_iter_target_update < 2: 
                            pass # print("using average directly")
                        else:
                            target_agent_outs = target_out_select_sum / target_select_bool_sum
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out, dim=1) # Concat across time    

        # Mask out unavailable actions
        target_chosen_action_qvals = th.gather(target_mac_out, 3, batch['actions']).squeeze(-1)  

        # Mix
        if self.mixer is None:
            target_qvals = target_chosen_action_qvals
        else:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])  
            if not self.args.SubAVG_Mixer_flag:
                target_qvals = self.target_mixer(target_chosen_action_qvals, batch['state'])
            elif self.args.mixer == "qmix":
                target_max_qvals_sum = 0
                self.target_mixer_out_list = []
                for i in range(self.args.SubAVG_Mixer_K):
                    targe_mixer_out = self.target_mixer_list[i](target_chosen_action_qvals, batch['state'])
                    target_max_qvals_sum = target_max_qvals_sum + targe_mixer_out
                    if self.args.SubAVG_Mixer_flag_select:
                        self.target_mixer_out_list.append(targe_mixer_out)
                target_max_qvals = target_max_qvals_sum / self.args.SubAVG_Mixer_K

                # levin: mixer select
                if self.args.SubAVG_Mixer_flag_select:
                    if self.args.SubAVG_Mixer_name_select_replacement == 'mean':
                        target_mixer_select_sum = 0
                        for i in range(self.args.SubAVG_Mixer_K):
                            if self.args.SubAVG_Mixer_flag_select > 0:
                                target_mixer_select = th.where(self.target_mixer_out_list[i] < target_max_qvals, target_max_qvals, self.target_mixer_out_list[i])
                            else:
                                target_mixer_select = th.where(self.target_mixer_out_list[i] > target_max_qvals, target_max_qvals, self.target_mixer_out_list[i])
                            target_mixer_select_sum = target_mixer_select_sum + target_mixer_select
                        target_max_qvals = target_mixer_select_sum / self.args.SubAVG_Mixer_K
                    elif self.args.SubAVG_Mixer_name_select_replacement == 'zero':
                        target_mixer_select_sum = 0
                        target_mixer_select_bool_sum = 0
                        for i in range(self.args.SubAVG_Mixer_K):
                            if self.args.SubAVG_Mixer_flag_select > 0:
                                target_mixer_select_bool = (self.target_mixer_out_list[i] > target_max_qvals).float()
                                target_mixer_select = th.where(self.target_mixer_out_list[i] > target_max_qvals, self.target_mixer_out_list[i], th.full_like(target_max_qvals,0))
                            else:
                                target_mixer_select_bool = (self.target_mixer_out_list[i] < target_max_qvals).float()
                                target_mixer_select = th.where(self.target_mixer_out_list[i] < target_max_qvals, self.target_mixer_out_list[i], th.full_like(target_max_qvals,0))
                            target_mixer_select_bool_sum = target_mixer_select_bool_sum + target_mixer_select_bool
                            target_mixer_select_sum = target_mixer_select_sum + target_mixer_select
                        if self.levin_iter_target_mixer_update < 2:   
                            pass # print("using average-mix directly")
                        else:
                            target_max_qvals = target_mixer_select_sum / target_mixer_select_bool_sum
                target_qvals = target_max_qvals
                
        if self.args.td_lambda <= 1 and self.args.td_lambda > 0:  
            targets = build_td_lambda_targets(rewards, terminated, mask, target_qvals,
                                              self.args.n_agents,
                                              self.args.gamma,
                                              self.args.td_lambda)  
        else:
            if self.args.td_lambda == 0:
                n = 1  # 1-step TD
            else:
                n = self.args.td_lambda

            targets = th.zeros_like(batch['reward'])
            targets += batch['reward']

            for i in range(1, n):
                targets[:, :-i] += (self.args.gamma ** i) * (1 - terminated[:, i - 1:]) * batch['reward'][:, i:]
            targets[:, :-n] += (self.args.gamma ** n) * (1 - terminated[:, n - 1:]) * target_qvals[:, n:]

            targets = targets[:, :-1]
        
        # Td-error
        td_error = (chosen_action_qvals - targets.detach())   

        mask = mask.expand_as(td_error) 
        
        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask
        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum() * 2



        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            # self.logger.log_stat("loss_levin", loss_levin.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env


    def _update_targets(self):
        if not self.args.SubAVG_Agent_flag:
            self.target_mac.load_state(self.mac)
        else:
            self.number = self.levin_iter_target_update % self.args.SubAVG_Agent_K
            self.target_mac_list[self.number].load_state(self.mac)
            self.levin_iter_target_update = self.levin_iter_target_update + 1

        if self.mixer is not None:
            if not self.args.SubAVG_Mixer_flag:
                self.target_mixer.load_state_dict(self.mixer.state_dict())
            elif self.args.mixer == "qmix":
                mixer_number = self.levin_iter_target_mixer_update % self.args.SubAVG_Mixer_K
                self.target_mixer_list[mixer_number].load_state_dict(self.mixer.state_dict())
                self.levin_iter_target_mixer_update = self.levin_iter_target_mixer_update + 1
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        if not self.args.SubAVG_Agent_flag:
            self.target_mac.cuda()
        else:
            for i in range(self.args.SubAVG_Agent_K):
                self.target_mac_list[i].cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            if not self.args.SubAVG_Mixer_flag:
                self.target_mixer.cuda()
            elif self.args.mixer == "qmix":
                for i in range(self.args.SubAVG_Mixer_K):
                    self.target_mixer_list[i].cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        if not self.args.SubAVG_Agent_flag:
            self.target_mac.load_models(path)
        else:
            for i in range(self.args.SubAVG_Agent_K):
                self.target_mac_list[i].load_models(path)

        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))