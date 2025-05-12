# agents/a2c_manual/agent.py
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import numpy as np

from . import config
from .model import ActorCriticNetwork
from .utils import preprocess_state, format_action


class A2CAgent:
    def __init__(self):
        self.network = ActorCriticNetwork().to(config.DEVICE)
        self.optimizer = optim.Adam(self.network.parameters(), lr=config.LEARNING_RATE)
        self.player_id = None

        self.log_probs_buffer = []
        self.values_buffer = []
        self.rewards_buffer = []
        self.entropies_buffer = []

    def set_player_id(self, player_id: int):
        self.player_id = player_id
        print(f"Agentens player_id er satt til: {self.player_id}")

    def select_action(self, current_game_state_json: dict):
        if self.player_id is None:
            print("FEIL select_action: Agent player_id er ikke satt. Returnerer 'stand'.")
            return {"action": "stand"}

        map_tensor, worm_vector_tensor = preprocess_state(current_game_state_json, self.player_id)

        actor_outputs, state_value = self.network(map_tensor, worm_vector_tensor)

        squeezed_state_value = state_value.squeeze()
        self.values_buffer.append(squeezed_state_value)

        action_type_probs = actor_outputs['action_type_probs']
        action_type_dist = Categorical(action_type_probs)
        network_action_idx_tensor = action_type_dist.sample()  # Dette er en tensor, f.eks. tensor([2])
        network_action_idx_item = network_action_idx_tensor.item()  # Få Python-int for indeksering

        # Gjør om til skalar tensor ved å bruke .squeeze() hvis den har batch-dim,
        # eller ved å hente ut elementet hvis det er en 1-element tensor.
        # For Categorical.log_prob(sample), hvis sample er en 0-dim tensor, er output 0-dim.
        # Hvis sample er 1-dim (som network_action_idx_tensor), er output 1-dim.
        log_prob_action = action_type_dist.log_prob(network_action_idx_tensor).squeeze()
        entropy_action = action_type_dist.entropy().squeeze()

        # print(f"DEBUG select_action: log_prob_action={log_prob_action}, shape={log_prob_action.shape}")
        # print(f"DEBUG select_action: entropy_action={entropy_action}, shape={entropy_action.shape}")

        step_log_probs = [log_prob_action]
        step_entropies = [entropy_action]

        chosen_network_action_name = config.NETWORK_ACTION_ORDER[network_action_idx_item]
        # print(f"DEBUG select_action: Chosen action_name: {chosen_network_action_name}")
        params_for_formatting = {}

        if chosen_network_action_name == 'walk':
            walk_dx_probs = actor_outputs['walk_dx_probs']
            walk_dx_dist = Categorical(walk_dx_probs)
            walk_dx_bin_idx_tensor = walk_dx_dist.sample()
            params_for_formatting['walk_dx_bin_idx'] = walk_dx_bin_idx_tensor.item()
            log_p = walk_dx_dist.log_prob(walk_dx_bin_idx_tensor).squeeze()
            ent_p = walk_dx_dist.entropy().squeeze()
            # print(f"DEBUG select_action (walk): log_p_param={log_p}, shape={log_p.shape}")
            step_log_probs.append(log_p)
            step_entropies.append(ent_p)

        elif chosen_network_action_name == 'attack_kick':
            mean, std = actor_outputs['kick_params']
            # Normal tar skalar mean/std hvis output fra FC er [batch, 1] og vi squeezer
            dist = Normal(mean.squeeze(), std.squeeze())
            force_val = dist.sample()  # force_val vil være skalar
            params_for_formatting['kick_force_val'] = force_val.item()
            log_p = dist.log_prob(force_val)  # log_prob av skalar er skalar
            ent_p = dist.entropy()  # entropi av dist med skalar params er skalar
            # print(f"DEBUG select_action (kick): log_p_param={log_p}, shape={log_p.shape}")
            step_log_probs.append(log_p)
            step_entropies.append(ent_p)

        elif chosen_network_action_name == 'attack_bazooka':
            angle_mean, angle_std = actor_outputs['bazooka_params']
            dist = Normal(angle_mean.squeeze(), angle_std.squeeze())
            angle_val = dist.sample()
            params_for_formatting['bazooka_angle_val'] = angle_val.item()
            log_p = dist.log_prob(angle_val)
            ent_p = dist.entropy()
            # print(f"DEBUG select_action (bazooka): log_p_param_angle={log_p}, shape={log_p.shape}")
            step_log_probs.append(log_p)
            step_entropies.append(ent_p)

        elif chosen_network_action_name == 'attack_grenade':
            angle_mean, angle_std, force_mean, force_std = actor_outputs['grenade_params']
            # Angle
            angle_dist = Normal(angle_mean.squeeze(), angle_std.squeeze())
            angle_val = angle_dist.sample()
            params_for_formatting['grenade_angle_val'] = angle_val.item()
            log_p_angle = angle_dist.log_prob(angle_val)
            ent_p_angle = angle_dist.entropy()
            # print(f"DEBUG select_action (grenade): log_p_param_angle={log_p_angle}, shape={log_p_angle.shape}")
            step_log_probs.append(log_p_angle)
            step_entropies.append(ent_p_angle)

            # Force
            force_dist = Normal(force_mean.squeeze(), force_std.squeeze())
            force_val = force_dist.sample()
            params_for_formatting['grenade_force_val'] = force_val.item()
            log_p_force = force_dist.log_prob(force_val)
            ent_p_force = force_dist.entropy()
            # print(f"DEBUG select_action (grenade): log_p_param_force={log_p_force}, shape={log_p_force.shape}")
            step_log_probs.append(log_p_force)
            step_entropies.append(ent_p_force)

        # Nå skal alle elementer i step_log_probs og step_entropies være skalar-tensorer (0-dim)
        # print(f"DEBUG select_action: step_log_probs before stack: {[lp.shape for lp in step_log_probs]}")
        try:
            # torch.stack lager en 1D tensor fra listen av skalarer
            stacked_log_probs = torch.stack(step_log_probs)
            summed_log_probs = stacked_log_probs.sum()  # Summerer 1D tensoren til en skalar
        except Exception as e:
            print(f"FEIL i select_action under stacking/summing av log_probs: {e}")
            print(f"  step_log_probs var: {step_log_probs}")
            summed_log_probs = torch.tensor(0.0, device=config.DEVICE, dtype=torch.float32)

        self.log_probs_buffer.append(summed_log_probs)

        # print(f"DEBUG select_action: step_entropies before stack: {[e.shape for e in step_entropies]}")
        try:
            stacked_entropies = torch.stack(step_entropies)
            summed_entropies = stacked_entropies.sum()
        except Exception as e:
            print(f"FEIL i select_action under stacking/summing av entropies: {e}")
            print(f"  step_entropies var: {step_entropies}")
            summed_entropies = torch.tensor(0.0, device=config.DEVICE, dtype=torch.float32)

        self.entropies_buffer.append(summed_entropies)

        action_json_to_send = format_action(network_action_idx_item, params_for_formatting)

        return action_json_to_send

    def store_reward(self, reward: float):
        self.rewards_buffer.append(reward)

    def learn(self, next_game_state_json: dict | None, done: bool):
        if not self.log_probs_buffer or not self.rewards_buffer or not self.values_buffer:
            # print(f"DEBUG learn: Buffers tomme. log_probs_buffer: {len(self.log_probs_buffer)}, rewards_buffer: {len(self.rewards_buffer)}, values_buffer: {len(self.values_buffer)}")
            self.clear_buffers()
            return None, None, None

            # print(f"DEBUG learn: Antall steg i buffer: log_probs={len(self.log_probs_buffer)}, values={len(self.values_buffer)}, rewards={len(self.rewards_buffer)}, entropies={len(self.entropies_buffer)}")

        R_bootstrap = torch.tensor(0.0, device=config.DEVICE, dtype=torch.float32)
        if not done:
            if next_game_state_json and self.player_id is not None:
                if isinstance(next_game_state_json, dict) and next_game_state_json.get(
                        "map") and next_game_state_json.get("worms"):
                    next_map, next_worm = preprocess_state(next_game_state_json, self.player_id)
                    with torch.no_grad():
                        _, next_value = self.network(next_map, next_worm)
                        R_bootstrap = next_value.squeeze()  # Blir skalar
                else:
                    # print(f"WARN learn: next_game_state_json er ugyldig for bootstrapping ved done=False: {str(next_game_state_json)[:200]}. Bruker V(s_next)=0.")
                    pass  # R_bootstrap forblir 0.0

        returns = []
        R_discounted = R_bootstrap
        for r_idx in range(len(self.rewards_buffer) - 1, -1, -1):
            reward_val = torch.tensor(self.rewards_buffer[r_idx], device=config.DEVICE, dtype=torch.float32)
            R_discounted = reward_val + config.GAMMA * R_discounted
            returns.insert(0, R_discounted)

        if not returns:
            # print("FEIL learn: `returns` listen er tom etter beregning.")
            self.clear_buffers()
            return None, None, None

        try:
            # Alle buffere skal nå inneholde skalar-tensorer.
            # torch.stack vil lage 1D tensorer.
            returns_tensor = torch.stack(returns).detach()
            values_tensor = torch.stack(self.values_buffer)
            log_probs_tensor = torch.stack(self.log_probs_buffer)
            entropies_tensor = torch.stack(self.entropies_buffer)
        except RuntimeError as e_stack:
            print(f"FEIL learn: RuntimeError under torch.stack: {e_stack}")
            print(
                f"  Antall elementer: returns={len(returns)}, values_buffer={len(self.values_buffer)}, log_probs_buffer={len(self.log_probs_buffer)}, entropies_buffer={len(self.entropies_buffer)}")
            # Detaljert sjekk av formen på elementene i bufferne hvis feilen vedvarer
            if len(self.values_buffer) > 0: print(f"  values_buffer[0] shape: {self.values_buffer[0].shape}")
            if len(self.log_probs_buffer) > 0: print(f"  log_probs_buffer[0] shape: {self.log_probs_buffer[0].shape}")
            if len(self.entropies_buffer) > 0: print(f"  entropies_buffer[0] shape: {self.entropies_buffer[0].shape}")
            self.clear_buffers()
            return None, None, None

        returns_tensor = returns_tensor.to(config.DEVICE)
        values_tensor = values_tensor.to(config.DEVICE)
        log_probs_tensor = log_probs_tensor.to(config.DEVICE)
        entropies_tensor = entropies_tensor.to(config.DEVICE)

        # print(f"DEBUG learn: shapes etter stack: R={returns_tensor.shape}, V={values_tensor.shape}, LP={log_probs_tensor.shape}, E={entropies_tensor.shape}")

        if values_tensor.numel() == 0 or returns_tensor.numel() == 0 or log_probs_tensor.numel() == 0:
            self.clear_buffers()
            return None, None, None

        # Sikre at tensorer er minst 1D for .mean() og F.mse_loss
        # Dette er normalt håndtert av torch.stack hvis inputlistene ikke er tomme.
        # Men en ekstra sjekk hvis en tensor ble 0-dim av en eller annen grunn (f.eks. stack av én 0-dim tensor).
        if values_tensor.ndim == 0: values_tensor = values_tensor.unsqueeze(0)
        if returns_tensor.ndim == 0: returns_tensor = returns_tensor.unsqueeze(0)
        if log_probs_tensor.ndim == 0: log_probs_tensor = log_probs_tensor.unsqueeze(0)
        if entropies_tensor.ndim == 0: entropies_tensor = entropies_tensor.unsqueeze(0)

        advantages = returns_tensor - values_tensor
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        elif advantages.numel() == 0:
            self.clear_buffers()
            return None, None, None

        policy_loss = (-log_probs_tensor * advantages.detach()).mean()
        value_loss = F.mse_loss(values_tensor, returns_tensor)
        entropy_loss = -entropies_tensor.mean()

        total_loss = policy_loss + \
                     config.VALUE_LOSS_COEF * value_loss + \
                     config.ENTROPY_COEF * entropy_loss

        # print(f"DEBUG learn: policy_loss={policy_loss.item():.4f}, value_loss={value_loss.item():.4f}, entropy_loss={entropy_loss.item():.4f}, total_loss={total_loss.item():.4f}")

        self.optimizer.zero_grad()
        total_loss.backward()
        if hasattr(config, 'MAX_GRAD_NORM') and config.MAX_GRAD_NORM is not None:
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), config.MAX_GRAD_NORM)
        self.optimizer.step()

        self.clear_buffers()

        return policy_loss.item(), value_loss.item(), entropy_loss.item()

    def clear_buffers(self):
        self.log_probs_buffer = []
        self.values_buffer = []
        self.rewards_buffer = []
        self.entropies_buffer = []