# agents/a2c_manual/agent.py
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import numpy as np
from pathlib import Path

from . import config
from .model import ActorCriticNetwork
from .utils import preprocess_state, format_action


class A2CAgent:
    def __init__(self, agent_name="A2C_Agent_Default"):
        self.network = ActorCriticNetwork().to(config.DEVICE)
        self.optimizer = optim.Adam(self.network.parameters(), lr=config.LEARNING_RATE)
        self.player_id = None  # Settes av main_a2c.py ved ASSIGN_ID
        self.agent_name = agent_name  # For logging og unike sjekkpunktfiler

        # Buffere for ett læringssteg (tømmes etter hver .learn())
        self.log_probs_buffer = []
        self.values_buffer = []
        self.rewards_buffer = []
        self.entropies_buffer = []

    def set_player_id(self, player_id: int):
        self.player_id = player_id
        # print(f"[{self.agent_name}] Player ID satt til: {self.player_id}")

    def select_action(self, current_game_state_json: dict):
        if self.player_id is None:
            # print(f"[{self.agent_name}] FEIL select_action: Player ID ikke satt. Sender 'stand'.")
            return {"action": "stand"}

        # Sjekk om vår orm er i live før vi gjør noe
        my_worm_json_id = self.player_id - 1
        is_my_worm_alive = any(
            w['id'] == my_worm_json_id and w['health'] > 0
            for w in current_game_state_json.get('worms', [])
        )
        if not is_my_worm_alive:
            # print(f"[{self.agent_name}] Info select_action: Min orm (P{self.player_id}) er ikke lenger i live. Sender 'stand'.")
            # Det er viktig å lagre noe i bufferne selv om vi sender stand, slik at learn() ikke krasjer
            # pga. ulik bufferlengde. En dummy-verdi (f.eks. for V(s)) er nok.
            # Siden vi ikke tar en reell handling, lagrer vi ikke log_prob eller entropi.
            # Value for en "død" state kan anses som 0.
            self.values_buffer.append(torch.tensor(0.0, device=config.DEVICE, dtype=torch.float32))
            return {"action": "stand"}

        map_tensor, worm_vector_tensor = preprocess_state(current_game_state_json, self.player_id)

        try:
            actor_outputs, state_value_tensor = self.network(map_tensor, worm_vector_tensor)
        except Exception as e:
            print(f"[{self.agent_name}] FEIL under network forward pass i select_action: {e}")
            # Fallback for å unngå krasj, og lagre dummy value
            self.values_buffer.append(torch.tensor(0.0, device=config.DEVICE, dtype=torch.float32))
            return {"action": "stand"}

        self.values_buffer.append(state_value_tensor.squeeze())  # Skalar tensor

        action_type_probs = actor_outputs['action_type_probs']
        action_type_dist = Categorical(action_type_probs)
        network_action_idx_tensor = action_type_dist.sample()  # 0-dim tensor
        network_action_idx_item = network_action_idx_tensor.item()

        log_prob_action_type = action_type_dist.log_prob(network_action_idx_tensor)  # Skalar
        entropy_action_type = action_type_dist.entropy()  # Skalar

        step_log_probs = [log_prob_action_type]
        step_entropies = [entropy_action_type]
        params_for_formatting = {}
        chosen_network_action_name = config.NETWORK_ACTION_ORDER[network_action_idx_item]

        if chosen_network_action_name == 'walk':
            walk_dx_probs = actor_outputs['walk_dx_probs']
            walk_dx_dist = Categorical(walk_dx_probs)
            walk_dx_bin_idx_tensor = walk_dx_dist.sample()
            params_for_formatting['walk_dx_bin_idx'] = walk_dx_bin_idx_tensor.item()
            step_log_probs.append(walk_dx_dist.log_prob(walk_dx_bin_idx_tensor))
            step_entropies.append(walk_dx_dist.entropy())

        elif chosen_network_action_name == 'attack_kick':
            # Ingen parametere å sample for kick iht. nye specs
            pass

        elif chosen_network_action_name == 'attack_bazooka':
            angle_mean, angle_std = actor_outputs['bazooka_params']
            dist = Normal(angle_mean.squeeze(), angle_std.squeeze())
            angle_val_tensor = dist.sample()
            params_for_formatting['bazooka_angle_val'] = angle_val_tensor.item()
            step_log_probs.append(dist.log_prob(angle_val_tensor))
            step_entropies.append(dist.entropy())

        elif chosen_network_action_name == 'attack_grenade':
            grenade_dx_probs = actor_outputs['grenade_dx_probs']
            grenade_dx_dist = Categorical(grenade_dx_probs)
            grenade_dx_bin_idx_tensor = grenade_dx_dist.sample()
            params_for_formatting['grenade_dx_bin_idx'] = grenade_dx_bin_idx_tensor.item()
            step_log_probs.append(grenade_dx_dist.log_prob(grenade_dx_bin_idx_tensor))
            step_entropies.append(grenade_dx_dist.entropy())

        try:
            # Sikre at alle elementer er skalar-tensorer før stack
            step_log_probs = [lp.squeeze() for lp in step_log_probs]
            step_entropies = [e.squeeze() for e in step_entropies]

            stacked_log_probs = torch.stack(step_log_probs)
            summed_log_probs = stacked_log_probs.sum()
            self.log_probs_buffer.append(summed_log_probs)

            stacked_entropies = torch.stack(step_entropies)
            summed_entropies = stacked_entropies.sum()
            self.entropies_buffer.append(summed_entropies)
        except Exception as e_stack:
            print(f"[{self.agent_name}] FEIL select_action stacking/summing: {e_stack}")
            # Fallback for å opprettholde buffer-integritet
            self.log_probs_buffer.append(torch.tensor(0.0, device=config.DEVICE, dtype=torch.float32))
            self.entropies_buffer.append(torch.tensor(0.0, device=config.DEVICE, dtype=torch.float32))

        action_json_to_send = format_action(network_action_idx_item, params_for_formatting)
        return action_json_to_send

    def store_reward(self, reward: float):
        self.rewards_buffer.append(reward)

    def learn(self, next_game_state_json: dict | None, done: bool):
        # Kritisk sjekk: Antall lagrede verdier må matche antall lagrede rewards.
        # Hver 'select_action' legger til i log_probs, values, entropies.
        # Hver 'store_reward' (som kalles etter en handling) legger til i rewards.
        # Så lengden på rewards_buffer skal være lik lengden på de andre bufferne
        # *før* vi legger til V(s_next) eller gjør noe med returns.

        num_rewards = len(self.rewards_buffer)
        consistent_buffers = (
                len(self.log_probs_buffer) == num_rewards and
                len(self.values_buffer) == num_rewards and  # values_buffer har V(s_0) ... V(s_T-1)
                len(self.entropies_buffer) == num_rewards
        )

        if not consistent_buffers or num_rewards == 0:
            # print(f"[{self.agent_name}] DEBUG learn: Ujevne eller tomme buffere. Tømmer og returnerer.")
            # print(f"  LP:{len(self.log_probs_buffer)} V:{len(self.values_buffer)} R:{len(self.rewards_buffer)} E:{len(self.entropies_buffer)}")
            self.clear_buffers()
            return None, None, None

        R_bootstrap = torch.tensor(0.0, device=config.DEVICE, dtype=torch.float32)
        if not done:
            if next_game_state_json and self.player_id is not None:
                my_worm_json_id = self.player_id - 1
                is_my_worm_alive_in_next_state = any(
                    w['id'] == my_worm_json_id and w['health'] > 0
                    for w in next_game_state_json.get('worms', [])
                )
                if is_my_worm_alive_in_next_state:
                    if isinstance(next_game_state_json, dict) and \
                            next_game_state_json.get("map") and next_game_state_json.get("worms"):
                        next_map, next_worm = preprocess_state(next_game_state_json, self.player_id)
                        with torch.no_grad():
                            _, next_value_tensor = self.network(next_map, next_worm)
                            R_bootstrap = next_value_tensor.squeeze()

        returns = []
        R_discounted = R_bootstrap
        for r_idx in range(num_rewards - 1, -1, -1):
            reward_val = torch.tensor(self.rewards_buffer[r_idx], device=config.DEVICE, dtype=torch.float32)
            R_discounted = reward_val + config.GAMMA * R_discounted
            returns.insert(0, R_discounted)

        if not returns:  # Bør ikke skje hvis num_rewards > 0
            self.clear_buffers()
            return None, None, None

        try:
            returns_tensor = torch.stack(returns).detach()
            values_tensor = torch.stack(self.values_buffer)  # values_buffer skal ha num_rewards elementer
            log_probs_tensor = torch.stack(self.log_probs_buffer)
            entropies_tensor = torch.stack(self.entropies_buffer)
        except RuntimeError as e_stack:
            print(f"[{self.agent_name}] FEIL learn (stack): {e_stack}")
            print(
                f"  R:{len(returns)}, V:{len(self.values_buffer)}, LP:{len(self.log_probs_buffer)}, E:{len(self.entropies_buffer)}")
            self.clear_buffers()
            return None, None, None

        # Nå SKAL alle tensorer ha samme lengde (num_rewards)
        if not (returns_tensor.shape[0] == values_tensor.shape[0] == \
                log_probs_tensor.shape[0] == entropies_tensor.shape[0] == num_rewards):
            print(f"[{self.agent_name}] KRITISK FEIL learn: Tensorstørrelser stemmer ikke etter stack!")
            print(
                f"  Shape R:{returns_tensor.shape}, V:{values_tensor.shape}, LP:{log_probs_tensor.shape}, E:{entropies_tensor.shape}, Expected:{num_rewards}")
            self.clear_buffers()
            return None, None, None

        advantages = returns_tensor - values_tensor
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        policy_loss = (-log_probs_tensor * advantages.detach()).mean()
        value_loss = F.mse_loss(values_tensor, returns_tensor)
        entropy_loss = -entropies_tensor.mean()

        total_loss = policy_loss + \
                     config.VALUE_LOSS_COEF * value_loss + \
                     config.ENTROPY_COEF * entropy_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        if hasattr(config, 'MAX_GRAD_NORM') and config.MAX_GRAD_NORM is not None:
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), config.MAX_GRAD_NORM)
        self.optimizer.step()

        # print(f"[{self.agent_name}] Lært. Losses - P:{policy_loss.item():.3f} V:{value_loss.item():.3f} E:{entropy_loss.item():.3f}")
        self.clear_buffers()
        return policy_loss.item(), value_loss.item(), entropy_loss.item()

    def clear_buffers(self):
        self.log_probs_buffer = []
        self.values_buffer = []
        self.rewards_buffer = []
        self.entropies_buffer = []

    def save_model(self, path_str: str):
        path = Path(path_str)
        path.parent.mkdir(parents=True, exist_ok=True)  # Sikre at mappen finnes
        try:
            torch.save(self.network.state_dict(), path)
            # print(f"[{self.agent_name}] Modell lagret til {path}")
        except Exception as e:
            print(f"[{self.agent_name}] Kunne ikke lagre modell til {path}: {e}")

    def load_model(self, path_str: str):
        path = Path(path_str)
        if not path.exists():
            print(f"[{self.agent_name}] Ingen modell funnet på {path}, starter med ny/tilfeldig initialisert modell.")
            return

        try:
            self.network.load_state_dict(torch.load(path, map_location=torch.device(config.DEVICE)))
            self.network.eval()  # Sett til evaluation mode etter lasting
            print(f"[{self.agent_name}] Modell lastet fra {path}")
        except Exception as e:
            print(f"[{self.agent_name}] Kunne ikke laste modell fra {path}: {e}. Bruker ny modell.")