# agents/ppo_manual/agent.py
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import numpy as np
from pathlib import Path

from agents.common import config
from agents.ppo_manual.model import ActorCriticNetwork  # Korrekt import her
from agents.common.utils import preprocess_state, format_action


class PPOAgent:
    def __init__(self, agent_name="PPO_Agent_Default"):
        self.network = ActorCriticNetwork().to(config.DEVICE)
        self.optimizer = optim.Adam(self.network.parameters(), lr=config.LEARNING_RATE_PPO if hasattr(config,
                                                                                                      'LEARNING_RATE_PPO') else config.LEARNING_RATE)
        self.player_id = None
        self.agent_name = agent_name

        self.map_pixel_buffer = []
        self.worm_vector_buffer = []
        self.action_indices_buffer = []
        self.action_params_raw_buffer = []
        self.log_probs_buffer = []
        self.values_buffer = []
        self.rewards_buffer = []
        self.dones_buffer = []
        self.entropies_buffer = []
        self.batch_count = 0

    def set_player_id(self, player_id: int):
        self.player_id = player_id

    def select_action(self, current_game_state_json: dict):
        if self.player_id is None:
            return {"action": "stand"}

        my_worm_json_id = self.player_id - 1
        is_my_worm_alive = any(
            w['id'] == my_worm_json_id and w['health'] > 0
            for w in current_game_state_json.get('worms', [])
        )
        if not is_my_worm_alive:
            self.map_pixel_buffer.append(
                torch.zeros(1, config.CNN_INPUT_CHANNELS, config.MAP_HEIGHT, config.MAP_WIDTH, device=config.DEVICE))
            self.worm_vector_buffer.append(torch.zeros(1, config.WORM_VECTOR_DIM, device=config.DEVICE))
            self.values_buffer.append(torch.tensor(0.0, device=config.DEVICE, dtype=torch.float32))
            self.log_probs_buffer.append(torch.tensor(0.0, device=config.DEVICE, dtype=torch.float32))
            self.entropies_buffer.append(torch.tensor(0.0, device=config.DEVICE, dtype=torch.float32))
            self.action_indices_buffer.append(-1)
            self.action_params_raw_buffer.append({})
            return {"action": "stand"}

        map_tensor, worm_vector_tensor = preprocess_state(current_game_state_json, self.player_id)
        self.map_pixel_buffer.append(map_tensor)
        self.worm_vector_buffer.append(worm_vector_tensor)

        with torch.no_grad():
            actor_outputs, state_value_tensor = self.network(map_tensor, worm_vector_tensor)

        self.values_buffer.append(state_value_tensor.squeeze())

        action_type_probs = actor_outputs['action_type_probs']
        action_type_dist = Categorical(action_type_probs)
        network_action_idx_tensor = action_type_dist.sample()
        network_action_idx_item = network_action_idx_tensor.item()
        self.action_indices_buffer.append(network_action_idx_item)

        log_prob_action_type = action_type_dist.log_prob(network_action_idx_tensor)
        entropy_action_type = action_type_dist.entropy()

        step_log_probs = [log_prob_action_type]
        step_entropies = [entropy_action_type]
        params_for_formatting = {}
        raw_params_for_buffer = {}
        chosen_network_action_name = config.NETWORK_ACTION_ORDER[network_action_idx_item]

        if chosen_network_action_name == 'walk':
            walk_dx_probs = actor_outputs['walk_dx_probs']
            walk_dx_dist = Categorical(walk_dx_probs)
            walk_dx_bin_idx_tensor = walk_dx_dist.sample()
            params_for_formatting['walk_dx_bin_idx'] = walk_dx_bin_idx_tensor.item()
            raw_params_for_buffer['walk_dx_bin_idx'] = walk_dx_bin_idx_tensor
            step_log_probs.append(walk_dx_dist.log_prob(walk_dx_bin_idx_tensor))
            step_entropies.append(walk_dx_dist.entropy())

        elif chosen_network_action_name == 'attack_kick':
            pass

        elif chosen_network_action_name == 'attack_bazooka':
            angle_mean, angle_std = actor_outputs['bazooka_params']

            # Sikre skalar for mean og std for Normal (fjerner batch dim hvis den er der)
            current_angle_mean = angle_mean.squeeze(0) if angle_mean.ndim > 1 and angle_mean.shape[
                0] == 1 else angle_mean.squeeze()
            current_angle_std = angle_std.squeeze(0) if angle_std.ndim > 1 and angle_std.shape[
                0] == 1 else angle_std.squeeze()

            # Klem std for å unngå for små/store verdier som kan gi NaN i log_prob
            current_angle_std = torch.clamp(current_angle_std, 1e-5, 10.0)

            dist = Normal(current_angle_mean, current_angle_std)
            angle_val_tensor = dist.sample()
            params_for_formatting['bazooka_angle_val'] = angle_val_tensor.item()
            raw_params_for_buffer['bazooka_angle_val'] = angle_val_tensor
            step_log_probs.append(dist.log_prob(angle_val_tensor))
            step_entropies.append(dist.entropy())

        elif chosen_network_action_name == 'attack_grenade':
            grenade_dx_probs = actor_outputs['grenade_dx_probs']
            grenade_dx_dist = Categorical(grenade_dx_probs)
            grenade_dx_bin_idx_tensor = grenade_dx_dist.sample()
            params_for_formatting['grenade_dx_bin_idx'] = grenade_dx_bin_idx_tensor.item()
            raw_params_for_buffer['grenade_dx_bin_idx'] = grenade_dx_bin_idx_tensor
            step_log_probs.append(grenade_dx_dist.log_prob(grenade_dx_bin_idx_tensor))
            step_entropies.append(grenade_dx_dist.entropy())

        self.action_params_raw_buffer.append(raw_params_for_buffer)

        squeezed_log_probs = [lp.squeeze() for lp in step_log_probs if lp is not None and hasattr(lp, 'squeeze')]
        squeezed_entropies = [e.squeeze() for e in step_entropies if e is not None and hasattr(e, 'squeeze')]

        valid_log_probs = [lp for lp in squeezed_log_probs if lp.ndim == 0]
        valid_entropies = [e for e in squeezed_entropies if e.ndim == 0]

        if valid_log_probs:
            stacked_log_probs = torch.stack(valid_log_probs)
            self.log_probs_buffer.append(stacked_log_probs.sum())
        else:
            self.log_probs_buffer.append(torch.tensor(0.0, device=config.DEVICE, dtype=torch.float32))

        if valid_entropies:
            stacked_entropies = torch.stack(valid_entropies)
            self.entropies_buffer.append(stacked_entropies.sum())
        else:
            self.entropies_buffer.append(torch.tensor(0.0, device=config.DEVICE, dtype=torch.float32))

        action_json_to_send = format_action(network_action_idx_item, params_for_formatting)
        return action_json_to_send

    def store_reward_and_done(self, reward: float, done: bool):
        self.rewards_buffer.append(reward)
        self.dones_buffer.append(done)
        self.batch_count += 1

    def _compute_gae_and_returns(self, next_value_tensor_no_grad):
        rewards_np = np.array(self.rewards_buffer, dtype=np.float32)

        # Sikre at values_buffer kun inneholder tensorer før stack
        valid_values = [v for v in self.values_buffer if isinstance(v, torch.Tensor)]
        if not valid_values:  # Hvis ingen gyldige verdier (f.eks. alle var døde ormer)
            # Returner null-tensorer med forventet form hvis rewards_np ikke er tom
            if rewards_np.size > 0:
                return torch.zeros_like(torch.tensor(rewards_np, device=config.DEVICE), dtype=torch.float32), \
                    torch.zeros_like(torch.tensor(rewards_np, device=config.DEVICE), dtype=torch.float32)
            else:  # Hvis ingen rewards heller, returner tomme tensorer
                return torch.empty(0, device=config.DEVICE, dtype=torch.float32), \
                    torch.empty(0, device=config.DEVICE, dtype=torch.float32)

        values_stacked = torch.stack(valid_values)
        values_np = values_stacked.detach().cpu().numpy()
        dones_np = np.array(self.dones_buffer, dtype=np.float32)

        advantages = np.zeros_like(rewards_np)
        last_gae_lam = 0.0  # Initialiser som float
        num_steps = len(rewards_np)

        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_val = next_value_tensor_no_grad.detach().cpu().item()
                next_non_terminal = 1.0 - dones_np[t]
            else:
                next_val = values_np[t + 1]
                next_non_terminal = 1.0 - dones_np[t + 1]

            delta = rewards_np[t] + config.GAMMA_PPO * next_val * next_non_terminal - values_np[t]
            advantages[
                t] = last_gae_lam = delta + config.GAMMA_PPO * config.GAE_LAMBDA_PPO * next_non_terminal * last_gae_lam

        returns_np = advantages + values_np
        return torch.tensor(advantages, device=config.DEVICE, dtype=torch.float32), \
            torch.tensor(returns_np, device=config.DEVICE, dtype=torch.float32)

    def learn(self, next_game_state_json_for_bootstrap: dict | None):
        if self.batch_count < (config.PPO_BATCH_SIZE if hasattr(config, 'PPO_BATCH_SIZE') else 128):
            return None, None, None

        R_bootstrap_next_state = torch.tensor(0.0, device=config.DEVICE, dtype=torch.float32)
        if not self.dones_buffer[-1]:
            if next_game_state_json_for_bootstrap and self.player_id is not None:
                my_worm_json_id = self.player_id - 1
                is_my_worm_alive_in_next_state = any(
                    w['id'] == my_worm_json_id and w['health'] > 0
                    for w in next_game_state_json_for_bootstrap.get('worms', [])
                )
                if is_my_worm_alive_in_next_state:
                    if isinstance(next_game_state_json_for_bootstrap, dict) and \
                            next_game_state_json_for_bootstrap.get("map") and next_game_state_json_for_bootstrap.get(
                        "worms"):
                        next_map, next_worm = preprocess_state(next_game_state_json_for_bootstrap, self.player_id)
                        with torch.no_grad():
                            _, R_bootstrap_next_state_tensor = self.network(next_map, next_worm)
                            R_bootstrap_next_state = R_bootstrap_next_state_tensor.squeeze()

        advantages, returns = self._compute_gae_and_returns(R_bootstrap_next_state)

        # Sjekk om advantages/returns er tomme (kan skje hvis _compute_gae_and_returns returnerer tomme)
        if advantages.nelement() == 0 or returns.nelement() == 0:
            self.clear_buffers_and_count()
            return 0.0, 0.0, 0.0

        old_log_probs_tensor = torch.stack(self.log_probs_buffer).detach()

        map_pixel_batch = torch.cat(self.map_pixel_buffer, dim=0)
        worm_vector_batch = torch.cat(self.worm_vector_buffer, dim=0)

        total_policy_loss_epoch_sum = 0
        total_value_loss_epoch_sum = 0
        total_entropy_loss_epoch_sum = 0

        num_samples_in_batch = len(self.rewards_buffer)
        batch_indices = np.arange(num_samples_in_batch)

        for _ in range(config.PPO_EPOCHS if hasattr(config, 'PPO_EPOCHS') else 4):
            np.random.shuffle(batch_indices)

            shuffled_maps = map_pixel_batch[batch_indices]
            shuffled_worms = worm_vector_batch[batch_indices]

            actor_outputs_new, values_new_tensor = self.network(shuffled_maps, shuffled_worms)
            shuffled_values_new = values_new_tensor.squeeze()

            new_log_probs_parts_epoch = []
            new_entropies_parts_epoch = []

            for i_shuffled, original_idx in enumerate(batch_indices):
                action_type_idx_orig = self.action_indices_buffer[original_idx]

                if action_type_idx_orig == -1:
                    new_log_probs_parts_epoch.append(torch.tensor(0.0, device=config.DEVICE))
                    new_entropies_parts_epoch.append(torch.tensor(0.0, device=config.DEVICE))
                    continue

                # Hent output for det nåværende shufflet samplet
                current_actor_output_for_sample = {
                    'action_type_probs': actor_outputs_new['action_type_probs'][i_shuffled],
                    'walk_dx_probs': actor_outputs_new['walk_dx_probs'][
                        i_shuffled] if 'walk_dx_probs' in actor_outputs_new and actor_outputs_new[
                        'walk_dx_probs'] is not None else None,
                    'bazooka_params': (actor_outputs_new['bazooka_params'][0][i_shuffled],
                                       actor_outputs_new['bazooka_params'][1][i_shuffled]) if actor_outputs_new[
                                                                                                  'bazooka_params'] and
                                                                                              actor_outputs_new[
                                                                                                  'bazooka_params'][
                                                                                                  0] is not None else (
                    None, None),
                    'grenade_dx_probs': actor_outputs_new['grenade_dx_probs'][
                        i_shuffled] if 'grenade_dx_probs' in actor_outputs_new and actor_outputs_new[
                        'grenade_dx_probs'] is not None else None,
                }

                dist_action_type = Categorical(current_actor_output_for_sample['action_type_probs'])
                log_p_action_type = dist_action_type.log_prob(torch.tensor(action_type_idx_orig, device=config.DEVICE))
                entropy_action_type = dist_action_type.entropy()

                current_log_probs_parts = [log_p_action_type]
                current_entropies_parts = [entropy_action_type]

                action_name = config.NETWORK_ACTION_ORDER[action_type_idx_orig]
                raw_params_this_step_orig = self.action_params_raw_buffer[original_idx]

                if action_name == 'walk' and current_actor_output_for_sample['walk_dx_probs'] is not None:
                    dist_walk = Categorical(current_actor_output_for_sample['walk_dx_probs'])
                    log_p_walk = dist_walk.log_prob(raw_params_this_step_orig['walk_dx_bin_idx'])
                    entropy_walk = dist_walk.entropy()
                    current_log_probs_parts.append(log_p_walk)
                    current_entropies_parts.append(entropy_walk)
                elif action_name == 'attack_bazooka' and current_actor_output_for_sample['bazooka_params'][
                    0] is not None:
                    mean, std = current_actor_output_for_sample['bazooka_params']
                    # Klem std for å unngå for små/store verdier som kan gi NaN i log_prob
                    std = torch.clamp(std, 1e-5, 10.0)
                    dist_bzk = Normal(mean, std)
                    log_p_bzk = dist_bzk.log_prob(raw_params_this_step_orig['bazooka_angle_val'])
                    entropy_bzk = dist_bzk.entropy()
                    current_log_probs_parts.append(log_p_bzk)
                    current_entropies_parts.append(entropy_bzk)
                elif action_name == 'attack_grenade' and current_actor_output_for_sample[
                    'grenade_dx_probs'] is not None:
                    dist_grenade = Categorical(current_actor_output_for_sample['grenade_dx_probs'])
                    log_p_grenade = dist_grenade.log_prob(raw_params_this_step_orig['grenade_dx_bin_idx'])
                    entropy_grenade = dist_grenade.entropy()
                    current_log_probs_parts.append(log_p_grenade)
                    current_entropies_parts.append(entropy_grenade)

                valid_log_p = [lp for lp in current_log_probs_parts if lp is not None]
                valid_ent = [e for e in current_entropies_parts if e is not None]

                new_log_probs_parts_epoch.append(
                    torch.stack(valid_log_p).sum() if valid_log_p else torch.tensor(0.0, device=config.DEVICE))
                new_entropies_parts_epoch.append(
                    torch.stack(valid_ent).sum() if valid_ent else torch.tensor(0.0, device=config.DEVICE))

            active_mask_orig = torch.tensor([idx != -1 for idx in self.action_indices_buffer], device=config.DEVICE,
                                            dtype=torch.bool)
            shuffled_active_mask = active_mask_orig[batch_indices]

            if not torch.any(shuffled_active_mask):
                continue

            final_new_log_probs = torch.stack(new_log_probs_parts_epoch)[shuffled_active_mask]
            final_new_entropies = torch.stack(new_entropies_parts_epoch)[shuffled_active_mask]
            final_shuffled_values_new = shuffled_values_new[shuffled_active_mask]

            final_shuffled_old_log_probs = old_log_probs_tensor[batch_indices][shuffled_active_mask]
            final_shuffled_advantages = advantages[batch_indices][shuffled_active_mask]
            final_shuffled_returns = returns[batch_indices][shuffled_active_mask]

            if final_shuffled_old_log_probs.nelement() == 0:
                continue

            ratios = torch.exp(final_new_log_probs - final_shuffled_old_log_probs)

            surr1 = ratios * final_shuffled_advantages
            surr2 = torch.clamp(ratios, 1 - config.PPO_CLIP_EPSILON,
                                1 + config.PPO_CLIP_EPSILON) * final_shuffled_advantages

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(final_shuffled_values_new, final_shuffled_returns)
            entropy_loss = -final_new_entropies.mean()

            current_total_loss = policy_loss + \
                                 (config.VALUE_LOSS_COEF_PPO if hasattr(config,
                                                                        'VALUE_LOSS_COEF_PPO') else config.VALUE_LOSS_COEF) * value_loss + \
                                 (config.ENTROPY_COEF_PPO if hasattr(config,
                                                                     'ENTROPY_COEF_PPO') else config.ENTROPY_COEF) * entropy_loss

            self.optimizer.zero_grad()
            current_total_loss.backward()
            grad_norm_key = 'MAX_GRAD_NORM_PPO' if hasattr(config, 'MAX_GRAD_NORM_PPO') else 'MAX_GRAD_NORM'
            if hasattr(config, grad_norm_key) and getattr(config, grad_norm_key) is not None:
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), getattr(config, grad_norm_key))
            self.optimizer.step()

            total_policy_loss_epoch_sum += policy_loss.item()
            total_value_loss_epoch_sum += value_loss.item()
            total_entropy_loss_epoch_sum += entropy_loss.item()

        ppo_epochs_val = config.PPO_EPOCHS if hasattr(config, 'PPO_EPOCHS') else 4
        avg_policy_loss = total_policy_loss_epoch_sum / ppo_epochs_val if ppo_epochs_val > 0 else 0
        avg_value_loss = total_value_loss_epoch_sum / ppo_epochs_val if ppo_epochs_val > 0 else 0
        avg_entropy_loss = total_entropy_loss_epoch_sum / ppo_epochs_val if ppo_epochs_val > 0 else 0

        self.clear_buffers_and_count()
        return avg_policy_loss, avg_value_loss, avg_entropy_loss

    def clear_buffers_and_count(self):
        self.map_pixel_buffer = []
        self.worm_vector_buffer = []
        self.action_indices_buffer = []
        self.action_params_raw_buffer = []
        self.log_probs_buffer = []
        self.values_buffer = []
        self.rewards_buffer = []
        self.dones_buffer = []
        self.entropies_buffer = []
        self.batch_count = 0

    def save_model(self, path_str: str):
        path = Path(path_str)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            torch.save(self.network.state_dict(), path)
        except Exception as e:
            print(f"[{self.agent_name}] Kunne ikke lagre PPO-modell til {path}: {e}")

    def load_model(self, path_str: str):
        path = Path(path_str)
        if not path.exists():
            print(f"[{self.agent_name}] Ingen PPO-modell funnet på {path}, starter med ny.")
            return
        try:
            self.network.load_state_dict(torch.load(path, map_location=torch.device(config.DEVICE)))
            self.network.eval()
            print(f"[{self.agent_name}] PPO-Modell lastet fra {path}")
        except Exception as e:
            print(f"[{self.agent_name}] Kunne ikke laste PPO-modell fra {path}: {e}. Bruker ny.")