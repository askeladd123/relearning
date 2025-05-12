# agents/a2c_manual/agent.py
import torch
import torch.optim as optim
from torch.distributions import Categorical, Normal
import numpy as np

from . import config
from .model import ActorCriticNetwork
from .utils import preprocess_state, format_action

class A2CAgent:
    def __init__(self):
        self.network = ActorCriticNetwork().to(config.DEVICE)
        self.optimizer = optim.Adam(self.network.parameters(), lr=config.LEARNING_RATE)

        self.log_probs = []  # For lagring av log-sannsynligheter for policy gradient
        self.values = []     # For lagring av state values fra critic
        self.rewards = []    # For lagring av mottatte rewards
        self.entropies = []  # For lagring av entropi for exploration bonus

    def select_action(self, environment_json):
        """ Velger en handling basert på state og returnerer action JSON + lagrer for læring """
        map_tensor, worm_vector_tensor = preprocess_state(environment_json)

        # Få outputs fra nettverket
        actor_outputs, state_value = self.network(map_tensor, worm_vector_tensor)
        self.values.append(state_value) # Lagre V(s) for critic loss

        # --- Action Type Selection ---
        action_probs = actor_outputs['action_probs']
        action_dist = Categorical(action_probs)
        action_type_idx = action_dist.sample()
        self.log_probs.append(action_dist.log_prob(action_type_idx))
        self.entropies.append(action_dist.entropy())

        action_name = config.ACTION_LIST[action_type_idx.item()]
        selected_params = {}

        # --- Parameter Selection (basert på valgt action type) ---
        total_entropy = action_dist.entropy() # Start med entropi fra action type

        if action_name == 'walk':
            walk_probs = actor_outputs['walk_amount_probs']
            walk_dist = Categorical(walk_probs)
            walk_amount_idx = walk_dist.sample()
            selected_params['walk_amount'] = walk_amount_idx.item()
            self.log_probs.append(walk_dist.log_prob(walk_amount_idx))
            total_entropy += walk_dist.entropy()

        elif action_name == 'kick':
            mean, std = actor_outputs['kick_params']
            kick_dist = Normal(mean, std)
            kick_force = kick_dist.sample()
            # Clip force til et fornuftig område? Må diskuteres med Ask.
            # kick_force = torch.clamp(kick_force, min_force, max_force)
            selected_params['kick_force'] = kick_force.item()
            self.log_probs.append(kick_dist.log_prob(kick_force).sum()) # Sum for multi-dim Normal? Her 1D
            total_entropy += kick_dist.entropy().sum()

        elif action_name == 'bazooka':
            angle_mean, angle_std, force_mean, force_std = actor_outputs['bazooka_params']
            # Angle
            angle_dist = Normal(angle_mean, angle_std)
            bazooka_angle = angle_dist.sample()
            # bazooka_angle = torch.clamp(bazooka_angle, min_angle, max_angle)
            selected_params['bazooka_angle'] = bazooka_angle.item()
            self.log_probs.append(angle_dist.log_prob(bazooka_angle).sum())
            total_entropy += angle_dist.entropy().sum()
            # Force
            force_dist = Normal(force_mean, force_std)
            bazooka_force = force_dist.sample()
            # bazooka_force = torch.clamp(bazooka_force, min_force, max_force)
            selected_params['bazooka_force'] = bazooka_force.item()
            self.log_probs.append(force_dist.log_prob(bazooka_force).sum())
            total_entropy += force_dist.entropy().sum()

        elif action_name == 'grenade':
            angle_mean, angle_std, force_mean, force_std = actor_outputs['grenade_params']
            # Angle
            angle_dist = Normal(angle_mean, angle_std)
            grenade_angle = angle_dist.sample()
            # grenade_angle = torch.clamp(grenade_angle, min_angle, max_angle)
            selected_params['grenade_angle'] = grenade_angle.item()
            self.log_probs.append(angle_dist.log_prob(grenade_angle).sum())
            total_entropy += angle_dist.entropy().sum()
            # Force
            force_dist = Normal(force_mean, force_std)
            grenade_force = force_dist.sample()
            # grenade_force = torch.clamp(grenade_force, min_force, max_force)
            selected_params['grenade_force'] = grenade_force.item()
            self.log_probs.append(force_dist.log_prob(grenade_force).sum())
            total_entropy += force_dist.entropy().sum()

        self.entropies.append(total_entropy) # Lagre total entropi for dette steget

        # Formater action til JSON
        action_json = format_action(action_type_idx.item(), selected_params)

        return action_json

    def store_reward(self, reward):
        """ Lagrer mottatt reward. """
        self.rewards.append(reward)

    def learn(self, next_environment_json, done):
        """ Utfører A2C læringssteget etter en episode eller et antall steg. """
        if not self.log_probs: # Ingen handlinger tatt ennå
            return

        # Beregn siste state value (V(s_T))
        # Hvis 'done', er verdien 0. Ellers, estimer med critic.
        if done:
            R = torch.tensor([0.0]).to(config.DEVICE)
        else:
            # Få verdien av *neste* state for bootstrapping
            next_map, next_worm = preprocess_state(next_environment_json)
            with torch.no_grad(): # Ingen gradient nødvendig her
                 _, next_value = self.network(next_map, next_worm)
                 R = next_value.squeeze() # Bruk critic'ens estimat for neste state

        # Beregn discounted returns (Gt) og advantages (At)
        returns = []
        advantages = []
        gae = 0 # Generalized Advantage Estimation (kan starte enkelt med A=G-V)

        # Regn ut returns baklengs
        for r, v in zip(reversed(self.rewards), reversed(self.values)):
            R = r + config.GAMMA * R # R blir Gt for dette steget
            returns.insert(0, R)

        returns = torch.tensor(returns).to(config.DEVICE)
        values_tensor = torch.cat(self.values).squeeze() # Gjør om listen av V(s) til en tensor

        # Enkel Advantage: A(s,a) = Gt - V(s)
        advantages = returns - values_tensor
        # Normaliser advantages (ofte lurt for stabilitet)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) # Pass på deling på 0


        # Beregn Policy Loss (Actor Loss)
        # Negativ fordi vi bruker gradient ascent (maksimering), men optimizers minimerer.
        policy_loss = []
        for log_prob, adv in zip(self.log_probs, advantages.detach()): # .detach() for ikke å backprop'e gjennom advantage
            policy_loss.append(-log_prob * adv)
        policy_loss = torch.stack(policy_loss).sum() # Stack og summer

        # Beregn Value Loss (Critic Loss)
        # Hvor godt estimerte critic'en Gt?
        value_loss = F.mse_loss(values_tensor, returns.detach()) # .detach() på returns

        # Beregn Entropy Bonus (for exploration)
        entropy_loss = -torch.stack(self.entropies).mean() # Gjennomsnittlig entropi, negativ for maksimering

        # Total Loss
        # Vi vil maksimere policy reward og entropi, og minimere value loss.
        total_loss = policy_loss + config.VALUE_LOSS_COEF * value_loss + config.ENTROPY_COEF * entropy_loss

        # Backpropagation
        self.optimizer.zero_grad()
        total_loss.backward()
        # Gradient Clipping (valgfritt, men ofte nyttig for stabilitet)
        # torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
        self.optimizer.step()

        # Tøm lagrede data for neste læringssteg/episode
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.entropies = []

        # Returner tap for logging/plotting
        return policy_loss.item(), value_loss.item(), entropy_loss.item()