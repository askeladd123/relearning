# agents/a2c_manual/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import config  # Bruk relativ import


class ActorCriticNetwork(nn.Module):
    def __init__(self):
        super(ActorCriticNetwork, self).__init__()

        # --- CNN for Map Processing ---
        self.conv1 = nn.Conv2d(config.CNN_INPUT_CHANNELS, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.pool = nn.AdaptiveMaxPool2d((6, 6))  # Sikrer fast output-størrelse: [B, 32, 6, 6]
        # config.CNN_FEATURE_DIM er satt til 1152 (32*6*6)

        # --- Felles Fullt Tilkoblede Lag ---
        self.fc_shared1 = nn.Linear(config.CNN_FEATURE_DIM + config.WORM_VECTOR_DIM, 256)
        self.fc_shared2 = nn.Linear(256, 128)

        # --- Actor Hoder ---
        # 1. Hode for diskret action type (basert på config.NETWORK_ACTION_ORDER)
        self.action_type_head = nn.Linear(128, config.ACTION_DIM)  # ACTION_DIM = len(NETWORK_ACTION_ORDER)

        # 2. Parameterhoder
        # Walk 'dx' (logits for diskrete bins)
        self.walk_dx_head = nn.Linear(128, config.WALK_DX_BINS)

        # Kick 'force' (mean og log_std for Normalfordeling)
        self.kick_force_mean_head = nn.Linear(128, config.KICK_FORCE_PARAMS)
        self.kick_force_log_std_head = nn.Linear(128, config.KICK_FORCE_PARAMS)

        # Bazooka 'angle_deg' (mean og log_std)
        self.bazooka_angle_mean_head = nn.Linear(128, config.BAZOOKA_ANGLE_PARAMS)
        self.bazooka_angle_log_std_head = nn.Linear(128, config.BAZOOKA_ANGLE_PARAMS)
        # Ingen force for bazooka ifølge json-docs.md

        # Grenade 'angle_deg' (mean og log_std)
        self.grenade_angle_mean_head = nn.Linear(128, config.GRENADE_ANGLE_PARAMS)
        self.grenade_angle_log_std_head = nn.Linear(128, config.GRENADE_ANGLE_PARAMS)
        # Grenade 'force' (mean og log_std)
        self.grenade_force_mean_head = nn.Linear(128, config.GRENADE_FORCE_PARAMS)
        self.grenade_force_log_std_head = nn.Linear(128, config.GRENADE_FORCE_PARAMS)

        # --- Critic Hode ---
        self.value_head = nn.Linear(128, 1)  # Outputter state value V(s)

    def forward(self, map_tensor, worm_vector_tensor):
        # CNN
        x_map = F.relu(self.conv1(map_tensor))
        x_map = F.relu(self.conv2(x_map))
        x_map = self.pool(x_map)
        x_map = x_map.view(x_map.size(0), -1)  # Flatten til [BatchSize, CNN_FEATURE_DIM]

        # Kombiner med worm data
        if worm_vector_tensor.dim() == 1:  # Hvis batch size = 1 og worm_vector ikke har batch dim
            worm_vector_tensor = worm_vector_tensor.unsqueeze(0)

        try:
            x_combined = torch.cat((x_map, worm_vector_tensor), dim=1)
        except RuntimeError as e:
            print(f"FEIL ved torch.cat: map_shape={x_map.shape}, worm_shape={worm_vector_tensor.shape}")
            print(f"Forventet worm_vector_dim: {config.WORM_VECTOR_DIM}, CNN_feature_dim: {config.CNN_FEATURE_DIM}")
            raise e

        # Felles lag
        x_shared = F.relu(self.fc_shared1(x_combined))
        x_shared = F.relu(self.fc_shared2(x_shared))

        # --- Actor Outputs ---
        action_type_logits = self.action_type_head(x_shared)
        action_type_probs = F.softmax(action_type_logits, dim=-1)

        # Walk
        walk_dx_logits = self.walk_dx_head(x_shared)
        walk_dx_probs = F.softmax(walk_dx_logits, dim=-1)

        # Kick
        kick_force_mean = self.kick_force_mean_head(x_shared)
        kick_force_log_std = self.kick_force_log_std_head(x_shared)
        kick_force_std = torch.exp(kick_force_log_std.clamp(-20, 2))  # Stabilitet

        # Bazooka
        bazooka_angle_mean = self.bazooka_angle_mean_head(x_shared)
        bazooka_angle_log_std = self.bazooka_angle_log_std_head(x_shared)
        bazooka_angle_std = torch.exp(bazooka_angle_log_std.clamp(-20, 2))

        # Grenade
        grenade_angle_mean = self.grenade_angle_mean_head(x_shared)
        grenade_angle_log_std = self.grenade_angle_log_std_head(x_shared)
        grenade_angle_std = torch.exp(grenade_angle_log_std.clamp(-20, 2))
        grenade_force_mean = self.grenade_force_mean_head(x_shared)
        grenade_force_log_std = self.grenade_force_log_std_head(x_shared)
        grenade_force_std = torch.exp(grenade_force_log_std.clamp(-20, 2))

        actor_outputs = {
            'action_type_probs': action_type_probs,
            'walk_dx_probs': walk_dx_probs,
            'kick_params': (kick_force_mean, kick_force_std),
            'bazooka_params': (bazooka_angle_mean, bazooka_angle_std),  # Kun angle
            'grenade_params': (grenade_angle_mean, grenade_angle_std, grenade_force_mean, grenade_force_std)
        }

        # --- Critic Output ---
        state_value = self.value_head(x_shared)

        return actor_outputs, state_value