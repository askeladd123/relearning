# agents/ppo_manual/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
# Endre denne linjen:
# from . import config
# til:
from agents.common import config


class ActorCriticNetwork(nn.Module):
    def __init__(self):
        super(ActorCriticNetwork, self).__init__()

        # --- CNN for Map Processing ---
        self.conv1 = nn.Conv2d(config.CNN_INPUT_CHANNELS, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.pool = nn.AdaptiveMaxPool2d((6, 6))
        # CNN_FEATURE_DIM forblir 32*6*6 = 1152

        # --- Felles Fullt Tilkoblede Lag ---
        self.fc_shared1 = nn.Linear(config.CNN_FEATURE_DIM + config.WORM_VECTOR_DIM, 256)
        self.fc_shared2 = nn.Linear(256, 128)

        # --- Actor Hoder ---
        # 1. Hode for diskret action type
        self.action_type_head = nn.Linear(128, config.ACTION_DIM)

        # 2. Parameterhoder
        # Walk 'dx' (logits for diskrete bins)
        self.walk_dx_head = nn.Linear(128, config.WALK_DX_BINS)

        # Kick: Ingen parameterhoder, da 'force' ikke lenger sendes fra klienten.

        # Bazooka 'angle_deg' (mean og log_std for Normalfordeling)
        self.bazooka_angle_mean_head = nn.Linear(128, config.BAZOOKA_ANGLE_PARAMS)
        self.bazooka_angle_log_std_head = nn.Linear(128, config.BAZOOKA_ANGLE_PARAMS)

        # Grenade 'dx' (logits for diskrete bins)
        self.grenade_dx_head = nn.Linear(128, config.GRENADE_DX_BINS)

        # --- Critic Hode ---
        self.value_head = nn.Linear(128, 1)

    def forward(self, map_tensor, worm_vector_tensor):
        # CNN
        x_map = F.relu(self.conv1(map_tensor))
        x_map = F.relu(self.conv2(x_map))
        x_map = self.pool(x_map)
        x_map = x_map.view(x_map.size(0), -1)  # Flatten

        # Kombiner med worm data
        if worm_vector_tensor.dim() == 1:
            worm_vector_tensor = worm_vector_tensor.unsqueeze(0)

        try:
            x_combined = torch.cat((x_map, worm_vector_tensor), dim=1)
        except RuntimeError as e:
            print(f"FEIL ved torch.cat: map_shape={x_map.shape}, worm_shape={worm_vector_tensor.shape}")
            print(f"Forventet map flat dim: {config.CNN_FEATURE_DIM}, worm_vector_dim: {config.WORM_VECTOR_DIM}")
            raise e

        # Felles lag
        x_shared = F.relu(self.fc_shared1(x_combined))
        x_shared = F.relu(self.fc_shared2(x_shared))

        # --- Actor Outputs ---
        action_type_logits = self.action_type_head(x_shared)
        action_type_probs = F.softmax(action_type_logits, dim=-1)

        # Walk dx (logits for bins)
        walk_dx_logits = self.walk_dx_head(x_shared)
        walk_dx_probs = F.softmax(walk_dx_logits, dim=-1)

        # Bazooka angle (mean, std for Normal dist)
        bazooka_angle_mean = self.bazooka_angle_mean_head(x_shared)
        bazooka_angle_log_std = self.bazooka_angle_log_std_head(x_shared)
        bazooka_angle_std = torch.exp(bazooka_angle_log_std.clamp(-20, 2)) # Klem for stabilitet

        # Grenade dx (logits for bins)
        grenade_dx_logits = self.grenade_dx_head(x_shared)
        grenade_dx_probs = F.softmax(grenade_dx_logits, dim=-1)

        actor_outputs = {
            'action_type_probs': action_type_probs,
            'walk_dx_probs': walk_dx_probs,
            # Kick har ingen parametere som predikeres av nettverket
            'bazooka_params': (bazooka_angle_mean, bazooka_angle_std),
            'grenade_dx_probs': grenade_dx_probs,
        }

        # --- Critic Output ---
        state_value = self.value_head(x_shared)

        return actor_outputs, state_value