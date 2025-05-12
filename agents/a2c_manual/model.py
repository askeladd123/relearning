# agents/a2c_manual/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from . import config

class ActorCriticNetwork(nn.Module):
    def __init__(self):
        super(ActorCriticNetwork, self).__init__()

        # --- CNN for Map Processing ---
        # Eksempel arkitektur - kan justeres kraftig
        self.conv1 = nn.Conv2d(config.CNN_INPUT_CHANNELS, 16, kernel_size=8, stride=4) # Output: [B, 16, H', W']
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)                      # Output: [B, 32, H'', W'']
        # Beregn størrelsen etter conv-lagene for flattening
        # Dette må gjøres manuelt eller dynamisk basert på input-størrelse
        # Anta 250x250 input for eksempelberegning:
        # Etter conv1 (stride 4): H' = floor((250-8)/4 + 1) = 61, W' = 61 => (16, 61, 61)
        # Etter conv2 (stride 2): H'' = floor((61-4)/2 + 1) = 29, W'' = 29 => (32, 29, 29)
        # Man må kanskje legge til pooling lag også.
        # La oss anta output etter convs (og evt pooling/flatten) er config.CNN_FEATURE_DIM
        # For et konkret eksempel, anta at output er (32 * 29 * 29) hvis ingen pooling.
        # self.cnn_output_size = 32 * 29 * 29 # Eksempel
        # La oss bruke en AdaptiveMaxPool2d for å få fast størrelse uansett input
        self.pool = nn.AdaptiveMaxPool2d((6, 6)) # Output: [B, 32, 6, 6]
        self.cnn_output_size = 32 * 6 * 6         # = 1152. La config.CNN_FEATURE_DIM være dette.
        # Juster config.CNN_FEATURE_DIM til 1152 eller juster arkitekturen

        # --- Felles Fullt Tilkoblede Lag ---
        self.fc_shared1 = nn.Linear(self.cnn_output_size + config.WORM_VECTOR_DIM, 256) # Kombinerer CNN og worm data
        self.fc_shared2 = nn.Linear(256, 128)

        # --- Actor Hoder ---
        # 1. Hode for diskret action type (stand, walk, kick, etc.)
        self.action_head = nn.Linear(128, config.ACTION_DIM)

        # 2. Hoder for parametere (kun relevant for visse actions)
        #    Vi lager separate hoder for hver parameter type

        # Walk amount (antar diskrete bins)
        self.walk_amount_head = nn.Linear(128, config.WALK_AMOUNT_BINS)

        # Kick force (antar kontinuerlig - output mean og std dev)
        self.kick_force_mean_head = nn.Linear(128, 1)
        self.kick_force_log_std_head = nn.Linear(128, 1) # Log std for stabilitet

        # Bazooka angle (kontinuerlig)
        self.bazooka_angle_mean_head = nn.Linear(128, 1)
        self.bazooka_angle_log_std_head = nn.Linear(128, 1)
        # Bazooka force (kontinuerlig)
        self.bazooka_force_mean_head = nn.Linear(128, 1)
        self.bazooka_force_log_std_head = nn.Linear(128, 1)

        # Grenade angle (kontinuerlig)
        self.grenade_angle_mean_head = nn.Linear(128, 1)
        self.grenade_angle_log_std_head = nn.Linear(128, 1)
        # Grenade force (kontinuerlig)
        self.grenade_force_mean_head = nn.Linear(128, 1)
        self.grenade_force_log_std_head = nn.Linear(128, 1)


        # --- Critic Hode ---
        self.value_head = nn.Linear(128, 1) # Outputter state value V(s)

    def forward(self, map_tensor, worm_vector_tensor):
        # CNN
        # print("Map tensor shape:", map_tensor.shape) # Debugging shape
        x_map = F.relu(self.conv1(map_tensor))
        x_map = F.relu(self.conv2(x_map))
        x_map = self.pool(x_map)
        x_map = x_map.view(x_map.size(0), -1) # Flatten

        # Kombiner med worm data
        # print("CNN output shape:", x_map.shape) # Debugging shape
        # print("Worm vector shape:", worm_vector_tensor.shape) # Debugging shape
        x_combined = torch.cat((x_map, worm_vector_tensor), dim=1)

        # Felles lag
        x = F.relu(self.fc_shared1(x_combined))
        x = F.relu(self.fc_shared2(x))

        # --- Actor Outputs ---
        action_logits = self.action_head(x)
        action_probs = F.softmax(action_logits, dim=-1)

        walk_amount_logits = self.walk_amount_head(x)
        walk_amount_probs = F.softmax(walk_amount_logits, dim=-1)

        kick_force_mean = self.kick_force_mean_head(x)
        kick_force_log_std = self.kick_force_log_std_head(x)
        kick_force_std = torch.exp(kick_force_log_std) # Sørger for positiv std dev

        bazooka_angle_mean = self.bazooka_angle_mean_head(x)
        bazooka_angle_log_std = self.bazooka_angle_log_std_head(x)
        bazooka_angle_std = torch.exp(bazooka_angle_log_std)

        bazooka_force_mean = self.bazooka_force_mean_head(x)
        bazooka_force_log_std = self.bazooka_force_log_std_head(x)
        bazooka_force_std = torch.exp(bazooka_force_log_std)

        grenade_angle_mean = self.grenade_angle_mean_head(x)
        grenade_angle_log_std = self.grenade_angle_log_std_head(x)
        grenade_angle_std = torch.exp(grenade_angle_log_std)

        grenade_force_mean = self.grenade_force_mean_head(x)
        grenade_force_log_std = self.grenade_force_log_std_head(x)
        grenade_force_std = torch.exp(grenade_force_log_std)

        # Pakk actor outputs i en dict
        actor_outputs = {
            'action_probs': action_probs,
            'walk_amount_probs': walk_amount_probs,
            'kick_params': (kick_force_mean, kick_force_std),
            'bazooka_params': (bazooka_angle_mean, bazooka_angle_std, bazooka_force_mean, bazooka_force_std),
            'grenade_params': (grenade_angle_mean, grenade_angle_std, grenade_force_mean, grenade_force_std)
        }

        # --- Critic Output ---
        state_value = self.value_head(x)

        return actor_outputs, state_value