# agents/common/config.py
import torch

# ---- Kartdimensjoner og Normalisering (Felles) ----
MAP_WIDTH = 250
MAP_HEIGHT = 250
MAX_WORM_HEALTH = 100.0

# ---- Modell Dimensjoner (Felles) ----
CNN_INPUT_CHANNELS = 1
CNN_FEATURE_DIM = 1152 # 32*6*6
ACTIVE_WORM_FEATURE_DIM = 3
WORM_VECTOR_DIM = ACTIVE_WORM_FEATURE_DIM
COMBINED_FEATURE_DIM = CNN_FEATURE_DIM + WORM_VECTOR_DIM

# ---- Action Space Definisjoner (Felles) ----
NETWORK_ACTION_ORDER = ['stand', 'walk', 'attack_kick', 'attack_bazooka', 'attack_grenade']
ACTION_DIM = len(NETWORK_ACTION_ORDER)
SERVER_ACTION_MAPPING = {
    'stand': {'action': 'stand'},
    'walk': {'action': 'walk'},
    'attack_kick': {'action': 'attack', 'weapon': 'kick'},
    'attack_bazooka': {'action': 'attack', 'weapon': 'bazooka'},
    'attack_grenade': {'action': 'attack', 'weapon': 'grenade'}
}

# ---- Parameter Space for Nettverket (Felles) ----
WALK_DX_BINS = 5
WALK_DX_MIN = -2.0
WALK_DX_MAX = 2.0
BAZOOKA_ANGLE_PARAMS = 1 # For mean (std beregnes også, så 2 output noder per)
GRENADE_DX_BINS = 11
GRENADE_DX_MIN = -5.0
GRENADE_DX_MAX = 5.0

# ---- A2C Hyperparametre ----
LEARNING_RATE = 0.0003 # For A2C
GAMMA = 0.99  # Felles discount factor
ENTROPY_COEF = 0.01 # Felles
VALUE_LOSS_COEF = 0.5 # Felles
MAX_GRAD_NORM = 0.5 # Felles for A2C

# ---- PPO Hyperparametre ----
LEARNING_RATE_PPO = 0.0003 # Kan være lik A2C, eller justeres
GAMMA_PPO = 0.99 # Ofte lik vanlig gamma   # <--- DENNE LINJEN SKAL IKKE VÆRE UTKOMMENTERT
ENTROPY_COEF_PPO = 0.01
VALUE_LOSS_COEF_PPO = 0.5
MAX_GRAD_NORM_PPO = 0.5
PPO_EPOCHS = 4             # Antall ganger å iterere over batchen
PPO_BATCH_SIZE = 128       # Antall steg å samle før PPO-oppdatering
PPO_CLIP_EPSILON = 0.2     # PPO clipping parameter
GAE_LAMBDA_PPO = 0.95      # Lambda for Generalized Advantage Estimation

# ---- Trening & Lagring (Felles) ----
NUM_GAMES_PER_AGENT_SESSION = 10000
SAVE_MODEL_EVERY_N_GAMES = 50
PLOT_STATS_EVERY_N_GAMES = 10

# ---- Websocket (Felles) ----
SERVER_HOST = '127.0.0.1'
SERVER_PORT = 8765

# ---- Annet (Felles) ----
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Bruker enhet: {DEVICE}")