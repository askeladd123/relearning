# agents/a2c_manual/config.py
import torch

# ---- Kartdimensjoner og Normalisering ----
# Disse bør ideelt sett være dynamiske eller store nok for padding.
# game_core.py har et lite 8x4 kart. Vi setter en MAKS forventet størrelse
# for CNN input, men utils.py må håndtere mindre kart.
MAP_WIDTH = 250
MAP_HEIGHT = 250

MAX_WORM_HEALTH = 100.0
# MAX_X_POS og MAX_Y_POS brukes for normalisering i utils.py og bør
# reflektere faktiske kartdimensjoner mottatt fra serveren, ikke padding-størrelsen.

# ---- Modell Dimensjoner ----
CNN_INPUT_CHANNELS = 1
# Gitt AdaptiveMaxPool2d((6, 6)) og 32 output kanaler fra conv2, blir dette 32*6*6 = 1152
CNN_FEATURE_DIM = 1152 # Dette er output etter CNN og pooling, før flattening

# WORM_VECTOR_DIM: Hvilke features skal vi ha for ormene?
# For aktiv orm: normalisert health, x, y
# For andre ormer (potensielt): health, relativ x, relativ y, er fiende?
# Enkel start: Kun egen orms health, x, y (normalisert)
ACTIVE_WORM_FEATURE_DIM = 3 # health, x, y
# La oss for nå kun fokusere på den aktive ormen.
# Hvis vi vil inkludere andre ormer, må WORM_VECTOR_DIM økes.
WORM_VECTOR_DIM = ACTIVE_WORM_FEATURE_DIM
COMBINED_FEATURE_DIM = CNN_FEATURE_DIM + WORM_VECTOR_DIM

# ---- Action Space Definisjoner (basert på json-docs.md og internt for nettverket) ----
# Dette er rekkefølgen handlingene presenteres for nettverkets output-hode (policy).
# Må matche output-laget i model.py og logikken i agent.py.
# 'attack_...' er interne navn for nettverket for å skille våpentyper.
NETWORK_ACTION_ORDER = ['stand', 'walk', 'attack_kick', 'attack_bazooka', 'attack_grenade']
ACTION_DIM = len(NETWORK_ACTION_ORDER) # Antall diskrete hovedhandlinger for nettverket

# Mapping fra nettverkets action navn til serverens action format
# Brukes i utils.format_action
SERVER_ACTION_MAPPING = {
    'stand': {'action': 'stand'},
    'walk': {'action': 'walk'}, # 'dx' parameter legges til dynamisk
    'attack_kick': {'action': 'attack', 'weapon': 'kick'}, # 'force' parameter legges til
    'attack_bazooka': {'action': 'attack', 'weapon': 'bazooka'}, # 'angle_deg' parameter
    'attack_grenade': {'action': 'attack', 'weapon': 'grenade'} # 'angle_deg', 'force' params
}

# ---- Parameter Space for Nettverket ----
# 'walk' -> 'dx' (diskretiserte bins for nettverket)
WALK_DX_BINS = 11 # Gir verdier fra -5 til +5 hvis sentrert rundt 0
WALK_DX_MIN = -5.0 # Minste faktiske dx verdi
WALK_DX_MAX = 5.0  # Største faktiske dx verdi

# Kontinuerlige parametere (nettverket outputer mean og std for Normalfordeling)
# Force (0-100)
KICK_FORCE_PARAMS = 1 # Nettverket outputer 1 verdi for mean, 1 for std (totalt 2 nodes)
BAZOOKA_ANGLE_PARAMS = 1
GRENADE_ANGLE_PARAMS = 1
GRENADE_FORCE_PARAMS = 1

# ---- A2C Hyperparametre ----
LEARNING_RATE = 0.0007
GAMMA = 0.99  # Discount factor
ENTROPY_COEF = 0.01
VALUE_LOSS_COEF = 0.5
MAX_GRAD_NORM = 0.5 # For gradient clipping (valgfritt, men ofte lurt)

# ---- Trening ----
NUM_EPISODES = 10000
# STEPS_PER_UPDATE = 20 # For N-step A2C, hvis ikke episode-basert

# ---- Websocket ----
SERVER_HOST = '127.0.0.1'
SERVER_PORT = 8765

# ---- Agent ----
PLAYER_ID = None # Vil bli satt av serveren ved 'ASSIGN_ID'

# ---- Annet ----
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")