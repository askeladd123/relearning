# agents/a2c_manual/config.py
import torch

# ---- Kartdimensjoner og Normalisering ----
MAP_WIDTH = 250  # Forventet maks bredde for CNN-padding/cropping
MAP_HEIGHT = 250 # Forventet maks høyde for CNN-padding/cropping
# utils.py vil håndtere faktiske kartstørrelser for normalisering.

MAX_WORM_HEALTH = 100.0

# ---- Modell Dimensjoner ----
CNN_INPUT_CHANNELS = 1
# Gitt AdaptiveMaxPool2d((6, 6)) og 32 output kanaler fra conv2, blir dette 32*6*6 = 1152
CNN_FEATURE_DIM = 1152

ACTIVE_WORM_FEATURE_DIM = 3  # Egen orms: health, x, y (normalisert)
# TODO: Utvid senere til å inkludere info om andre ormer hvis ønskelig
WORM_VECTOR_DIM = ACTIVE_WORM_FEATURE_DIM
COMBINED_FEATURE_DIM = CNN_FEATURE_DIM + WORM_VECTOR_DIM

# ---- Action Space Definisjoner (basert på ny json-docs.md) ----
# Rekkefølgen nettverket ser handlingene i:
NETWORK_ACTION_ORDER = ['stand', 'walk', 'attack_kick', 'attack_bazooka', 'attack_grenade']
ACTION_DIM = len(NETWORK_ACTION_ORDER)

# Mapping fra nettverkets actionnavn til serverens JSON-format
SERVER_ACTION_MAPPING = {
    'stand': {'action': 'stand'},
    'walk': {'action': 'walk'},  # 'dx' parameter legges til dynamisk
    'attack_kick': {'action': 'attack', 'weapon': 'kick'}, # Ingen 'force' fra klient lenger
    'attack_bazooka': {'action': 'attack', 'weapon': 'bazooka'},  # 'angle_deg' parameter
    'attack_grenade': {'action': 'attack', 'weapon': 'grenade'}  # 'dx' parameter (erstatter angle/force)
}

# ---- Parameter Space for Nettverket ----
# Walk 'dx'. Server maks er +/-2.0.
WALK_DX_BINS = 5  # Gir f.eks. bins for [-2.0, -1.0, 0.0, 1.0, 2.0]
WALK_DX_MIN = -2.0
WALK_DX_MAX = 2.0

# Kick: Ingen parametere som predikeres av nettverket (ingen 'force' fra klient).

# Bazooka 'angle_deg' (kontinuerlig: nettverket outputer mean og std for Normalfordeling)
BAZOOKA_ANGLE_PARAMS = 1 # 1 for mean, 1 for std (totalt 2 noder i modellen for dette)

# Grenade 'dx'. Server maks er +/-5.0.
GRENADE_DX_BINS = 11 # Gir f.eks. bins for [-5.0, -4.0, ..., 0.0, ..., 4.0, 5.0]
GRENADE_DX_MIN = -5.0
GRENADE_DX_MAX = 5.0

# ---- A2C Hyperparametre ----
LEARNING_RATE = 0.0003 # Ofte en god startverdi for A2C
GAMMA = 0.99  # Discount factor
ENTROPY_COEF = 0.01
VALUE_LOSS_COEF = 0.5
MAX_GRAD_NORM = 0.5 # For gradient clipping

# ---- Trening & Lagring ----
# Serveren kjører nå kontinuerlig spill. Klienten bør lagre periodisk.
# NUM_EPISODES i main_coordinator.py styrer hvor mange *spill* hver agent prøver å fullføre
# før den potensielt avslutter sin egen økt (men serveren fortsetter).
# For kontinuerlig trening kan man sette NUM_EPISODES veldig høyt eller la den kjøre evig.
NUM_GAMES_PER_AGENT_SESSION = 10000 # Hvor mange spill en agent-instans skal sikte mot

SAVE_MODEL_EVERY_N_GAMES = 50   # Lagre modell etter X antall *fullførte spill* for denne agenten
PLOT_STATS_EVERY_N_GAMES = 10   # Hvor ofte generere/oppdatere plot

# ---- Websocket ----
SERVER_HOST = '127.0.0.1'
SERVER_PORT = 8765

# ---- Annet ----
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu" # Kan overstyres for debugging
print(f"Bruker enhet: {DEVICE}")