# agents/a2c_manual/config.py
import torch

# ---- Viktig: Disse må kanskje justeres etter diskusjon med Ask ----
# Kartdimensjoner (Eksempel - BRUK VERDIER FRA ASK!)
MAP_WIDTH = 250  # Eksempel basert på Ask sin kommentar
MAP_HEIGHT = 250 # Eksempel basert på Ask sin kommentar
# Antall Worms per lag (eller totalt hvis free-for-all) - Trenger avklaring
NUM_WORMS = 3 # Eksempel basert på environment.json

# Maks verdier for normalisering (Eksempel - trenger justering)
MAX_WORM_HEALTH = 100.0
MAX_X_POS = MAP_WIDTH -1
MAX_Y_POS = MAP_HEIGHT -1
# ------------------------------------------------------------------

# Modell Dimensjoner
# Input til CNN (1 kanal for kartet)
CNN_INPUT_CHANNELS = 1
# Output fra CNN (etter flattening, juster basert på CNN-arkitektur)
CNN_FEATURE_DIM = 256 # Eksempel, avhenger av Conv/Pool lag
# Dimensjon på worm-vektor (id?, health, x, y for *alle* worms)
# (id [one-hot?], health, x, y) * NUM_WORMS? Eller bare health,x,y? La oss starte enkelt:
# health, x, y for *egen* orm + health, x, y for *alle andre* ormer?
# Enkleste start: Kun egen health, x, y (normalisert)
# Må avklares med Ask hvordan AI vet *hvilken* orm den styrer
WORM_VECTOR_DIM = 3 # Kun health, x, y for aktiv orm
# Total input til FC-lag etter CNN og konkatinering
COMBINED_FEATURE_DIM = CNN_FEATURE_DIM + WORM_VECTOR_DIM

# Action Space Definisjoner
ACTION_LIST = ['stand', 'walk', 'kick', 'bazooka', 'grenade']
ACTION_DIM = len(ACTION_LIST) # Antall diskrete handlinger

# Parameter Space (Eksempler - Må defineres nøyere!)
WALK_AMOUNT_BINS = 11 # F.eks., diskrete steg fra -5 til +5
KICK_FORCE_PARAMS = 2  # F.eks., mean og std dev for Gaussian
BAZOOKA_ANGLE_PARAMS = 2 # F.eks., mean og std dev
BAZOOKA_FORCE_PARAMS = 2 # F.eks., mean og std dev
GRENADE_ANGLE_PARAMS = 2 # F.eks., mean og std dev
GRENADE_FORCE_PARAMS = 2 # F.eks., mean og std dev

# Hyperparametre for A2C
LEARNING_RATE = 0.001
GAMMA = 0.99  # Discount factor for fremtidige belønninger
ENTROPY_COEF = 0.01 # Koeffisient for entropi-bonus (oppmuntrer utforskning)
VALUE_LOSS_COEF = 0.5 # Koeffisient for critic loss

# Trening
NUM_EPISODES = 10000 # Antall spill-episoder å trene

# Websocket
SERVER_HOST = '127.0.0.1' # Ask sin server IP
SERVER_PORT = 8765      # Ask sin server port

# Annet
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")