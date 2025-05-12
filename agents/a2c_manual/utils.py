# agents/a2c_manual/utils.py
import torch
import numpy as np
from . import config

def preprocess_state(environment_json):
    """
    Konverterer rå environment JSON til tensorer klar for nettverket.
    Returnerer en tuple: (map_tensor, worm_vector_tensor)
    """
    try:
        env_data = environment_json['new-environment']
        map_data = env_data['map']
        worms_data = env_data['worms']

        # --- Kart Preprocessing ---
        # Anta map_data er en liste av lister
        map_array = np.array(map_data, dtype=np.float32)

        # Resize/Pad kartet til fast størrelse (config.MAP_HEIGHT, config.MAP_WIDTH)
        # Dette er en enkel padding-metode, mer avansert resizing kan være nødvendig
        current_h, current_w = map_array.shape
        padded_map = np.zeros((config.MAP_HEIGHT, config.MAP_WIDTH), dtype=np.float32)
        # Senterer det mindre kartet i det større null-fylte kartet
        pad_h_start = (config.MAP_HEIGHT - current_h) // 2
        pad_w_start = (config.MAP_WIDTH - current_w) // 2
        pad_h_end = pad_h_start + current_h
        pad_w_end = pad_w_start + current_w

        # Sikrer at indekser er innenfor grensene
        if pad_h_start >= 0 and pad_h_end <= config.MAP_HEIGHT and \
           pad_w_start >= 0 and pad_w_end <= config.MAP_WIDTH:
             padded_map[pad_h_start:pad_h_end, pad_w_start:pad_w_end] = map_array[:config.MAP_HEIGHT-pad_h_start, :config.MAP_WIDTH-pad_w_start]
        else:
             # Fallback hvis originalt kart er større enn definert config-størrelse
             padded_map = map_array[:config.MAP_HEIGHT, :config.MAP_WIDTH]


        # Legg til en kanal-dimensjon for CNN (C, H, W)
        map_tensor = torch.from_numpy(padded_map).unsqueeze(0).unsqueeze(0).to(config.DEVICE) # Shape: [1, 1, H, W]

        # --- Worm Data Preprocessing ---
        # VIKTIG: Anta at AI styrer worm med id 0 for nå. Må avklares med Ask!
        # Hvordan vet AI hvilken orm som er aktiv? Får den en ID?
        # For nå, antar vi at ormen AI styrer er den første i listen med health > 0
        active_worm = None
        other_worms_features = []

        # Finn den aktive ormen (første levende)
        for worm in worms_data:
            if worm['health'] > 0 and active_worm is None:
                 active_worm = worm
                 break # Styrer kun en om gangen per design nå

        if active_worm is None and worms_data: # Hvis ingen er levende, ta data fra den første (døde)
            active_worm = worms_data[0]
        elif active_worm is None: # Hvis ingen ormer finnes
             # Returner null-vektor hvis ingen ormer finnes
             worm_vector_tensor = torch.zeros(1, config.WORM_VECTOR_DIM).to(config.DEVICE)
             return map_tensor, worm_vector_tensor

        # Normaliser aktiv orm data
        norm_health = active_worm['health'] / config.MAX_WORM_HEALTH
        norm_x = active_worm['x'] / config.MAX_X_POS
        norm_y = active_worm['y'] / config.MAX_Y_POS

        worm_features = [norm_health, norm_x, norm_y]

        # TODO (Avansert): Inkluder info om *andre* ormer også?
        # for worm in worms_data:
        #    if worm['id'] != active_worm['id']:
        #       # Legg til normaliserte data for andre ormer
        #       pass

        worm_vector_tensor = torch.FloatTensor([worm_features]).to(config.DEVICE) # Shape: [1, WORM_VECTOR_DIM]

        return map_tensor, worm_vector_tensor

    except KeyError as e:
        print(f"Error preprocessing state: Missing key {e} in JSON: {environment_json}")
        # Returner dummy tensors ved feil
        dummy_map = torch.zeros(1, config.CNN_INPUT_CHANNELS, config.MAP_HEIGHT, config.MAP_WIDTH).to(config.DEVICE)
        dummy_worm = torch.zeros(1, config.WORM_VECTOR_DIM).to(config.DEVICE)
        return dummy_map, dummy_worm
    except Exception as e:
        print(f"An unexpected error occurred during preprocessing: {e}")
        # Returner dummy tensors ved feil
        dummy_map = torch.zeros(1, config.CNN_INPUT_CHANNELS, config.MAP_HEIGHT, config.MAP_WIDTH).to(config.DEVICE)
        dummy_worm = torch.zeros(1, config.WORM_VECTOR_DIM).to(config.DEVICE)
        return dummy_map, dummy_worm


def format_action(action_type_idx, params):
    """ Formaterer valgt handling og parametere til JSON for serveren. """
    action_name = config.ACTION_LIST[action_type_idx]
    action_json = {"action": action_name}

    if action_name == 'walk':
        # Konverter bin index tilbake til en verdi, f.eks. (-5 til +5)
        amount = params['walk_amount'] - (config.WALK_AMOUNT_BINS // 2)
        action_json['amount-x'] = float(amount) # Sørg for float
    elif action_name == 'kick':
        action_json['weapon'] = 'kick'
        action_json['force'] = float(params['kick_force']) # Sørg for float
    elif action_name == 'bazooka':
        action_json['weapon'] = 'bazooka'
        action_json['angle'] = float(params['bazooka_angle']) # Sørg for float
        action_json['force'] = float(params['bazooka_force']) # Sørg for float
    elif action_name == 'grenade':
        action_json['weapon'] = 'grenade'
        action_json['angle'] = float(params['grenade_angle']) # Sørg for float
        action_json['force'] = float(params['grenade_force']) # Sørg for float
    # 'stand' trenger ingen ekstra parametere

    # Viktig: Sjekk at datatyper (float vs int) stemmer med hva Ask forventer!
    return action_json