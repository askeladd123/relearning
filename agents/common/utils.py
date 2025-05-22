# agents/a2c_manual/utils.py
import torch
import numpy as np
from agents.common import config


def preprocess_state(current_game_state_json, agent_player_id):
    """
    Konverterer game_state JSON (fra msg['state']) til tensorer klar for nettverket.
    Returnerer en tuple: (map_tensor, worm_vector_tensor)
    """
    try:
        map_data = current_game_state_json['map']
        worms_data = current_game_state_json['worms']

        # --- Kart Preprocessing ---
        map_array = np.array(map_data, dtype=np.float32)
        actual_map_h, actual_map_w = map_array.shape

        # Senter-justert padding/cropping til config.MAP_HEIGHT, config.MAP_WIDTH
        processed_map_array = np.zeros((config.MAP_HEIGHT, config.MAP_WIDTH), dtype=np.float32)

        copy_h_len = min(actual_map_h, config.MAP_HEIGHT)
        src_start_h = max(0, (actual_map_h - copy_h_len) // 2)
        dst_start_h = max(0, (config.MAP_HEIGHT - copy_h_len) // 2)

        copy_w_len = min(actual_map_w, config.MAP_WIDTH)
        src_start_w = max(0, (actual_map_w - copy_w_len) // 2)
        dst_start_w = max(0, (config.MAP_WIDTH - copy_w_len) // 2)

        processed_map_array[dst_start_h: dst_start_h + copy_h_len,
        dst_start_w: dst_start_w + copy_w_len] = \
            map_array[src_start_h: src_start_h + copy_h_len,
            src_start_w: src_start_w + copy_w_len]

        map_tensor = torch.from_numpy(processed_map_array).unsqueeze(0).unsqueeze(0).to(config.DEVICE)

        # --- Worm Data Preprocessing ---
        active_worm_features = [0.0] * config.ACTIVE_WORM_FEATURE_DIM
        my_worm_json_id = agent_player_id - 1  # Server player_id er 1-basert, JSON orm id er 0-basert

        my_worm_alive = False
        for worm_info in worms_data:
            if worm_info['id'] == my_worm_json_id:
                if worm_info['health'] > 0:
                    norm_health = worm_info['health'] / config.MAX_WORM_HEALTH
                    # Normaliser x og y basert på *faktiske* kartdimensjoner for korrekthet
                    norm_x = np.clip(worm_info['x'] / float(actual_map_w - 1 if actual_map_w > 1 else 1.0), 0.0, 1.0)
                    norm_y = np.clip(worm_info['y'] / float(actual_map_h - 1 if actual_map_h > 1 else 1.0), 0.0, 1.0)

                    active_worm_features = [np.clip(norm_health, 0, 1), norm_x, norm_y]
                    my_worm_alive = True
                break  # Funnet vår orm (død eller levende), ikke nødvendig å lete mer

        # if not my_worm_alive:
        #     print(f"Advarsel preprocess_state: Fant ikke levende orm data for P{agent_player_id} (JSON id {my_worm_json_id}). Bruker null-vektor.")

        worm_vector_tensor = torch.FloatTensor([active_worm_features]).to(config.DEVICE)
        return map_tensor, worm_vector_tensor

    except KeyError as e:
        print(
            f"FEIL preprocess_state: Manglende nøkkel '{e}' i game_state_json: {str(current_game_state_json)[:200]}...")
        # Returner dummy-tensorer for å unngå krasj, men dette indikerer et problem.
        dummy_map = torch.zeros(1, config.CNN_INPUT_CHANNELS, config.MAP_HEIGHT, config.MAP_WIDTH).to(config.DEVICE)
        dummy_worm = torch.zeros(1, config.WORM_VECTOR_DIM).to(config.DEVICE)
        return dummy_map, dummy_worm
    except Exception as e:
        print(f"Uventet FEIL preprocess_state: {type(e).__name__} - {e} data: {str(current_game_state_json)[:200]}...")
        dummy_map = torch.zeros(1, config.CNN_INPUT_CHANNELS, config.MAP_HEIGHT, config.MAP_WIDTH).to(config.DEVICE)
        dummy_worm = torch.zeros(1, config.WORM_VECTOR_DIM).to(config.DEVICE)
        return dummy_map, dummy_worm


def _convert_bin_to_value(bin_idx, num_bins, min_val, max_val):
    """ Hjelpefunksjon for å konvertere en bin-indeks til en faktisk verdi. """
    if num_bins <= 1:  # Unngå divisjon med null hvis num_bins er 0 eller 1
        return (min_val + max_val) / 2.0
    value = min_val + bin_idx * ((max_val - min_val) / float(num_bins - 1))
    return np.clip(value, min_val, max_val)


def format_action(network_action_idx, params_from_network):
    """
    Formaterer valgt handling og parametere til JSON-formatet serveren forventer.
    """
    network_action_name = config.NETWORK_ACTION_ORDER[network_action_idx]

    if network_action_name not in config.SERVER_ACTION_MAPPING:
        print(f"FEIL format_action: Ukjent nettverkshandling '{network_action_name}'. Sender 'stand'.")
        return {"action": "stand"}

    action_json = config.SERVER_ACTION_MAPPING[network_action_name].copy()

    if network_action_name == 'walk':
        dx_bin_idx = params_from_network.get('walk_dx_bin_idx', config.WALK_DX_BINS // 2)
        dx_value = _convert_bin_to_value(dx_bin_idx, config.WALK_DX_BINS, config.WALK_DX_MIN, config.WALK_DX_MAX)
        action_json['dx'] = float(np.clip(dx_value, -2.0, 2.0))  # Server klipper til +/-2.0

    elif network_action_name == 'attack_kick':
        # Ingen 'force' parameter fra klienten lenger
        pass

    elif network_action_name == 'attack_bazooka':
        angle_val = params_from_network.get('bazooka_angle_val', 0.0)
        action_json['angle_deg'] = float(angle_val)
        # Ingen 'force' for bazooka

    elif network_action_name == 'attack_grenade':
        # Bruker nå 'dx' for grenade
        dx_bin_idx = params_from_network.get('grenade_dx_bin_idx', config.GRENADE_DX_BINS // 2)
        dx_value = _convert_bin_to_value(dx_bin_idx, config.GRENADE_DX_BINS, config.GRENADE_DX_MIN,
                                         config.GRENADE_DX_MAX)
        action_json['dx'] = float(np.clip(dx_value, -5.0, 5.0))  # Server klipper til +/-5.0

    return action_json