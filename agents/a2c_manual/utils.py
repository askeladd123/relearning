# agents/a2c_manual/utils.py
import torch
import numpy as np
from . import config


def preprocess_state(current_game_state_json, agent_player_id):
    """
    Konverterer game_state JSON (fra msg['state']) til tensorer klar for nettverket.
    Returnerer en tuple: (map_tensor, worm_vector_tensor)
    """
    try:
        # `current_game_state_json` er nå det som var `new-environment` før,
        # dvs. direkte innholdet i `msg['state']`.
        map_data = current_game_state_json['map']
        worms_data = current_game_state_json['worms']

        # --- Kart Preprocessing ---
        map_array = np.array(map_data, dtype=np.float32)
        actual_map_h, actual_map_w = map_array.shape

        # Padding/Cropping for å matche config.MAP_HEIGHT, config.MAP_WIDTH
        processed_map_array = np.zeros((config.MAP_HEIGHT, config.MAP_WIDTH), dtype=np.float32)

        copy_h_len = min(actual_map_h, config.MAP_HEIGHT)
        copy_w_len = min(actual_map_w, config.MAP_WIDTH)

        # Senter-justert padding/cropping
        src_start_h = (actual_map_h - copy_h_len) // 2
        src_start_w = (actual_map_w - copy_w_len) // 2
        dst_start_h = (config.MAP_HEIGHT - copy_h_len) // 2
        dst_start_w = (config.MAP_WIDTH - copy_w_len) // 2

        processed_map_array[dst_start_h: dst_start_h + copy_h_len,
        dst_start_w: dst_start_w + copy_w_len] = \
            map_array[src_start_h: src_start_h + copy_h_len,
            src_start_w: src_start_w + copy_w_len]

        map_tensor = torch.from_numpy(processed_map_array).unsqueeze(0).unsqueeze(0).to(config.DEVICE)

        # --- Worm Data Preprocessing ---
        active_worm_features = [0.0] * config.ACTIVE_WORM_FEATURE_DIM  # Default til null-vektor

        # agent_player_id er 1-basert. Orm ID i JSON er 0-basert.
        my_worm_json_id = agent_player_id - 1

        found_my_worm = False
        for worm_info in worms_data:
            if worm_info['id'] == my_worm_json_id:
                # Normaliser aktiv orm data
                # Bruk faktiske kartdimensjoner for normalisering av posisjon
                norm_health = worm_info['health'] / config.MAX_WORM_HEALTH
                # Klipp x og y for å unngå verdier utenfor [0,1] ved kanten av kartet
                norm_x = np.clip(worm_info['x'] / float(actual_map_w - 1 if actual_map_w > 1 else 1), 0.0, 1.0)
                norm_y = np.clip(worm_info['y'] / float(actual_map_h - 1 if actual_map_h > 1 else 1), 0.0, 1.0)

                active_worm_features = [
                    np.clip(norm_health, 0.0, 1.0),
                    norm_x,
                    norm_y
                ]
                found_my_worm = True
                break

        if not found_my_worm:
            # Dette kan skje hvis ormen er død og fjernet fra listen,
            # eller hvis det er en feil i player_id matching.
            # Agenten bør fortsatt få en input.
            # print(f"Advarsel: Fant ikke orm data for agent player_id {agent_player_id} (intern id {my_worm_json_id})")
            pass  # Bruker default null-vektor

        worm_vector_tensor = torch.FloatTensor([active_worm_features]).to(config.DEVICE)

        return map_tensor, worm_vector_tensor

    except KeyError as e:
        print(
            f"FEIL i preprocess_state: Manglende nøkkel '{e}' i game_state_json: {str(current_game_state_json)[:200]}...")
        dummy_map = torch.zeros(1, config.CNN_INPUT_CHANNELS, config.MAP_HEIGHT, config.MAP_WIDTH).to(config.DEVICE)
        dummy_worm = torch.zeros(1, config.WORM_VECTOR_DIM).to(config.DEVICE)
        return dummy_map, dummy_worm
    except Exception as e:
        print(
            f"Uventet FEIL i preprocess_state: {type(e).__name__} - {e} med data {str(current_game_state_json)[:200]}...")
        dummy_map = torch.zeros(1, config.CNN_INPUT_CHANNELS, config.MAP_HEIGHT, config.MAP_WIDTH).to(config.DEVICE)
        dummy_worm = torch.zeros(1, config.WORM_VECTOR_DIM).to(config.DEVICE)
        return dummy_map, dummy_worm


def format_action(network_action_idx, params_from_network):
    """
    Formaterer valgt handling (fra nettverkets interne representasjon) og parametere
    til JSON-formatet som serveren forventer (basert på json-docs.md).

    Args:
        network_action_idx (int): Indeksen til handlingen i config.NETWORK_ACTION_ORDER.
        params_from_network (dict): En dictionary med samplede parametere fra nettverket.
                                    Nøklene bør matche det agent.py legger inn,
                                    f.eks. 'walk_dx_bin_idx', 'kick_force_val', etc.
    """
    network_action_name = config.NETWORK_ACTION_ORDER[network_action_idx]

    # Start med basis action type fra mapping
    if network_action_name not in config.SERVER_ACTION_MAPPING:
        print(f"FEIL: Ukjent nettverkshandling '{network_action_name}' i format_action. Sender 'stand'.")
        return {"action": "stand"}

    action_json = config.SERVER_ACTION_MAPPING[network_action_name].copy()  # Viktig med .copy()

    # Legg til/modifiser parametere basert på nettverkets output
    if network_action_name == 'walk':
        # Konverter bin-indeks for dx tilbake til en faktisk dx-verdi
        # Eks: 11 bins (-5 til +5). Bin 0 -> -5, Bin 5 -> 0, Bin 10 -> +5
        # Formel: verdi = min_verdi + bin_idx * ( (max_verdi - min_verdi) / (antall_bins - 1) )
        # Eller enklere: (bin_idx - (antall_bins // 2)) * step_size, hvis step_size er 1.
        dx_bin_idx = params_from_network.get('walk_dx_bin_idx', config.WALK_DX_BINS // 2)  # Default til midten (0)
        dx_value = config.WALK_DX_MIN + dx_bin_idx * \
                   ((config.WALK_DX_MAX - config.WALK_DX_MIN) / (config.WALK_DX_BINS - 1))
        action_json['dx'] = float(dx_value)

    elif network_action_name == 'attack_kick':
        # Force er 0-100
        force_val = params_from_network.get('kick_force_val', 50.0)  # Default til 50
        action_json['force'] = float(np.clip(force_val, 0.0, 100.0))

    elif network_action_name == 'attack_bazooka':
        angle_val = params_from_network.get('bazooka_angle_val', 0.0)  # Default til 0
        action_json['angle_deg'] = float(angle_val)
        # Merk: Bazooka har ikke 'force' i json-docs.md ACTION, kun i client.py eksempelet.
        # Hvis serveren forventer det, må det legges til i json-docs og her.

    elif network_action_name == 'attack_grenade':
        angle_val = params_from_network.get('grenade_angle_val', 0.0)
        action_json['angle_deg'] = float(angle_val)
        force_val = params_from_network.get('grenade_force_val', 50.0)
        action_json['force'] = float(np.clip(force_val, 0.0, 100.0))

    # 'stand' trenger ingen ekstra parametere utover det som er i SERVER_ACTION_MAPPING

    return action_json