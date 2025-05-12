# agents/a2c_manual/main_a2c.py
import asyncio
import websockets
import json
import time
import torch
import numpy as np
from .agent import A2CAgent
from . import config  # Bruk relativ import


# (Setup logging her hvis du vil, likt som i client.py eller server.py. For enkelhets skyld, print brukes her)

async def run_agent():
    uri = f"ws://{config.SERVER_HOST}:{config.SERVER_PORT}"
    a2c_agent = A2CAgent()
    episode_count = 0
    all_episode_rewards_log = []  # For plotting
    losses_log = {'policy': [], 'value': [], 'entropy': []}  # For plotting

    checkpoint_path = "a2c_worms_checkpoint.pth"
    try:
        a2c_agent.network.load_state_dict(torch.load(checkpoint_path, map_location=torch.device(config.DEVICE)))
        print(f"Lastet sjekkpunkt for modellen fra {checkpoint_path}.")
    except FileNotFoundError:
        print(f"Ingen sjekkpunkt funnet på {checkpoint_path}, starter med ny modell.")
    except Exception as e:
        print(f"Kunne ikke laste sjekkpunkt: {e}")

    print(f"Starter {config.NUM_EPISODES} episoder med A2C agent...")

    for episode_idx in range(config.NUM_EPISODES):
        episode_reward_sum = 0
        # Agentens buffere tømmes i agent.learn() eller agent.clear_buffers()
        # a2c_agent.clear_buffers() # Kan kalles her for ekstra sikkerhet

        game_over_flag = False
        last_received_state = None  # For å ha state tilgjengelig for learn() hvis tilkobling brytes

        print(f"\n--- Starter Episode {episode_idx + 1}/{config.NUM_EPISODES} ---")
        try:
            async with websockets.connect(uri, ping_interval=20, ping_timeout=20) as websocket:
                # 1. CONNECT og ASSIGN_ID
                await websocket.send(json.dumps({"type": "CONNECT", "nick": "A2C-JetBot"}))
                assign_id_msg_str = await websocket.recv()
                assign_id_msg = json.loads(assign_id_msg_str)

                if assign_id_msg.get("type") == "ASSIGN_ID":
                    player_id = assign_id_msg.get("player_id")
                    if player_id is None:
                        print("FEIL: Fikk ikke player_id. Avslutter episoden.")
                        await asyncio.sleep(1)  # Vent litt før neste forsøk
                        continue  # Neste episode
                    a2c_agent.set_player_id(player_id)
                    print(f"Tilkoblet server. Tildelt Player ID: {a2c_agent.player_id}.")
                else:
                    print(f"FEIL: Forventet ASSIGN_ID, fikk: {assign_id_msg}. Avslutter episoden.")
                    await asyncio.sleep(1)
                    continue

                # Hoved meldingsløkke for episoden
                async for message_str in websocket:
                    try:
                        msg = json.loads(message_str)
                        msg_type = msg.get("type")
                        # print(f"DEBUG: Mottok: {msg_type} - {str(msg)[:150]}") # Kort debug

                        if msg_type == "TURN_BEGIN":
                            last_received_state = msg.get("state")  # Lagre for learn() ved evt. disconnect
                            if msg.get("player_id") == a2c_agent.player_id:
                                turn_idx = msg.get('turn_index', -1)
                                print(f"  Min tur (P{a2c_agent.player_id}, Turn {turn_idx}). State mottatt.")
                                if not last_received_state:
                                    print("  FEIL: TURN_BEGIN mangler 'state'. Hopper over handling.")
                                    continue

                                action_to_send_obj = a2c_agent.select_action(last_received_state)
                                action_payload = {
                                    "type": "ACTION",
                                    "player_id": a2c_agent.player_id,
                                    "action": action_to_send_obj
                                }
                                print(f"    Sender handling: {action_payload['action']}")
                                await websocket.send(json.dumps(action_payload))
                            # else: Ikke min tur, ignorer

                        elif msg_type == "TURN_RESULT":
                            # Kun lagre reward hvis det var vår handling
                            if msg.get("player_id") == a2c_agent.player_id:
                                reward = msg.get("reward", 0.0)
                                a2c_agent.store_reward(reward)
                                episode_reward_sum += reward
                                print(f"  Mottok reward: {reward}. Total ep. reward: {episode_reward_sum:.2f}")
                            last_received_state = msg.get("state")  # Oppdater uansett

                        elif msg_type == "TURN_END":
                            pass  # Vent på neste TURN_BEGIN

                        elif msg_type == "GAME_OVER":
                            print(f"GAME OVER mottatt. Vinner: {msg.get('winner_id')}. ")
                            game_over_flag = True
                            final_state = msg.get("final_state")
                            # Hvis siste handling ikke var vår, har vi ikke en reward for final_state.
                            # A2C lærer vanligvis fra (s, a, r, s_next).
                            # Hvis det ikke var vår tur sist, er siste reward i buffer fra forrige handling.
                            # V(final_state) vil være 0.

                            # Sørg for at vi har en state å sende til learn, selv om det var en annen spillers tur sist
                            current_state_for_learn = final_state if final_state else last_received_state

                            if not a2c_agent.rewards_buffer:  # Hvis ingen handlinger ble tatt av agenten
                                print("  Ingen handlinger/rewards lagret denne episoden. Ingen læring.")
                            else:
                                pol_l, val_l, ent_l = a2c_agent.learn(current_state_for_learn, True)  # True for done
                                if pol_l is not None:
                                    losses_log['policy'].append(pol_l)
                                    losses_log['value'].append(val_l)
                                    losses_log['entropy'].append(ent_l)
                                    print(
                                        f"  Lært fra episode. Losses - P: {pol_l:.4f}, V: {val_l:.4f}, E: {ent_l:.4f}")
                            break  # Bryt meldingsløkken, episoden er ferdig

                        elif msg_type == "ERROR":
                            print(f"FEIL fra server: {msg.get('msg')}")
                            # Vurder å tømme buffere hvis feilen gjør episoden ugyldig
                            # a2c_agent.clear_buffers()

                    except json.JSONDecodeError:
                        print(f"FEIL: Kunne ikke dekode JSON fra server: {message_str}")
                    except Exception as e_inner:
                        print(f"FEIL i meldingsløkke: {type(e_inner).__name__} - {e_inner}")
                        game_over_flag = True  # Anta episoden er korrupt
                        break  # Bryt meldingsløkken

                # Etter meldingsløkken (enten GAME_OVER eller disconnect)
                if not game_over_flag and a2c_agent.rewards_buffer:  # Disconnect før GAME_OVER
                    print("Advarsel: Tilkobling lukket før GAME_OVER, men data samlet. Lærer (antar 'done').")
                    # Bruk sist kjente state hvis mulig
                    state_for_learn = last_received_state if last_received_state else {}
                    pol_l, val_l, ent_l = a2c_agent.learn(state_for_learn, True)
                    if pol_l is not None:
                        losses_log['policy'].append(pol_l);
                        losses_log['value'].append(val_l);
                        losses_log['entropy'].append(ent_l)

        except websockets.exceptions.ConnectionClosed as e:
            print(f"Tilkobling lukket: {e}. Prøver neste episode.")
            if not game_over_flag and a2c_agent.rewards_buffer:  # Lær hvis data finnes
                print("  Lærer fra ufullstendig episode (ConnectionClosed).")
                state_for_learn = last_received_state if last_received_state else {}
                pol_l, val_l, ent_l = a2c_agent.learn(state_for_learn, True)
                if pol_l: losses_log['policy'].append(pol_l); losses_log['value'].append(val_l); losses_log[
                    'entropy'].append(ent_l)
        except ConnectionRefusedError:
            print(f"FEIL: Kunne ikke koble til server {uri}. Er den startet? Venter 10s.")
            await asyncio.sleep(10)
            continue  # Prøv å starte en ny episodeforbindelse
        except Exception as e_outer:
            print(f"Alvorlig FEIL i episode {episode_idx + 1}: {type(e_outer).__name__} - {e_outer}")
            if not game_over_flag and a2c_agent.rewards_buffer:
                print("  Lærer fra ufullstendig episode (Alvorlig feil).")
                state_for_learn = last_received_state if last_received_state else {}
                pol_l, val_l, ent_l = a2c_agent.learn(state_for_learn, True)
                if pol_l: losses_log['policy'].append(pol_l); losses_log['value'].append(val_l); losses_log[
                    'entropy'].append(ent_l)
            await asyncio.sleep(5)  # Pause før neste forsøk

        all_episode_rewards_log.append(episode_reward_sum)
        print(f"--- Episode {episode_idx + 1} avsluttet. Total belønning: {episode_reward_sum:.2f} ---")

        # Lagre sjekkpunkt periodisk
        if (episode_idx + 1) % 50 == 0:
            try:
                torch.save(a2c_agent.network.state_dict(), checkpoint_path)
                print(f"Lagret modell sjekkpunkt til {checkpoint_path} etter episode {episode_idx + 1}")
            except Exception as e_save:
                print(f"Kunne ikke lagre modell: {e_save}")

        # Liten pause mellom episoder for å unngå å hamre serveren
        await asyncio.sleep(0.1)

    print("\nTrening ferdig etter alle episoder.")
    # Plotting (samme som før)
    try:
        import matplotlib.pyplot as plt
        avg_rewards = []
        if all_episode_rewards_log:
            window_size = min(100, len(all_episode_rewards_log))
            if window_size > 0:
                avg_rewards = [np.mean(all_episode_rewards_log[max(0, i - window_size + 1):i + 1]) for i in
                               range(len(all_episode_rewards_log))]

        plt.figure(figsize=(18, 6))
        plt.subplot(1, 3, 1)
        plt.plot(all_episode_rewards_log, label='Rå belønning', alpha=0.6)
        if avg_rewards: plt.plot(avg_rewards, label=f'Glidende gj.snitt ({window_size} ep)', color='red', linewidth=2)
        plt.xlabel("Episode");
        plt.ylabel("Total belønning");
        plt.title("Belønning per episode")
        plt.legend();
        plt.grid(True)

        if losses_log['policy']:  # Sjekk om det er noe å plotte
            plt.subplot(1, 3, 2)
            plt.plot(losses_log['policy'], label='Policy Loss')
            plt.xlabel("Læringssteg");
            plt.ylabel("Policy Loss");
            plt.title("Policy (Actor) Tap")
            plt.legend();
            plt.grid(True)

            plt.subplot(1, 3, 3)
            plt.plot(losses_log['value'], label='Value Loss', color='green')
            plt.xlabel("Læringssteg");
            plt.ylabel("Value Loss");
            plt.title("Value (Critic) Tap")
            plt.legend();
            plt.grid(True)

        # Du kan også plotte entropi hvis du vil:
        # if losses_log['entropy']:
        #     plt.figure() # Ny figur for entropi
        #     plt.plot(losses_log['entropy'], label='Entropy (neg)')
        #     plt.xlabel("Læringssteg"); plt.ylabel("Entropy (neg)"); plt.title("Entropi")
        #     plt.legend(); plt.grid(True)

        plt.tight_layout()
        plt.savefig("training_plots_worms_a2c_new_protocol.png")
        print("Lagret treningsplot som training_plots_worms_a2c_new_protocol.png")
        # plt.show() # Kommenter ut hvis du kjører uten GUI
    except ImportError:
        print("Matplotlib ikke installert. Kan ikke plotte resultater.")
    except Exception as plot_e:
        print(f"Kunne ikke plotte resultater: {type(plot_e).__name__} - {plot_e}")


if __name__ == "__main__":
    try:
        asyncio.run(run_agent())
    except KeyboardInterrupt:
        print("\nTrening avbrutt av bruker.")