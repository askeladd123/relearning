# agents/a2c_manual/main_a2c.py
import asyncio
import websockets
import json
import time
import torch # For å kunne laste/lagre modell
import numpy as np # For glidende gjennomsnitt
from .agent import A2CAgent
from . import config

async def run_agent():
    uri = f"ws://{config.SERVER_HOST}:{config.SERVER_PORT}"
    a2c_agent = A2CAgent()
    episode_count = 0
    total_steps_across_episodes = 0
    all_episode_rewards = []
    losses_log = {'policy': [], 'value': [], 'entropy': []}

    checkpoint_path = "a2c_worms_checkpoint.pth"
    try:
        a2c_agent.network.load_state_dict(torch.load(checkpoint_path, map_location=torch.device(config.DEVICE)))
        print(f"Lastet sjekkpunkt for modellen fra {checkpoint_path}.")
    except FileNotFoundError:
        print(f"Ingen sjekkpunkt funnet på {checkpoint_path}, starter med ny modell.")
    except Exception as e:
        print(f"Kunne ikke laste sjekkpunkt: {e}")

    print(f"Kobler til server på {uri}...")

    while episode_count < config.NUM_EPISODES:
        episode_reward = 0
        current_step_in_episode = 0
        done = False
        current_environment_json = None # Viktig å nullstille for hver episode

        try:
            async with websockets.connect(uri, ping_interval=20, ping_timeout=20) as websocket:
                print(f"--- Episode {episode_count + 1}/{config.NUM_EPISODES} ---")
                print("Tilkoblet server.")

                # 1. SEND EN INITIAL HANDLING FOR Å FÅ FØRSTE STATE
                initial_action_payload = {"action": "stand"} # Enkel start-handling
                print(f"Sender initial handling: {initial_action_payload}")
                await websocket.send(json.dumps(initial_action_payload))

                # 2. MOTTA FØRSTE ENVIRONMENT ETTER INITIAL HANDLING
                print("Venter på første environment fra server...")
                initial_response_message = await websocket.recv()
                print(f"Mottok første respons: {initial_response_message[:200]}...") # Logg starten av responsen

                try:
                    current_environment_json = json.loads(initial_response_message)
                    if 'new-environment' not in current_environment_json or \
                       'worms' not in current_environment_json.get('new-environment', {}) or \
                       'map' not in current_environment_json.get('new-environment', {}):
                        print("Ugyldig format på første environment (mangler nøkler). Hopper over episode.")
                        await asyncio.sleep(1)
                        episode_count += 1 # Tell som et forsøk
                        continue
                except json.JSONDecodeError:
                     print("Mottok ugyldig JSON som første environment. Hopper over episode.")
                     await asyncio.sleep(1)
                     episode_count += 1
                     continue
                except Exception as e:
                    print(f"Uventet feil ved behandling av første environment: {e}. Hopper over episode.")
                    await asyncio.sleep(1)
                    episode_count += 1
                    continue

                # Episodeløkke
                while not done:
                    print(f"Steg {current_step_in_episode + 1}: Nåværende state (start): {str(current_environment_json)[:100]}...")
                    # 3. Agent velger handling basert på nåværende state
                    action_to_send = a2c_agent.select_action(current_environment_json)

                    # 4. Send handling til server
                    print(f"Sender handling: {action_to_send}")
                    await websocket.send(json.dumps(action_to_send))

                    # 5. Motta resultat fra server
                    print("Venter på respons fra server...")
                    response_message = await websocket.recv()
                    print(f"Mottok respons: {response_message[:200]}...")

                    try:
                        response_data = json.loads(response_message)
                    except json.JSONDecodeError:
                        print("Mottok ugyldig JSON respons fra server. Avslutter episode.")
                        done = True
                        response_data = None
                    except websockets.exceptions.ConnectionClosed:
                        print("Tilkobling lukket av server under spill. Avslutter episode.")
                        done = True
                        response_data = None

                    if response_data:
                        reward = response_data.get('score', 0)
                        # VIKTIG: Diskuter med Ask om 'score' er steg-reward eller total score.
                        # Hvis total score, må du beregne delta:
                        # prev_total_score = current_environment_json.get('score', 0) # Eller en annen måte å hente forrige
                        # step_reward = reward - prev_total_score
                        # For nå, anta 'score' er steg-reward
                        step_reward = reward

                        done = response_data.get('finished', False)
                        next_environment_json = response_data

                        a2c_agent.store_reward(step_reward)
                        episode_reward += step_reward

                        current_environment_json = next_environment_json
                        current_step_in_episode += 1
                        total_steps_across_episodes += 1
                    else: # Ingen responsdata, avslutt episoden
                        done = True

                    if done:
                        print(f"Episode {episode_count + 1} ferdig etter {current_step_in_episode} steg. Total belønning: {episode_reward}")
                        policy_l, value_l, entropy_l = a2c_agent.learn(current_environment_json, True)
                        if policy_l is not None:
                             losses_log['policy'].append(policy_l)
                             losses_log['value'].append(value_l)
                             losses_log['entropy'].append(entropy_l)
                             print(f"  Losses - Policy: {policy_l:.4f}, Value: {value_l:.4f}, Entropy (neg): {entropy_l:.4f}")
                        all_episode_rewards.append(episode_reward)

                        if (episode_count + 1) % 50 == 0:
                            try:
                                torch.save(a2c_agent.network.state_dict(), checkpoint_path)
                                print(f"Lagret modell sjekkpunkt til {checkpoint_path} ved episode {episode_count + 1}")
                            except Exception as e_save:
                                print(f"Kunne ikke lagre modell: {e_save}")
                        break # Gå ut av while not done

                episode_count += 1 # Øk etter at en episode er helt ferdig eller avbrutt og håndtert

        except websockets.exceptions.ConnectionClosedOK:
            print(f"Tilkobling lukket (OK) av server for episode {episode_count +1}.")
            if a2c_agent.rewards and not done:
                 print("Tilkobling lukket før episoden var 'finished'. Prøver å lære.")
                 valid_last_state = current_environment_json if current_environment_json else {}
                 policy_l, value_l, entropy_l = a2c_agent.learn(valid_last_state, True)
                 if policy_l is not None:
                      losses_log['policy'].append(policy_l); losses_log['value'].append(value_l); losses_log['entropy'].append(entropy_l)
                 all_episode_rewards.append(episode_reward)
                 a2c_agent.log_probs, a2c_agent.values, a2c_agent.rewards, a2c_agent.entropies = [], [], [], []
            episode_count +=1
            await asyncio.sleep(1)

        except websockets.exceptions.ConnectionClosedError as e:
            print(f"Tilkobling lukket med feil: {e} (Episode {episode_count+1}). Prøver igjen om 5s.")
            await asyncio.sleep(5)

        except ConnectionRefusedError:
            print(f"Kunne ikke koble til serveren {uri}. Er serveren startet? Prøver igjen om 10s.")
            await asyncio.sleep(10)

        except Exception as e:
            print(f"Uventet feil i episode {episode_count + 1}: {type(e).__name__} - {e}")
            if a2c_agent.rewards and not done:
                try:
                    print("Uventet feil midt i episoden. Prøver å lære.")
                    valid_last_state = current_environment_json if current_environment_json else {}
                    policy_l, value_l, entropy_l = a2c_agent.learn(valid_last_state, True)
                    if policy_l is not None:
                        losses_log['policy'].append(policy_l); losses_log['value'].append(value_l); losses_log['entropy'].append(entropy_l)
                    all_episode_rewards.append(episode_reward)
                except Exception as learn_e:
                    print(f"Feil under læring etter feil: {learn_e}")
                finally:
                    a2c_agent.log_probs, a2c_agent.values, a2c_agent.rewards, a2c_agent.entropies = [], [], [], []
            episode_count +=1
            print("Prøver å fortsette eller koble til på nytt om 5s...")
            await asyncio.sleep(5)

    print("Trening ferdig.")
    try:
        import matplotlib.pyplot as plt
        # Beregn glidende gjennomsnitt for rewards
        avg_rewards = []
        if all_episode_rewards: # Sjekk at listen ikke er tom
            window_size = min(100, len(all_episode_rewards)) # Unngå feil hvis færre enn 100 episoder
            avg_rewards = [np.mean(all_episode_rewards[max(0, i - window_size + 1):i + 1]) for i in range(len(all_episode_rewards))]


        plt.figure(figsize=(18, 6))
        plt.subplot(1, 3, 1)
        plt.plot(all_episode_rewards, label='Rå belønning', alpha=0.6)
        if avg_rewards: plt.plot(avg_rewards, label=f'Glidende gj.snitt ({window_size} ep)', color='red', linewidth=2)
        plt.xlabel("Episode")
        plt.ylabel("Total belønning")
        plt.title("Belønning per episode")
        plt.legend()
        plt.grid(True)

        if losses_log['policy']:
            plt.subplot(1, 3, 2)
            plt.plot(losses_log['policy'], label='Policy Loss')
            plt.xlabel("Læringssteg (episoder)")
            plt.ylabel("Policy Loss")
            plt.title("Policy (Actor) Tap")
            plt.legend()
            plt.grid(True)

            plt.subplot(1, 3, 3)
            plt.plot(losses_log['value'], label='Value Loss', color='green')
            plt.xlabel("Læringssteg (episoder)")
            plt.ylabel("Value Loss")
            plt.title("Value (Critic) Tap")
            plt.legend()
            plt.grid(True)

        plt.tight_layout()
        plt.savefig("training_plots_worms.png")
        print("Lagret treningsplot som training_plots_worms.png")
        # plt.show() # Vurder å kommentere ut plt.show() hvis du kjører mange tester uten interaksjon
    except ImportError:
        print("Matplotlib ikke installert. Kan ikke plotte resultater.")
    except Exception as plot_e:
        print(f"Kunne ikke plotte resultater: {type(plot_e).__name__} - {plot_e}")

if __name__ == "__main__":
    asyncio.run(run_agent())