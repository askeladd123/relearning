# agents/a2c_manual/main_a2c.py
import os

# Prøv å sette denne FØR andre importer for å håndtere OMP-feilen
# Dette er en workaround og kan skjule underliggende problemer i sjeldne tilfeller.
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import asyncio
import websockets
import json
import torch
import numpy as np
import argparse
from pathlib import Path
import signal
import matplotlib

matplotlib.use('Agg')  # Bytt til Agg backend FØR pyplot importeres for ikke-interaktiv plotting
import matplotlib.pyplot as plt
import random  # For random sleep

from .agent import A2CAgent  # Relativ import
from . import config  # Relativ import

# --- Global logging for plotting ---
TRAINING_STATS = {}  # {agent_id_str: {"game_rewards": [], "policy_losses": [], ...}}
PLOT_DIR = Path(__file__).resolve().parent / "training_plots_output"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

SHUTDOWN_FLAG = asyncio.Event()


def signal_handler_main(signum, frame):
    if not SHUTDOWN_FLAG.is_set():
        print("\n(Hovedkoordinator) Mottok avslutningssignal (Ctrl+C). Setter shutdown flagg...")
        SHUTDOWN_FLAG.set()
    else:
        print("(Hovedkoordinator) Avslutningssignal allerede mottatt...")


async def run_single_agent_session(agent_id_str: str, checkpoint_filename: str):
    uri = f"ws://{config.SERVER_HOST}:{config.SERVER_PORT}"
    agent_name = f"A2C_Agent_{agent_id_str}"
    a2c_agent = A2CAgent(agent_name=agent_name)

    if agent_id_str not in TRAINING_STATS:
        TRAINING_STATS[agent_id_str] = {
            "game_rewards_raw": [],  # Belønning for hvert spill
            "policy_loss_per_game": [],  # Gj.snitt policy loss for hvert spill
            "value_loss_per_game": [],  # Gj.snitt value loss for hvert spill
            "entropy_loss_per_game": [],  # Gj.snitt entropy loss for hvert spill
            "games_played_count": 0,
            "total_steps_across_games": 0
        }

    checkpoint_path = Path(__file__).resolve().parent / checkpoint_filename
    a2c_agent.load_model(checkpoint_path)

    connection_attempts = 0
    max_connection_attempts = 10  # Litt mer tålmodig

    while not SHUTDOWN_FLAG.is_set() and \
            TRAINING_STATS[agent_id_str]["games_played_count"] < config.NUM_GAMES_PER_AGENT_SESSION:

        # Buffere for ett enkelt spill
        temp_policy_losses = []
        temp_value_losses = []
        temp_entropy_losses = []
        current_game_step_rewards = []  # Belønninger for hvert steg i dette spillet

        a2c_agent.clear_buffers()  # Sørg for at agentens interne buffere er tomme

        current_game_id = None
        is_my_turn_flag = False
        am_i_eliminated_this_game = False
        last_known_state_for_learn = None
        steps_this_game = 0

        # print(f"[{agent_name}] Venter på nytt spill / prøver å koble til...")
        try:
            async with websockets.connect(uri, ping_interval=20, ping_timeout=30, open_timeout=15) as websocket:
                connection_attempts = 0
                await websocket.send(json.dumps({"type": "CONNECT", "nick": agent_name}))

                assign_id_msg_str = await asyncio.wait_for(websocket.recv(), timeout=10)
                assign_id_msg = json.loads(assign_id_msg_str)

                if assign_id_msg.get("type") == "ASSIGN_ID":
                    player_id = assign_id_msg.get("player_id")
                    if player_id is None:
                        print(f"[{agent_name}] FEIL: Fikk ikke player_id. Venter.")
                        await asyncio.sleep(random.uniform(3, 7))
                        continue
                    a2c_agent.set_player_id(player_id)
                else:
                    print(f"[{agent_name}] FEIL: Forventet ASSIGN_ID, fikk: {assign_id_msg}. Venter.")
                    await asyncio.sleep(random.uniform(3, 7))
                    continue

                # print(f"[{agent_name}] Tilkoblet. Min Player ID: {a2c_agent.player_id}.")

                async for message_str in websocket:
                    if SHUTDOWN_FLAG.is_set(): break
                    try:
                        msg = json.loads(message_str)
                        msg_type = msg.get("type")

                        if msg_type == "NEW_GAME":
                            new_game_id = msg.get("game_id")
                            # Hvis vi var midt i et spill, og det ikke ble fullført (ingen GAME_OVER/PLAYER_ELIMINATED)
                            # og vi har data, prøv å lære fra det (mindre ideelt, men bedre enn ingenting).
                            if current_game_id is not None and current_game_id != new_game_id and \
                                    not am_i_eliminated_this_game and a2c_agent.rewards_buffer:
                                print(
                                    f"[{agent_name}] Uventet NEW_GAME (ID: {new_game_id}) før GAME_OVER for spill {current_game_id}. Prøver å lære fra ufullstendig data.")
                                p_l, v_l, e_l = a2c_agent.learn(last_known_state_for_learn, True)  # Anta done=True
                                if p_l is not None:
                                    temp_policy_losses.append(p_l)
                                    temp_value_losses.append(v_l)
                                    temp_entropy_losses.append(e_l)
                                # Logg dette "ufullstendige" spillet
                                TRAINING_STATS[agent_id_str]['game_rewards_raw'].append(sum(current_game_step_rewards))
                                if temp_policy_losses: TRAINING_STATS[agent_id_str]['policy_loss_per_game'].append(
                                    np.mean(temp_policy_losses))
                                if temp_value_losses: TRAINING_STATS[agent_id_str]['value_loss_per_game'].append(
                                    np.mean(temp_value_losses))
                                if temp_entropy_losses: TRAINING_STATS[agent_id_str]['entropy_loss_per_game'].append(
                                    np.mean(temp_entropy_losses))
                                TRAINING_STATS[agent_id_str]["games_played_count"] += 1
                                TRAINING_STATS[agent_id_str]["total_steps_across_games"] += steps_this_game

                            current_game_id = new_game_id
                            # print(f"[{agent_name}] --- Nytt spill (ID: {current_game_id}) for P{a2c_agent.player_id} ---")
                            a2c_agent.clear_buffers()
                            temp_policy_losses, temp_value_losses, temp_entropy_losses, current_game_step_rewards = [], [], [], []
                            am_i_eliminated_this_game = False
                            is_my_turn_flag = False
                            last_known_state_for_learn = msg.get("state")
                            steps_this_game = 0

                        elif msg_type == "TURN_BEGIN":
                            if current_game_id is None: continue

                            game_state_for_action = msg.get("state")
                            last_known_state_for_learn = game_state_for_action

                            if msg.get("player_id") == a2c_agent.player_id and not am_i_eliminated_this_game:
                                is_my_turn_flag = True
                                steps_this_game += 1
                                if not game_state_for_action:
                                    action_to_send_obj = {"action": "stand"}
                                    a2c_agent.values_buffer.append(torch.tensor(0.0, device=config.DEVICE))
                                else:
                                    action_to_send_obj = a2c_agent.select_action(game_state_for_action)

                                await websocket.send(json.dumps({
                                    "type": "ACTION", "player_id": a2c_agent.player_id,
                                    "action": action_to_send_obj
                                }))
                            else:
                                is_my_turn_flag = False

                        elif msg_type == "TURN_RESULT":
                            if current_game_id is None: continue
                            last_known_state_for_learn = msg.get("state")
                            if msg.get("player_id") == a2c_agent.player_id and is_my_turn_flag:
                                reward = msg.get("reward", 0.0)
                                a2c_agent.store_reward(reward)
                                current_game_step_rewards.append(reward)
                            is_my_turn_flag = False

                        elif msg_type == "PLAYER_ELIMINATED":
                            if current_game_id is None: continue
                            elim_player_id = msg.get("player_id")
                            if elim_player_id == a2c_agent.player_id and not am_i_eliminated_this_game:
                                am_i_eliminated_this_game = True
                                if a2c_agent.rewards_buffer:  # Bare lær hvis det var noen handlinger
                                    p_l, v_l, e_l = a2c_agent.learn(last_known_state_for_learn, True)  # done=True
                                    if p_l is not None:
                                        temp_policy_losses.append(p_l)
                                        temp_value_losses.append(v_l)
                                        temp_entropy_losses.append(e_l)
                                else:
                                    a2c_agent.clear_buffers()

                        elif msg_type == "GAME_OVER":
                            if current_game_id is None: continue

                            final_state_for_learn = msg.get("final_state")
                            if not am_i_eliminated_this_game and a2c_agent.rewards_buffer:
                                p_l, v_l, e_l = a2c_agent.learn(final_state_for_learn, True)  # done=True
                                if p_l is not None:
                                    temp_policy_losses.append(p_l)
                                    temp_value_losses.append(v_l)
                                    temp_entropy_losses.append(e_l)
                            elif not am_i_eliminated_this_game and not a2c_agent.rewards_buffer:
                                a2c_agent.clear_buffers()

                            # Logg statistikk for det fullførte spillet
                            TRAINING_STATS[agent_id_str]['game_rewards_raw'].append(sum(current_game_step_rewards))
                            if temp_policy_losses: TRAINING_STATS[agent_id_str]['policy_loss_per_game'].append(
                                np.mean(temp_policy_losses))
                            if temp_value_losses: TRAINING_STATS[agent_id_str]['value_loss_per_game'].append(
                                np.mean(temp_value_losses))
                            if temp_entropy_losses: TRAINING_STATS[agent_id_str]['entropy_loss_per_game'].append(
                                np.mean(temp_entropy_losses))

                            TRAINING_STATS[agent_id_str]["games_played_count"] += 1
                            TRAINING_STATS[agent_id_str]["total_steps_across_games"] += steps_this_game

                            games_count_this_agent = TRAINING_STATS[agent_id_str]["games_played_count"]
                            print(
                                f"[{agent_name}] Spill {current_game_id} ferdig. Belønning: {sum(current_game_step_rewards):.2f}. (Agent totalt {games_count_this_agent} spill, {steps_this_game} steg i dette spillet)")

                            if games_count_this_agent > 0 and games_count_this_agent % config.SAVE_MODEL_EVERY_N_GAMES == 0:
                                a2c_agent.save_model(checkpoint_path)

                            if games_count_this_agent > 0 and games_count_this_agent % config.PLOT_STATS_EVERY_N_GAMES == 0:
                                plot_aggregated_training_results()

                            current_game_id = None  # Klar for neste NEW_GAME melding

                        elif msg_type == "TURN_END":
                            pass

                        elif msg_type == "ERROR":
                            print(f"[{agent_name}] FEIL fra server: {msg.get('msg')}")

                    except json.JSONDecodeError:
                        print(f"[{agent_name}] FEIL: Kunne ikke dekode JSON: {message_str}")
                    except websockets.exceptions.ConnectionClosed as e_ws_msg:
                        print(f"[{agent_name}] Tilkobling lukket (under melding): {e_ws_msg}.")
                        break
                    except Exception as e_inner:
                        print(f"[{agent_name}] FEIL i meldingsløkke: {type(e_inner).__name__} - {e_inner}")
                        break

            # Håndter disconnect utenom GAME_OVER (hvis data finnes og spillet var i gang)
            if current_game_id is not None and not am_i_eliminated_this_game and a2c_agent.rewards_buffer:
                print(f"[{agent_name}] Lærer fra ufullstendig spill {current_game_id} pga. disconnect.")
                p_l, v_l, e_l = a2c_agent.learn(last_known_state_for_learn, True)
                if p_l is not None:
                    temp_policy_losses.append(p_l)
                    temp_value_losses.append(v_l)
                    temp_entropy_losses.append(e_l)
                # Logg dette ufullstendige spillet også
                TRAINING_STATS[agent_id_str]['game_rewards_raw'].append(sum(current_game_step_rewards))
                if temp_policy_losses: TRAINING_STATS[agent_id_str]['policy_loss_per_game'].append(
                    np.mean(temp_policy_losses))
                if temp_value_losses: TRAINING_STATS[agent_id_str]['value_loss_per_game'].append(
                    np.mean(temp_value_losses))
                if temp_entropy_losses: TRAINING_STATS[agent_id_str]['entropy_loss_per_game'].append(
                    np.mean(temp_entropy_losses))
                TRAINING_STATS[agent_id_str]["games_played_count"] += 1
                TRAINING_STATS[agent_id_str]["total_steps_across_games"] += steps_this_game

            if SHUTDOWN_FLAG.is_set():
                print(f"[{agent_name}] Avslutter økt pga shutdown flagg.")
                break

        except (websockets.exceptions.WebSocketException, ConnectionRefusedError, asyncio.TimeoutError) as e_conn:
            connection_attempts += 1
            wait_time = min(2 ** connection_attempts, 60)
            print(f"[{agent_name}] Tilkoblingsfeil ({type(e_conn).__name__}). Prøver igjen om {wait_time}s.")
            if connection_attempts >= max_connection_attempts and not SHUTDOWN_FLAG.is_set():
                print(f"[{agent_name}] Maks antall tilkoblingsforsøk nådd. Avslutter denne agent-økten.")
                SHUTDOWN_FLAG.set()
                break
            try:
                await asyncio.sleep(wait_time)
            except asyncio.CancelledError:
                if SHUTDOWN_FLAG.is_set(): break
        except Exception as e_outer:
            print(
                f"[{agent_name}] Alvorlig FEIL (ytre løkke): {type(e_outer).__name__} - {e_outer}. Avslutter agent-økt.")
            # import traceback; traceback.print_exc()
            SHUTDOWN_FLAG.set()  # Signaliser at noe gikk galt
            break

    print(f"[{agent_name}] Økt ferdig. Totalt {TRAINING_STATS[agent_id_str]['games_played_count']} spill behandlet.")
    a2c_agent.save_model(checkpoint_path)


def plot_aggregated_training_results():
    try:
        num_agents_with_data = sum(1 for agent_id in TRAINING_STATS if TRAINING_STATS[agent_id]["game_rewards_raw"])
        if num_agents_with_data == 0:
            print("Ingen data å plotte for noen agenter.")
            return

        # Dynamisk bestem antall rader for subplots
        num_plot_rows = 0
        for agent_id_str in TRAINING_STATS:
            if TRAINING_STATS[agent_id_str]["game_rewards_raw"]:  # Bare tell agenter med data
                num_plot_rows += 1
        if num_plot_rows == 0: return  # Ingen data å plotte

        fig, axs = plt.subplots(num_plot_rows, 3, figsize=(20, 6 * num_plot_rows), squeeze=False)
        current_plot_row = 0

        for agent_id_str, stats in TRAINING_STATS.items():
            if not stats["game_rewards_raw"]:
                continue

            # Rewards
            rewards_raw = stats["game_rewards_raw"]
            policy_losses_pg = stats.get('policy_loss_per_game', [])
            value_losses_pg = stats.get('value_loss_per_game', [])

            axs[current_plot_row, 0].cla()
            axs[current_plot_row, 0].plot(rewards_raw, label='Belønning per spill', alpha=0.7, linestyle='-',
                                          marker='.', markersize=3)
            if len(rewards_raw) >= 10:
                window = min(max(10, len(rewards_raw) // 10), 100)
                avg_rewards = np.convolve(rewards_raw, np.ones(window) / window, mode='valid')
                axs[current_plot_row, 0].plot(np.arange(window - 1, len(rewards_raw)), avg_rewards,
                                              label=f'Gj.snitt ({window} spill)', color='red', lw=2)
            axs[current_plot_row, 0].set_title(f"Agent {agent_id_str} - Belønning")
            axs[current_plot_row, 0].set_xlabel("Spill #");
            axs[current_plot_row, 0].set_ylabel("Total Belønning")
            axs[current_plot_row, 0].legend();
            axs[current_plot_row, 0].grid(True)

            # Policy Loss
            axs[current_plot_row, 1].cla()
            if policy_losses_pg: axs[current_plot_row, 1].plot(policy_losses_pg, label='Policy Loss', marker='.',
                                                               markersize=3)
            axs[current_plot_row, 1].set_title(f"Agent {agent_id_str} - Policy Tap (per spill)")
            axs[current_plot_row, 1].set_xlabel("Spill #");
            axs[current_plot_row, 1].set_ylabel("Gj.snitt Policy Tap")
            axs[current_plot_row, 1].legend();
            axs[current_plot_row, 1].grid(True)

            # Value Loss
            axs[current_plot_row, 2].cla()
            if value_losses_pg: axs[current_plot_row, 2].plot(value_losses_pg, label='Value Loss', color='green',
                                                              marker='.', markersize=3)
            axs[current_plot_row, 2].set_title(f"Agent {agent_id_str} - Value Tap (per spill)")
            axs[current_plot_row, 2].set_xlabel("Spill #");
            axs[current_plot_row, 2].set_ylabel("Gj.snitt Value Tap")
            axs[current_plot_row, 2].legend();
            axs[current_plot_row, 2].grid(True)

            current_plot_row += 1

        plt.tight_layout(pad=2.5)  # Juster padding
        plot_filename = PLOT_DIR / f"aggregated_training_plots_a2c_game_avg.png"
        plt.savefig(plot_filename)
        print(f"Lagret/oppdatert aggregert treningsplot: {plot_filename}")
        plt.close(fig)

    except ImportError:
        print("Matplotlib ikke installert. Kan ikke plotte resultater.")
    except Exception as plot_e:
        print(f"Kunne ikke plotte resultater: {type(plot_e).__name__} - {plot_e}")
        # import traceback; traceback.print_exc()


async def main_coordinator():
    parser = argparse.ArgumentParser(description="Kjør A2C agenter for W.O.R.M.S.")
    parser.add_argument("--num_agents", type=int, default=1, choices=[1, 2],
                        help="Antall agenter å kjøre (1 eller 2). Server må støtte dette.")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal_handler_main)
    signal.signal(signal.SIGTERM, signal_handler_main)

    tasks = []
    if args.num_agents >= 1:
        print("(Koordinator) Forbereder Agent 1...")
        tasks.append(asyncio.create_task(run_single_agent_session(
            agent_id_str="1",
            checkpoint_filename="a2c_worms_agent1_checkpoint.pth"
        )))
    if args.num_agents == 2:
        print("(Koordinator) Forbereder Agent 2...")
        # Ingen sleep her, la dem starte så samtidig som mulig
        tasks.append(asyncio.create_task(run_single_agent_session(
            agent_id_str="2",
            checkpoint_filename="a2c_worms_agent2_checkpoint.pth"
        )))

    if not tasks:
        print("Ingen agenter spesifisert.")
        return

    print(f"(Koordinator) Starter {len(tasks)} agent-økt(er)...")
    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        print("(Koordinator) Hovedoppgave (gather) kansellert.")

    print("\n(Koordinator) Alle agent-økter er avsluttet eller avbrutt.")
    print("(Koordinator) Genererer endelige plott...")
    plot_aggregated_training_results()
    print("(Koordinator) Programmet avsluttes.")


if __name__ == "__main__":
    try:
        asyncio.run(main_coordinator())
    except KeyboardInterrupt:
        print("\n(Hovedprogram) KeyboardInterrupt fanget helt på slutten. Avslutter.")
        SHUTDOWN_FLAG.set()