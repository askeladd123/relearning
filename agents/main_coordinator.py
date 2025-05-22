# agents/main_coordinator.py
import os

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

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random

# Felles config og utils
from agents.common import config
from agents.common.utils import format_action, \
    preprocess_state  # Selv om agenten gjør dette internt, greit å ha for ev. debug

# Import agent klasser
from agents.a2c_manual.agent import A2CAgent
from agents.ppo_manual.agent import PPOAgent

# --- Global logging for plotting ---
TRAINING_STATS = {}
PLOT_DIR = Path(__file__).resolve().parent / "training_plots_output"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
SHUTDOWN_FLAG = asyncio.Event()


def signal_handler_main(signum, frame):
    if not SHUTDOWN_FLAG.is_set():
        print("\n(Hovedkoordinator) Mottok avslutningssignal. Setter shutdown flagg...")
        SHUTDOWN_FLAG.set()
    else:
        print("(Hovedkoordinator) Avslutningssignal allerede mottatt.")


async def run_single_agent_session(agent_id_str: str, agent_type: str, checkpoint_filename: str):
    uri = f"ws://{config.SERVER_HOST}:{config.SERVER_PORT}"
    agent_name = f"{agent_type.upper()}_Agent_{agent_id_str}"

    if agent_type == 'a2c':
        agent = A2CAgent(agent_name=agent_name)
    elif agent_type == 'ppo':
        agent = PPOAgent(agent_name=agent_name)
    else:
        raise ValueError(f"Ukjent agent_type: {agent_type}")

    if agent_id_str not in TRAINING_STATS:
        TRAINING_STATS[agent_id_str] = {
            "agent_type": agent_type,  # Nytt for plotting
            "game_rewards_raw": [],
            "policy_loss_per_game": [],
            "value_loss_per_game": [],
            "entropy_loss_per_game": [],
            "games_played_count": 0,
            "total_steps_across_games": 0
        }

    checkpoint_path = Path(__file__).resolve().parent / f"{agent_type}_manual" / checkpoint_filename
    agent.load_model(checkpoint_path)

    connection_attempts = 0
    max_connection_attempts = 10

    while not SHUTDOWN_FLAG.is_set() and \
            TRAINING_STATS[agent_id_str]["games_played_count"] < config.NUM_GAMES_PER_AGENT_SESSION:

        temp_policy_losses_for_game = []
        temp_value_losses_for_game = []
        temp_entropy_losses_for_game = []
        current_game_step_rewards = []

        if agent_type == 'a2c':
            agent.clear_buffers()  # A2C tømmer internt etter learn()
        # PPO tømmer bufferne (og batch_count) når learn() faktisk kjører en oppdatering

        current_game_id = None
        is_my_turn_flag = False
        am_i_eliminated_this_game = False
        last_known_state_for_learn = None
        steps_this_game = 0

        try:
            async with websockets.connect(uri, ping_interval=20, ping_timeout=30, open_timeout=15) as websocket:
                connection_attempts = 0
                await websocket.send(json.dumps({"type": "CONNECT", "nick": agent_name}))

                assign_id_msg_str = await asyncio.wait_for(websocket.recv(), timeout=10)
                assign_id_msg = json.loads(assign_id_msg_str)

                if assign_id_msg.get("type") == "ASSIGN_ID":
                    player_id = assign_id_msg.get("player_id")
                    if player_id is None:
                        print(f"[{agent_name}] FEIL: Fikk ikke player_id.")
                        await asyncio.sleep(random.uniform(3, 7));
                        continue
                    agent.set_player_id(player_id)
                else:
                    print(f"[{agent_name}] FEIL: Forventet ASSIGN_ID, fikk {assign_id_msg}.")
                    await asyncio.sleep(random.uniform(3, 7));
                    continue

                async for message_str in websocket:
                    if SHUTDOWN_FLAG.is_set(): break
                    try:
                        msg = json.loads(message_str)
                        msg_type = msg.get("type")

                        if msg_type == "NEW_GAME":
                            new_game_id = msg.get("game_id")
                            if current_game_id is not None and current_game_id != new_game_id and \
                                    not am_i_eliminated_this_game:
                                if agent_type == 'a2c' and agent.rewards_buffer:
                                    p_l, v_l, e_l = agent.learn(last_known_state_for_learn, True)
                                    if p_l is not None:
                                        temp_policy_losses_for_game.append(p_l)
                                        temp_value_losses_for_game.append(v_l)
                                        temp_entropy_losses_for_game.append(e_l)
                                elif agent_type == 'ppo' and agent.batch_count > 0:  # PPO might not have enough for a batch
                                    # For PPO, vi må tvinge en læring selv om batchen ikke er full,
                                    # siden spillet slutter uventet.
                                    # La oss si at PPO's learn håndterer dette ved å enten lære på det den har,
                                    # eller bare tømme bufferne hvis det er for lite.
                                    # Dette er en forenkling. Ideelt sett burde PPO lære når batchen er full.
                                    # Men for GAME_OVER-lignende eventer, må vi håndtere det.
                                    # Vi setter ikke done=True her, da learn() for PPO sjekker dones_buffer
                                    p_l, v_l, e_l = agent.learn(
                                        last_known_state_for_learn)  # PPO uses internal dones_buffer
                                    if p_l is not None:
                                        temp_policy_losses_for_game.append(p_l)
                                        temp_value_losses_for_game.append(v_l)
                                        temp_entropy_losses_for_game.append(e_l)

                                if temp_policy_losses_for_game:  # Logg ufullstendig spill
                                    TRAINING_STATS[agent_id_str]['game_rewards_raw'].append(
                                        sum(current_game_step_rewards))
                                    TRAINING_STATS[agent_id_str]['policy_loss_per_game'].append(
                                        np.mean(temp_policy_losses_for_game))
                                    TRAINING_STATS[agent_id_str]['value_loss_per_game'].append(
                                        np.mean(temp_value_losses_for_game))
                                    TRAINING_STATS[agent_id_str]['entropy_loss_per_game'].append(
                                        np.mean(temp_entropy_losses_for_game))
                                    TRAINING_STATS[agent_id_str]["games_played_count"] += 1
                                    TRAINING_STATS[agent_id_str]["total_steps_across_games"] += steps_this_game

                            current_game_id = new_game_id
                            if agent_type == 'a2c': agent.clear_buffers()
                            # PPO tømmer når batchen er full i learn()
                            temp_policy_losses_for_game, temp_value_losses_for_game, temp_entropy_losses_for_game, current_game_step_rewards = [], [], [], []
                            am_i_eliminated_this_game = False
                            is_my_turn_flag = False
                            last_known_state_for_learn = msg.get("state")
                            steps_this_game = 0

                        elif msg_type == "TURN_BEGIN":
                            if current_game_id is None: continue
                            game_state_for_action = msg.get("state")
                            last_known_state_for_learn = game_state_for_action

                            if msg.get("player_id") == agent.player_id and not am_i_eliminated_this_game:
                                is_my_turn_flag = True
                                steps_this_game += 1
                                if not game_state_for_action:  # Skulle ikke skje hvis orm er i live
                                    action_to_send_obj = {"action": "stand"}
                                    if agent_type == 'a2c': agent.values_buffer.append(
                                        torch.tensor(0.0, device=config.DEVICE))
                                    # For PPO, håndteres dette i select_action
                                else:
                                    action_to_send_obj = agent.select_action(game_state_for_action)

                                await websocket.send(json.dumps({
                                    "type": "ACTION", "player_id": agent.player_id,
                                    "action": action_to_send_obj
                                }))
                            else:
                                is_my_turn_flag = False

                        elif msg_type == "TURN_RESULT":
                            if current_game_id is None: continue
                            last_known_state_for_learn = msg.get("state")
                            if msg.get("player_id") == agent.player_id and is_my_turn_flag:
                                reward = msg.get("reward", 0.0)
                                current_game_step_rewards.append(reward)
                                # For A2C, done er per episode. For PPO, done er per steg.
                                # Serveren sender ikke 'done' i TURN_RESULT. Vi antar 'done=False' her for PPO.
                                # A2C venter på PLAYER_ELIMINATED eller GAME_OVER for 'done'.
                                if agent_type == 'a2c':
                                    agent.store_reward(reward)
                                elif agent_type == 'ppo':
                                    # Anta at ormen fortsatt er i live med mindre ELIMINATED eller GAME_OVER kommer
                                    is_my_worm_alive_now = any(
                                        w['id'] == (agent.player_id - 1) and w['health'] > 0 for w in
                                        last_known_state_for_learn.get('worms', []))
                                    step_done_for_ppo = not is_my_worm_alive_now
                                    agent.store_reward_and_done(reward, step_done_for_ppo)

                                    # PPO lærer per batch
                                    if agent.batch_count >= (
                                    config.PPO_BATCH_SIZE if hasattr(config, 'PPO_BATCH_SIZE') else 128):
                                        p_l, v_l, e_l = agent.learn(last_known_state_for_learn)
                                        if p_l is not None:
                                            temp_policy_losses_for_game.append(p_l)
                                            temp_value_losses_for_game.append(v_l)
                                            temp_entropy_losses_for_game.append(e_l)
                            is_my_turn_flag = False

                        elif msg_type == "PLAYER_ELIMINATED":
                            if current_game_id is None: continue
                            elim_player_id = msg.get("player_id")
                            if elim_player_id == agent.player_id and not am_i_eliminated_this_game:
                                am_i_eliminated_this_game = True
                                if agent_type == 'a2c':
                                    if agent.rewards_buffer:
                                        p_l, v_l, e_l = agent.learn(last_known_state_for_learn, True)  # done=True
                                        if p_l is not None:
                                            temp_policy_losses_for_game.append(p_l)
                                            temp_value_losses_for_game.append(v_l)
                                            temp_entropy_losses_for_game.append(e_l)
                                    else:
                                        agent.clear_buffers()
                                elif agent_type == 'ppo':
                                    # Siste 'done' for PPO ble satt i TURN_RESULT (hvis helse <=0).
                                    # Hvis det er data igjen i batchen, prøv å lære.
                                    if agent.batch_count > 0:
                                        p_l, v_l, e_l = agent.learn(last_known_state_for_learn)
                                        if p_l is not None:
                                            temp_policy_losses_for_game.append(p_l)
                                            temp_value_losses_for_game.append(v_l)
                                            temp_entropy_losses_for_game.append(e_l)
                                    agent.clear_buffers_and_count()  # PPO tømmer etter learn

                        elif msg_type == "GAME_OVER":
                            if current_game_id is None: continue
                            final_state_for_learn = msg.get("final_state")

                            if not am_i_eliminated_this_game:  # Hvis jeg fortsatt var i live
                                if agent_type == 'a2c':
                                    if agent.rewards_buffer:
                                        p_l, v_l, e_l = agent.learn(final_state_for_learn, True)
                                        if p_l is not None:
                                            temp_policy_losses_for_game.append(p_l)
                                            temp_value_losses_for_game.append(v_l)
                                            temp_entropy_losses_for_game.append(e_l)
                                    else:
                                        agent.clear_buffers()
                                elif agent_type == 'ppo':
                                    # PPO: Siste 'done' ble satt i siste TURN_RESULT. Lær hvis batch er full.
                                    if agent.batch_count > 0:
                                        p_l, v_l, e_l = agent.learn(final_state_for_learn)
                                        if p_l is not None:
                                            temp_policy_losses_for_game.append(p_l)
                                            temp_value_losses_for_game.append(v_l)
                                            temp_entropy_losses_for_game.append(e_l)
                                    agent.clear_buffers_and_count()

                            # Logg statistikk for det fullførte spillet
                            TRAINING_STATS[agent_id_str]['game_rewards_raw'].append(sum(current_game_step_rewards))
                            if temp_policy_losses_for_game: TRAINING_STATS[agent_id_str]['policy_loss_per_game'].append(
                                np.mean(temp_policy_losses_for_game))
                            if temp_value_losses_for_game: TRAINING_STATS[agent_id_str]['value_loss_per_game'].append(
                                np.mean(temp_value_losses_for_game))
                            if temp_entropy_losses_for_game: TRAINING_STATS[agent_id_str][
                                'entropy_loss_per_game'].append(np.mean(temp_entropy_losses_for_game))

                            TRAINING_STATS[agent_id_str]["games_played_count"] += 1
                            TRAINING_STATS[agent_id_str]["total_steps_across_games"] += steps_this_game

                            games_count_this_agent = TRAINING_STATS[agent_id_str]["games_played_count"]
                            print(
                                f"[{agent_name}] Spill {current_game_id} ferdig. Belønning: {sum(current_game_step_rewards):.2f}. (Agent totalt {games_count_this_agent} spill, {steps_this_game} steg)")

                            if games_count_this_agent > 0 and games_count_this_agent % config.SAVE_MODEL_EVERY_N_GAMES == 0:
                                agent.save_model(checkpoint_path)

                            if games_count_this_agent > 0 and games_count_this_agent % config.PLOT_STATS_EVERY_N_GAMES == 0:
                                plot_aggregated_training_results()

                            current_game_id = None  # Klar for neste NEW_GAME

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
                        # import traceback; traceback.print_exc() # For dypere debug
                        break

            # Håndter disconnect utenom GAME_OVER (hvis data finnes og spillet var i gang)
            if current_game_id is not None and not am_i_eliminated_this_game:
                if agent_type == 'a2c' and agent.rewards_buffer:
                    print(f"[{agent_name}] Lærer fra ufullstendig spill {current_game_id} (A2C) pga. disconnect.")
                    p_l, v_l, e_l = agent.learn(last_known_state_for_learn, True)
                    if p_l is not None: temp_policy_losses_for_game.append(p_l); temp_value_losses_for_game.append(
                        v_l); temp_entropy_losses_for_game.append(e_l)
                elif agent_type == 'ppo' and agent.batch_count > 0:
                    print(f"[{agent_name}] Lærer fra ufullstendig spill {current_game_id} (PPO) pga. disconnect.")
                    p_l, v_l, e_l = agent.learn(last_known_state_for_learn)
                    if p_l is not None: temp_policy_losses_for_game.append(p_l); temp_value_losses_for_game.append(
                        v_l); temp_entropy_losses_for_game.append(e_l)

                if temp_policy_losses_for_game:  # Logg dette ufullstendige spillet også
                    TRAINING_STATS[agent_id_str]['game_rewards_raw'].append(sum(current_game_step_rewards))
                    TRAINING_STATS[agent_id_str]['policy_loss_per_game'].append(np.mean(temp_policy_losses_for_game))
                    TRAINING_STATS[agent_id_str]['value_loss_per_game'].append(np.mean(temp_value_losses_for_game))
                    TRAINING_STATS[agent_id_str]['entropy_loss_per_game'].append(np.mean(temp_entropy_losses_for_game))
                    TRAINING_STATS[agent_id_str]["games_played_count"] += 1
                    TRAINING_STATS[agent_id_str]["total_steps_across_games"] += steps_this_game

            if SHUTDOWN_FLAG.is_set():
                print(f"[{agent_name}] Avslutter økt pga shutdown flagg.")
                break

            # Websocket-tilkobling brutt, vent og prøv igjen
            connection_attempts += 1
            wait_time = min(2 ** connection_attempts, 60)  # Exponential backoff
            print(
                f"[{agent_name}] Tilkoblingsfeil. Prøver igjen om {wait_time}s. (Forsøk {connection_attempts}/{max_connection_attempts})")
            if connection_attempts >= max_connection_attempts and not SHUTDOWN_FLAG.is_set():
                print(f"[{agent_name}] Maks antall tilkoblingsforsøk nådd. Avslutter denne agent-økten.")
                SHUTDOWN_FLAG.set();
                break
            try:
                await asyncio.sleep(wait_time)
            except asyncio.CancelledError:
                if SHUTDOWN_FLAG.is_set(): break

        except Exception as e_outer:
            print(
                f"[{agent_name}] Alvorlig FEIL (ytre løkke): {type(e_outer).__name__} - {e_outer}. Avslutter agent-økt.")
            # import traceback; traceback.print_exc()
            SHUTDOWN_FLAG.set();
            break

    print(f"[{agent_name}] Økt ferdig. Totalt {TRAINING_STATS[agent_id_str]['games_played_count']} spill behandlet.")
    agent.save_model(checkpoint_path)


def plot_aggregated_training_results():
    try:
        num_agents_with_data = sum(1 for agent_id in TRAINING_STATS if TRAINING_STATS[agent_id]["game_rewards_raw"])
        if num_agents_with_data == 0:
            print("Ingen data å plotte for noen agenter.")
            return

        num_plot_rows = num_agents_with_data
        fig, axs = plt.subplots(num_plot_rows, 4, figsize=(24, 6 * num_plot_rows), squeeze=False)  # Lagt til entropy
        current_plot_row = 0

        for agent_id_str, stats in TRAINING_STATS.items():
            if not stats["game_rewards_raw"]:
                continue

            agent_type_label = stats.get("agent_type", "Ukjent").upper()

            # Rewards
            rewards_raw = stats["game_rewards_raw"]
            axs[current_plot_row, 0].cla()
            axs[current_plot_row, 0].plot(rewards_raw, label='Belønning per spill', alpha=0.7, linestyle='-',
                                          marker='.', markersize=3)
            if len(rewards_raw) >= 10:
                window = min(max(10, len(rewards_raw) // 10), 100)
                avg_rewards = np.convolve(rewards_raw, np.ones(window) / window, mode='valid')
                axs[current_plot_row, 0].plot(np.arange(window - 1, len(rewards_raw)), avg_rewards,
                                              label=f'Gj.snitt ({window} spill)', color='red', lw=2)
            axs[current_plot_row, 0].set_title(f"Agent {agent_id_str} ({agent_type_label}) - Belønning")
            axs[current_plot_row, 0].set_xlabel("Spill #");
            axs[current_plot_row, 0].set_ylabel("Total Belønning")
            axs[current_plot_row, 0].legend();
            axs[current_plot_row, 0].grid(True)

            # Policy Loss
            policy_losses_pg = stats.get('policy_loss_per_game', [])
            axs[current_plot_row, 1].cla()
            if policy_losses_pg: axs[current_plot_row, 1].plot(policy_losses_pg, label='Policy Loss', marker='.',
                                                               markersize=3)
            axs[current_plot_row, 1].set_title(f"Agent {agent_id_str} ({agent_type_label}) - Policy Tap")
            axs[current_plot_row, 1].set_xlabel("Spill #");
            axs[current_plot_row, 1].set_ylabel("Gj.snitt Policy Tap")
            axs[current_plot_row, 1].legend();
            axs[current_plot_row, 1].grid(True)

            # Value Loss
            value_losses_pg = stats.get('value_loss_per_game', [])
            axs[current_plot_row, 2].cla()
            if value_losses_pg: axs[current_plot_row, 2].plot(value_losses_pg, label='Value Loss', color='green',
                                                              marker='.', markersize=3)
            axs[current_plot_row, 2].set_title(f"Agent {agent_id_str} ({agent_type_label}) - Value Tap")
            axs[current_plot_row, 2].set_xlabel("Spill #");
            axs[current_plot_row, 2].set_ylabel("Gj.snitt Value Tap")
            axs[current_plot_row, 2].legend();
            axs[current_plot_row, 2].grid(True)

            # Entropy Loss
            entropy_losses_pg = stats.get('entropy_loss_per_game', [])
            axs[current_plot_row, 3].cla()
            if entropy_losses_pg: axs[current_plot_row, 3].plot(entropy_losses_pg, label='Entropy Loss', color='purple',
                                                                marker='.', markersize=3)
            axs[current_plot_row, 3].set_title(f"Agent {agent_id_str} ({agent_type_label}) - Entropy Tap")
            axs[current_plot_row, 3].set_xlabel("Spill #");
            axs[current_plot_row, 3].set_ylabel("Gj.snitt Entropy Tap")
            axs[current_plot_row, 3].legend();
            axs[current_plot_row, 3].grid(True)

            current_plot_row += 1

        plt.tight_layout(pad=2.5)
        plot_filename = PLOT_DIR / f"aggregated_training_plots_multialgo.png"
        plt.savefig(plot_filename)
        print(f"Lagret/oppdatert aggregert treningsplot: {plot_filename}")
        plt.close(fig)

    except ImportError:
        print("Matplotlib ikke installert.")
    except Exception as plot_e:
        print(f"Kunne ikke plotte: {type(plot_e).__name__} - {plot_e}")


async def main():
    parser = argparse.ArgumentParser(description="Kjør RL agenter for W.O.R.M.S.")
    parser.add_argument("--num_agents", type=int, default=1, choices=[1, 2],
                        help="Antall agenter å kjøre (1 eller 2).")
    parser.add_argument("--agent1_type", type=str, default="a2c", choices=["a2c", "ppo"],
                        help="Type for agent 1 (a2c eller ppo).")
    parser.add_argument("--agent2_type", type=str, default="ppo", choices=["a2c", "ppo"],
                        help="Type for agent 2 (a2c eller ppo), hvis num_agents=2.")

    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal_handler_main)
    signal.signal(signal.SIGTERM, signal_handler_main)

    tasks = []
    if args.num_agents >= 1:
        print(f"(Koordinator) Forbereder Agent 1 ({args.agent1_type.upper()})...")
        tasks.append(asyncio.create_task(run_single_agent_session(
            agent_id_str="1",
            agent_type=args.agent1_type,
            checkpoint_filename=f"{args.agent1_type}_worms_agent1_checkpoint.pth"
        )))

    if args.num_agents == 2:
        print(f"(Koordinator) Forbereder Agent 2 ({args.agent2_type.upper()})...")
        tasks.append(asyncio.create_task(run_single_agent_session(
            agent_id_str="2",
            agent_type=args.agent2_type,
            checkpoint_filename=f"{args.agent2_type}_worms_agent2_checkpoint.pth"
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
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n(Hovedprogram) KeyboardInterrupt. Avslutter.")
        SHUTDOWN_FLAG.set()