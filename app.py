from dotenv import load_dotenv
load_dotenv()
import json
import copy
import re
from glob import glob

import gradio as gr

from chatarena.arena import Arena
from chatarena.ui.mafia_session import sessions
from chatarena.backends import BACKEND_REGISTRY
from chatarena.config import ArenaConfig
from chatarena.environments import ENV_REGISTRY

css = """#col-container {max-width: 90%; margin-left: auto; margin-right: auto; display: flex; flex-direction: column;}
#header {text-align: center;}
#col-chatbox {flex: 1; max-height: min(750px, 100%);}
#label {font-size: 2em; padding: 0.5em; margin: 0;}
.message {font-size: 1.2em;}
.message-wrap {max-height: min(700px, 100vh);}
"""
# .wrap {min-width: min(640px, 100vh)}
# #env-desc {max-height: 100px; overflow-y: auto;}
# .textarea {height: 100px; max-height: 100px;}
# #chatbot-tab-all {height: 750px; max-height: min(750px, 100%);}
# #chatbox {height: min(750px, 100%); max-height: min(750px, 100%);}
# #chatbox.block {height: 730px}
# .wrap {max-height: 680px;}
# .scroll-hide {overflow-y: scroll; max-height: 100px;}


DEBUG = False

DEFAULT_BACKEND = "openai-chat"
DEFAULT_ENV = "conversation"
MAX_NUM_PLAYERS = 6
DEFAULT_NUM_PLAYERS = 2


def load_examples():
    example_configs = {}
    # Load json config files from examples folder
    example_files = glob("examples/*.json")
    for example_file in example_files:
        with open(example_file, encoding="utf-8") as f:
            example = json.load(f)
            try:
                example_configs[example["name"]] = example
            except KeyError:
                print(f"Example {example_file} is missing a name field. Skipping.")
    return example_configs


EXAMPLE_REGISTRY = load_examples()


def get_moderator_components(visible=True):
    name = "Moderator"
    with gr.Row():
        with gr.Column():
            role_desc = gr.Textbox(
                label="Moderator role",
                lines=1,
                visible=visible,
                interactive=True,
                placeholder=f"Enter the role description for {name}",
            )
            terminal_condition = gr.Textbox(
                show_label=False,
                lines=1,
                visible=visible,
                interactive=True,
                placeholder="Enter the termination criteria",
            )
        with gr.Column():
            backend_type = gr.Dropdown(
                show_label=False,
                visible=visible,
                interactive=True,
                choices=list(BACKEND_REGISTRY.keys()),
                value=DEFAULT_BACKEND,
            )
            with gr.Accordion(
                f"{name} Parameters", open=False, visible=visible
            ) as accordion:
                temperature = gr.Slider(
                    minimum=0,
                    maximum=2.0,
                    step=0.1,
                    interactive=True,
                    visible=visible,
                    label="temperature",
                    value=0.7,
                )
                max_tokens = gr.Slider(
                    minimum=10,
                    maximum=500,
                    step=10,
                    interactive=True,
                    visible=visible,
                    label="max tokens",
                    value=200,
                )

    return [
        role_desc,
        terminal_condition,
        backend_type,
        accordion,
        temperature,
        max_tokens,
    ]


def get_player_components(name, visible):
    with gr.Row():
        with gr.Column():
            role_name = gr.Textbox(
                lines=1,
                show_label=False,
                interactive=True,
                visible=visible,
                placeholder=f"Player name for {name}",
            )
            role_desc = gr.Textbox(
                lines=3,
                show_label=False,
                interactive=True,
                visible=visible,
                placeholder=f"Enter the role description for {name}",
            )
        with gr.Column():
            backend_type = gr.Dropdown(
                show_label=False,
                choices=list(BACKEND_REGISTRY.keys()),
                interactive=True,
                visible=visible,
                value=DEFAULT_BACKEND,
            )
            with gr.Accordion(
                f"{name} Parameters", open=False, visible=visible
            ) as accordion:
                temperature = gr.Slider(
                    minimum=0,
                    maximum=2.0,
                    step=0.1,
                    interactive=True,
                    visible=visible,
                    label="temperature",
                    value=0.7,
                )
                max_tokens = gr.Slider(
                    minimum=10,
                    maximum=500,
                    step=10,
                    interactive=True,
                    visible=visible,
                    label="max tokens",
                    value=200,
                )

    return [role_name, role_desc, backend_type, accordion, temperature, max_tokens]


def get_empty_state():
    return gr.State({"session_id": None})


with gr.Blocks(css=css) as demo:
    state = get_empty_state()
    all_components = []

    with gr.Column(elem_id="col-container"):
        gr.Markdown(
            """# 🏟 ChatArena️<br>
Prompting multiple AI agents to play games in a language-driven environment.
**[Project Homepage](https://github.com/chatarena/chatarena)**""",
            elem_id="header",
        )

        with gr.Row():
            env_selector = gr.Dropdown(
                choices=list(ENV_REGISTRY.keys()),
                value=DEFAULT_ENV,
                interactive=True,
                label="Environment Type",
                show_label=True,
            )
            example_selector = gr.Dropdown(
                choices=list(EXAMPLE_REGISTRY.keys()),
                interactive=True,
                label="Select Example",
                show_label=True,
            )

        # Environment configuration
        env_desc_textbox = gr.Textbox(
            show_label=True,
            lines=2,
            visible=True,
            label="Environment Description",
            placeholder="Enter a description of a scenario or the game rules.",
        )

        all_components += [env_selector, example_selector, env_desc_textbox]

        with gr.Row():
            with gr.Column(elem_id="col-chatbox"):
                with gr.Tab("All", visible=True):
                    chatbot = gr.Chatbot(
                        elem_id="chatbox", visible=True, show_label=False
                    )

                player_chatbots = []
                for i in range(MAX_NUM_PLAYERS):
                    player_name = f"Player {i + 1}"
                    with gr.Tab(player_name, visible=(i < DEFAULT_NUM_PLAYERS)):
                        player_chatbot = gr.Chatbot(
                            elem_id=f"chatbox-{i}",
                            visible=i < DEFAULT_NUM_PLAYERS,
                            label=player_name,
                            show_label=False,
                        )
                        player_chatbots.append(player_chatbot)

            all_components += [chatbot, *player_chatbots]

            with gr.Column(elem_id="col-config"):  # Player Configuration
                # gr.Markdown("Player Configuration")
                parallel_checkbox = gr.Checkbox(
                    label="Parallel Actions", value=False, visible=True
                )
                with gr.Accordion("Moderator", open=False, visible=True):
                    moderator_components = get_moderator_components(True)
                all_components += [parallel_checkbox, *moderator_components]

                all_players_components, players_idx2comp = [], {}
                with gr.Blocks():
                    num_player_slider = gr.Slider(
                        2,
                        MAX_NUM_PLAYERS,
                        value=DEFAULT_NUM_PLAYERS,
                        step=1,
                        label="Number of players:",
                    )
                    for i in range(MAX_NUM_PLAYERS):
                        player_name = f"Player {i + 1}"
                        with gr.Tab(
                            player_name, visible=(i < DEFAULT_NUM_PLAYERS)
                        ) as tab:
                            player_comps = get_player_components(
                                player_name, visible=(i < DEFAULT_NUM_PLAYERS)
                            )

                        players_idx2comp[i] = player_comps + [tab]
                        all_players_components += player_comps + [tab]

                all_components += [num_player_slider] + all_players_components

                def variable_players(k):
                    k = int(k)
                    update_dict = {}
                    for i in range(MAX_NUM_PLAYERS):
                        if i < k:
                            for comp in players_idx2comp[i]:
                                update_dict[comp] = gr.update(visible=True)
                            update_dict[player_chatbots[i]] = gr.update(visible=True)
                        else:
                            for comp in players_idx2comp[i]:
                                update_dict[comp] = gr.update(visible=False)
                            update_dict[player_chatbots[i]] = gr.update(visible=False)
                    return update_dict

                num_player_slider.change(
                    variable_players,
                    num_player_slider,
                    all_players_components + player_chatbots,
                )

                human_input_textbox = gr.Textbox(
                    show_label=True,
                    label="Human Input",
                    lines=1,
                    visible=True,
                    interactive=True,
                    placeholder="Enter your input here",
                )
                with gr.Row():
                    btn_step = gr.Button("Start")
                    btn_pause = gr.Button("Pause")
                    btn_send = gr.Button("Send")
                    btn_restart = gr.Button("Clear")
                live_status = gr.Markdown("Select an example, then Start. Mafia runs automatically.")
                mafia_moderator_enabled = gr.Checkbox(label="Use LLM discussion moderator (Mafia)", value=False)
                mafia_options = gr.Textbox(label="Mafia discussion settings (JSON)",
                    value='{"max_discussion_messages": 24, "max_intent_rounds": 48, "discussion_seconds": 180, "silence_seconds": 10}',
                    lines=3)
                all_components += [mafia_moderator_enabled, mafia_options]

                all_components += [human_input_textbox, btn_step, btn_restart]

    def _convert_to_chatbot_output(all_messages, display_recv=False):
        chatbot_output = []
        for i, message in enumerate(all_messages):
            agent_name, msg, recv = (
                message.agent_name,
                message.content,
                str(message.visible_to),
            )
            new_msg = re.sub(
                r"\n+", "<br>", msg.strip()
            )  # Preprocess message for chatbot output
            if display_recv:
                new_msg = f"**{agent_name} (-> {recv})**: {new_msg}"  # Add role to the message
            else:
                new_msg = f"**{agent_name}**: {new_msg}"

            if agent_name == "Moderator":
                chatbot_output.append((new_msg, None))
            else:
                chatbot_output.append((None, new_msg))
        return chatbot_output

    def _create_arena_config_from_components(all_comps: dict) -> ArenaConfig:
        env_desc = all_comps[env_desc_textbox]

        # Initialize the players
        num_players = int(all_comps[num_player_slider])
        selected_example = EXAMPLE_REGISTRY.get(all_comps.get(example_selector), {})
        player_configs = []
        for i in range(num_players):
            role_name, role_desc, backend_type, temperature, max_tokens = (
                all_comps[c]
                for c in players_idx2comp[i]
                if not isinstance(c, (gr.Accordion, gr.Tab))
            )
            player_config = {
                "name": role_name,
                "role_desc": role_desc,
                "global_prompt": env_desc,
                "backend": {
                    "backend_type": backend_type,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
            }
            # Preserve model names and backend-specific options from examples.
            example_players = selected_example.get("players", [])
            if i < len(example_players):
                backend_config = copy.deepcopy(example_players[i].get("backend", {}))
                backend_config.update(player_config["backend"])
                player_config["backend"] = backend_config
            player_configs.append(player_config)

        # Initialize the environment
        env_type = all_comps[env_selector]
        # Get moderator config
        (
            mod_role_desc,
            mod_terminal_condition,
            moderator_backend_type,
            mod_temp,
            mod_max_tokens,
        ) = (
            all_comps[c]
            for c in moderator_components
            if not isinstance(c, (gr.Accordion, gr.Tab))
        )
        moderator_config = {
            "role_desc": mod_role_desc,
            "global_prompt": env_desc,
            "terminal_condition": mod_terminal_condition,
            "backend": {
                "backend_type": moderator_backend_type,
                "temperature": mod_temp,
                "max_tokens": mod_max_tokens,
            },
        }
        env_config = {
            "env_type": env_type,
            "parallel": all_comps[parallel_checkbox],
            "moderator": moderator_config,
            "moderator_visibility": "all",
            "moderator_period": None,
        }

        if env_type == "mafia":
            env_config = copy.deepcopy(selected_example.get("environment", {}))
            env_config["env_type"] = "mafia"
            env_config.pop("discussion_rounds", None)
            env_config.pop("moderator", None)
            options = json.loads(all_comps[mafia_options] or "{}")
            allowed = {"max_discussion_messages", "max_intent_rounds", "discussion_seconds",
                       "silence_seconds", "intent_weights", "repeat_speaker_factor", "seed"}
            if not isinstance(options, dict) or set(options) - allowed:
                raise ValueError("Unknown Mafia discussion setting")
            env_config.update(options)
            saved_moderator = env_config.get("discussion_moderator") or {}
            saved_backend = saved_moderator.get("backend", {})
            if saved_backend.get("backend_type") == moderator_backend_type:
                merged_backend = copy.deepcopy(saved_backend)
                merged_backend.update(moderator_config["backend"])
                moderator_config["backend"] = merged_backend
            env_config["discussion_moderator"] = (
                moderator_config if all_comps[mafia_moderator_enabled] else None
            )
        return ArenaConfig(players=player_configs, environment=env_config, global_prompt=env_desc)

    def render_session(session):
        snap = session.snapshot()
        label = f"Playing as {snap['human']}" if snap["human"] else "Spectator"
        result = {
            chatbot: _convert_to_chatbot_output(snap["messages"], display_recv=not snap["human"]),
            live_status: f"{label} · {snap['status']}" + (f" — {snap['error']}" if snap["error"] else ""),
            btn_step: gr.update(
                value=("Running" if snap["status"] in ("running", "waiting_human", "waiting_capacity")
                       else "Resume") if session.controller else "Next Step",
                interactive=not snap["terminal"] and (not session.controller or snap["status"] in ("paused", "error"))),
        }
        for i, player in enumerate(session.arena.players):
            if i < len(player_chatbots):
                result[player_chatbots[i]] = _convert_to_chatbot_output(snap["views"].get(player.name, []))
        return result

    def step_game(all_comps):
        current = all_comps[state]
        session = sessions.get(current.get("session_id"))
        if session is None:
            arena = Arena.from_config(_create_arena_config_from_components(all_comps))
            key = sessions.add(arena)
            current = {"session_id": key}
            session = sessions.get(key)
        session.start()
        result = render_session(session)
        result[state] = current
        return result

    def refresh_game(current):
        session = sessions.get(current.get("session_id"))
        return render_session(session) if session else {comp: gr.skip() for comp in live_outputs}

    def pause_game(current):
        session = sessions.get(current.get("session_id"))
        if session:
            session.pause()
            return render_session(session)
        return {comp: gr.skip() for comp in live_outputs}

    def send_message(current, text):
        session = sessions.get(current.get("session_id"))
        if session is None:
            return {live_status: "Start the game first."}
        try:
            session.submit(text)
        except ValueError as exc:
            return {live_status: str(exc)}
        # The polling callback owns chat rendering, avoiding old send snapshots.
        return {human_input_textbox: ""}

    def restart_game(current):
        sessions.remove(current.get("session_id"))
        return {state: {"session_id": None}, chatbot: [],
                **{comp: [] for comp in player_chatbots},
                btn_step: gr.update(value="Start", interactive=True),
                live_status: "Cleared. Start creates a new game."}

    # Remove Accordion and Tab from the list of components
    all_components = [
        comp for comp in all_components if not isinstance(comp, (gr.Accordion, gr.Tab))
    ]

    live_outputs = [chatbot, *player_chatbots, live_status, btn_step]
    btn_step.click(step_game, set(all_components + [state]), live_outputs + [state])
    btn_pause.click(pause_game, state, live_outputs, queue=False)
    btn_restart.click(restart_game, state, live_outputs + [state], queue=False)
    btn_send.click(send_message, [state, human_input_textbox],
                   [human_input_textbox, live_status], queue=False)
    human_input_textbox.submit(send_message, [state, human_input_textbox],
                               [human_input_textbox, live_status], queue=False)
    poll_timer = gr.Timer(0.5)
    poll_timer.tick(refresh_game, state, live_outputs, queue=False)

    # If an example is selected, update the components
    def update_components_from_example(all_comps: dict):
        example_name = all_comps[example_selector]
        example_config = EXAMPLE_REGISTRY[example_name]
        update_dict = {}

        # Update the environment components
        env_config = example_config["environment"]
        update_dict[env_desc_textbox] = gr.update(value=example_config["global_prompt"])
        update_dict[env_selector] = gr.update(value=env_config["env_type"])
        update_dict[parallel_checkbox] = gr.update(value=env_config.get("parallel", False))
        option_names = ("max_discussion_messages", "max_intent_rounds", "discussion_seconds",
                        "silence_seconds", "intent_weights", "repeat_speaker_factor", "seed")
        update_dict[mafia_options] = json.dumps({k: env_config[k] for k in option_names if k in env_config})
        update_dict[mafia_moderator_enabled] = bool(env_config.get("discussion_moderator"))

        # Update the moderator components
        if "moderator" in env_config or env_config.get("discussion_moderator"):
            env_config = dict(env_config, moderator=env_config.get("discussion_moderator") or env_config["moderator"])
            (
                mod_role_desc,
                mod_terminal_condition,
                moderator_backend_type,
                mod_temp,
                mod_max_tokens,
            ) = (
                c
                for c in moderator_components
                if not isinstance(c, (gr.Accordion, gr.Tab))
            )
            update_dict[mod_role_desc] = gr.update(
                value=env_config["moderator"]["role_desc"]
            )
            update_dict[mod_terminal_condition] = gr.update(
                value=env_config["moderator"].get("terminal_condition", "")
            )
            update_dict[moderator_backend_type] = gr.update(
                value=env_config["moderator"]["backend"]["backend_type"]
            )
            update_dict[mod_temp] = gr.update(
                value=env_config["moderator"]["backend"].get("temperature", 0.3)
            )
            update_dict[mod_max_tokens] = gr.update(
                value=env_config["moderator"]["backend"].get("max_tokens", 100)
            )

        # Update the player components
        update_dict[num_player_slider] = gr.update(value=len(example_config["players"]))
        for i, player_config in enumerate(example_config["players"]):
            role_name, role_desc, backend_type, temperature, max_tokens = (
                c
                for c in players_idx2comp[i]
                if not isinstance(c, (gr.Accordion, gr.Tab))
            )

            update_dict[role_name] = gr.update(value=player_config["name"])
            update_dict[role_desc] = gr.update(value=player_config["role_desc"])
            update_dict[backend_type] = gr.update(
                value=player_config["backend"]["backend_type"]
            )
            update_dict[temperature] = gr.update(
                value=player_config["backend"]["temperature"]
            )
            update_dict[max_tokens] = gr.update(
                value=player_config["backend"]["max_tokens"]
            )

        return update_dict

    example_selector.change(
        update_components_from_example,
        set(all_components + [state]),
        all_components + [state],
    )

if __name__ == "__main__":
    demo.queue()
    demo.launch(debug=DEBUG, server_port=8080)
