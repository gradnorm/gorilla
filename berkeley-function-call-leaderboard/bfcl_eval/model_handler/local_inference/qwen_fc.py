import os
import json
import re
from copy import deepcopy
from typing import Any

from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler
from bfcl_eval.model_handler.utils import convert_to_function_call
from overrides import override


class QwenFCHandler(OSSHandler):
    VEHICLE_ENV_CONTEXT_FLAG = "ENABLE_VEHICLE_ENV_CONTEXT"

    def __init__(
        self,
        model_name,
        temperature,
        registry_name,
        is_fc_model,
        dtype="bfloat16",
        **kwargs,
    ) -> None:
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)
        self.model_name_huggingface = model_name

    @staticmethod
    def _render_file_system_context(initial_config: dict) -> str:
        """Render BFCL GorillaFileSystem initial state for the model prompt."""
        file_system = initial_config.get("GorillaFileSystem", {})
        root = file_system.get("root", {})
        if not root:
            return ""

        lines = ["<file_system>"]

        def append_node(path: str, node: dict) -> None:
            node_type = node.get("type")
            if node_type == "directory":
                lines.append(f"- {path}/")
                for name, child in node.get("contents", {}).items():
                    append_node(f"{path}/{name}".replace("//", "/"), child)
            elif node_type == "file":
                content = str(node.get("content", ""))
                preview = content.replace("\n", "\\n")
                if len(preview) > 200:
                    preview = preview[:200] + "..."
                lines.append(f'- {path}: "{preview}"')

        root_names = list(root.keys())
        for root_name in root_names:
            root_node = root[root_name]
            append_node(f"/{root_name}", root_node)

        # GorillaFileSystem starts at the loaded root directory.
        lines.append("")
        lines.append(f"Current directory: /{root_names[0]}")
        lines.append("</file_system>")
        return "\n".join(lines)

    @staticmethod
    def _render_trading_state_context(initial_config: dict) -> str:
        """Render compact BFCL TradingBot initial state for the model prompt."""
        trading_state = initial_config.get("TradingBot", {})
        if not trading_state:
            return ""

        account_info = trading_state.get("account_info", {})
        orders = trading_state.get("orders", {})
        watch_list = trading_state.get("watch_list", [])
        stocks = trading_state.get("stocks", {})

        def format_order(order_id: Any, order: Any) -> str | None:
            if not isinstance(order, dict):
                return None

            symbol = order.get("symbol", "unknown")
            status = order.get("status", "unknown")
            order_type = order.get("order_type", order.get("type", "unknown"))
            amount = order.get("amount", order.get("num_shares", order.get("shares")))
            price = order.get("price")

            details = [str(order_id), str(symbol), str(status)]
            if order_type != "unknown":
                details.append(str(order_type))
            if amount is not None:
                details.append(f"shares={amount}")
            if price is not None:
                details.append(f"price={price}")
            return ":".join(details)

        formatted_orders = [
            format_order(order_id, order)
            for order_id, order in orders.items()
            if isinstance(order, dict)
        ]
        formatted_orders = [order for order in formatted_orders if order]

        lines = ["<trading_state>"]
        lines.append(f"Authenticated: {trading_state.get('authenticated', 'unknown')}")
        lines.append(f"Market status: {trading_state.get('market_status', 'unknown')}")
        if "balance" in account_info:
            lines.append(f"Balance: {account_info['balance']}")
        if watch_list:
            lines.append(f"Watchlist: {', '.join(map(str, watch_list))}")
        else:
            lines.append("Watchlist: empty")
        if formatted_orders:
            lines.append(f"Orders: {', '.join(formatted_orders)}")
        else:
            lines.append("Orders: empty")
        if stocks:
            lines.append(f"Available stocks: {', '.join(sorted(map(str, stocks.keys())))}")
        lines.append("</trading_state>")
        return "\n".join(lines)

    @staticmethod
    def _render_vehicle_state_context(initial_config: dict) -> str:
        """Render compact BFCL VehicleControlAPI initial state for the model prompt."""
        vehicle_state = initial_config.get("VehicleControlAPI", {})
        if not vehicle_state:
            return ""

        door_status = vehicle_state.get("doorStatus", {})
        if isinstance(door_status, dict):
            doors = ", ".join(
                f"{door}:{status}" for door, status in door_status.items()
            )
            remaining_unlocked_doors = vehicle_state.get(
                "remainingUnlockedDoors",
                sum(1 for status in door_status.values() if status == "unlocked"),
            )
        else:
            doors = str(door_status)
            remaining_unlocked_doors = vehicle_state.get(
                "remainingUnlockedDoors", "unknown"
            )

        lines = ["<vehicle_state>"]
        lines.append(f"Engine: {vehicle_state.get('engineState', 'unknown')}")
        lines.append(f"Fuel level: {vehicle_state.get('fuelLevel', 'unknown')} gallons")
        lines.append(
            f"Battery voltage: {vehicle_state.get('batteryVoltage', 'unknown')} V"
        )
        lines.append(f"Doors: {doors}")
        lines.append(f"Remaining unlocked doors: {remaining_unlocked_doors}")
        lines.append(
            "Climate: "
            f"{vehicle_state.get('acTemperature', 'unknown')} C, "
            f"fan {vehicle_state.get('fanSpeed', 'unknown')}, "
            f"mode {vehicle_state.get('acMode', 'unknown')}, "
            f"humidity {vehicle_state.get('humidityLevel', 'unknown')}"
        )
        lines.append(f"Headlights: {vehicle_state.get('headLightStatus', 'unknown')}")
        lines.append(
            "Parking brake: "
            f"{vehicle_state.get('parkingBrakeStatus', 'unknown')}, "
            f"force {vehicle_state.get('parkingBrakeForce', 'unknown')}, "
            f"slope {vehicle_state.get('slopeAngle', 'unknown')}"
        )
        lines.append(
            "Brake pedal: "
            f"{vehicle_state.get('brakePedalStatus', 'unknown')}, "
            f"force {vehicle_state.get('brakePedalForce', 'unknown')}"
        )
        lines.append(
            "Cruise control: "
            f"{vehicle_state.get('cruiseStatus', 'unknown')}, "
            f"distance to next vehicle "
            f"{vehicle_state.get('distanceToNextVehicle', 'unknown')}"
        )
        lines.append(f"Destination: {vehicle_state.get('destination', 'unknown')}")
        lines.append(
            "Tire pressure: "
            f"front_left={vehicle_state.get('frontLeftTirePressure', 'unknown')}, "
            f"front_right={vehicle_state.get('frontRightTirePressure', 'unknown')}, "
            f"rear_left={vehicle_state.get('rearLeftTirePressure', 'unknown')}, "
            f"rear_right={vehicle_state.get('rearRightTirePressure', 'unknown')}"
        )
        lines.append("</vehicle_state>")
        return "\n".join(lines)

    @classmethod
    def _vehicle_env_context_enabled(cls) -> bool:
        value = os.getenv(cls.VEHICLE_ENV_CONTEXT_FLAG, "1").strip().lower()
        return value not in {"0", "false", "no", "off"}

    @classmethod
    def _render_environment_context(cls, initial_config: dict) -> str:
        blocks = [
            cls._render_file_system_context(initial_config),
            cls._render_trading_state_context(initial_config),
        ]
        if cls._vehicle_env_context_enabled():
            blocks.append(cls._render_vehicle_state_context(initial_config))
        blocks = [block for block in blocks if block]
        if not blocks:
            return ""

        return "<environment_context>\n" + "\n\n".join(blocks) + "\n</environment_context>"

    @override
    def decode_ast(self, result, language, has_tool_call_tag):
        # Model response is of the form:
        # "<tool_call>\n{\"name\": \"spotify.play\", \"arguments\": {\"artist\": \"Taylor Swift\", \"duration\": 20}}\n</tool_call>\n<tool_call>\n{\"name\": \"spotify.play\", \"arguments\": {\"artist\": \"Maroon 5\", \"duration\": 15}}\n</tool_call>"
        tool_calls = self._extract_tool_calls(result)
        if type(tool_calls) != list or any(type(item) != dict for item in tool_calls):
            raise ValueError(f"Model did not return a list of function calls: {result}")
        return [
            {call["name"]: {k: v for k, v in call["arguments"].items()}}
            for call in tool_calls
        ]

    @override
    def decode_execute(self, result, has_tool_call_tag):
        tool_calls = self._extract_tool_calls(result)
        if type(tool_calls) != list or any(type(item) != dict for item in tool_calls):
            raise ValueError(f"Model did not return a list of function calls: {result}")
        decoded_result = []
        for item in tool_calls:
            if type(item) == str:
                item = eval(item)
            decoded_result.append({item["name"]: item["arguments"]})
        return convert_to_function_call(decoded_result)

    @override
    def _format_prompt(self, messages, function):
        """
        "chat_template":
        {%- if tools %}
            {{- '<|im_start|>system\n' }}
            {%- if messages[0].role == 'system' %}
                {{- messages[0].content + '\n\n' }}
            {%- endif %}
            {{- "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
            {%- for tool in tools %}
                {{- "\n" }}
                {{- tool | tojson }}
            {%- endfor %}
            {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
        {%- else %}
            {%- if messages[0].role == 'system' %}
                {{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}
            {%- endif %}
        {%- endif %}
        {%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
        {%- for message in messages[::-1] %}
            {%- set index = (messages|length - 1) - loop.index0 %}
            {%- if ns.multi_step_tool and message.role == "user" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}
                {%- set ns.multi_step_tool = false %}
                {%- set ns.last_query_index = index %}
            {%- endif %}
        {%- endfor %}
        {%- for message in messages %}
            {%- if message.content is string %}
                {%- set content = message.content %}
            {%- else %}
                {%- set content = '' %}
            {%- endif %}
            {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
                {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>' + '\n' }}
            {%- elif message.role == "assistant" %}
                {%- set reasoning_content = '' %}
                {%- if message.reasoning_content is string %}
                    {%- set reasoning_content = message.reasoning_content %}
                {%- else %}
                    {%- if '</think>' in content %}
                        {%- set reasoning_content = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') %}
                        {%- set content = content.split('</think>')[-1].lstrip('\n') %}
                    {%- endif %}
                {%- endif %}
                {%- if loop.index0 > ns.last_query_index %}
                    {%- if loop.last or (not loop.last and reasoning_content) %}
                        {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n') }}
                    {%- else %}
                        {{- '<|im_start|>' + message.role + '\n' + content }}
                    {%- endif %}
                {%- else %}
                    {{- '<|im_start|>' + message.role + '\n' + content }}
                {%- endif %}
                {%- if message.tool_calls %}
                    {%- for tool_call in message.tool_calls %}
                        {%- if (loop.first and content) or (not loop.first) %}
                            {{- '\n' }}
                        {%- endif %}
                        {%- if tool_call.function %}
                            {%- set tool_call = tool_call.function %}
                        {%- endif %}
                        {{- '<tool_call>\n{"name": "' }}
                        {{- tool_call.name }}
                        {{- '", "arguments": ' }}
                        {%- if tool_call.arguments is string %}
                            {{- tool_call.arguments }}
                        {%- else %}
                            {{- tool_call.arguments | tojson }}
                        {%- endif %}
                        {{- '}\n</tool_call>' }}
                    {%- endfor %}
                {%- endif %}
                {{- '<|im_end|>\n' }}
            {%- elif message.role == "tool" %}
                {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}
                    {{- '<|im_start|>user' }}
                {%- endif %}
                {{- '\n<tool_response>\n' }}
                {{- content }}
                {{- '\n</tool_response>' }}
                {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
                    {{- '<|im_end|>\n' }}
                {%- endif %}
            {%- endif %}
        {%- endfor %}
        {%- if add_generation_prompt %}
            {{- '<|im_start|>assistant\n' }}
            {%- if enable_thinking is defined and enable_thinking is false %}
                {{- '<think>\n\n</think>\n\n' }}
            {%- endif %}
        {%- endif %}
        """
        formatted_prompt = ""

        if len(function) > 0:
            formatted_prompt += "<|im_start|>system\n"
            if messages[0]["role"] == "system":
                formatted_prompt += messages[0]["content"] + "\n\n"

            formatted_prompt += "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>"
            for tool in function:
                formatted_prompt += f"\n{json.dumps(tool)}"
            formatted_prompt += '\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call><|im_end|>\n'

        else:
            if messages[0]["role"] == "system":
                formatted_prompt += (
                    f"<|im_start|>system\n{messages[0]['content']}<|im_end|>\n"
                )

        last_query_index = len(messages) - 1
        for offset, message in enumerate(reversed(messages)):
            idx = len(messages) - 1 - offset
            if (
                message["role"] == "user"
                and type(message["content"]) == str
                and not (
                    message["content"].startswith("<tool_response>")
                    and message["content"].endswith("</tool_response>")
                )
            ):
                last_query_index = idx
                break

        for idx, message in enumerate(messages):
            role = message["role"]
            content = message["content"]

            if role == "user" or (role == "system" and idx != 0):
                formatted_prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"

            elif role == "assistant":
                reasoning_content = ""
                if "reasoning_content" in message and message["reasoning_content"]:
                    reasoning_content = message["reasoning_content"]

                elif "</think>" in content:
                    parts = content.split("</think>")
                    reasoning_content = (
                        parts[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
                    )
                    content = parts[-1].lstrip("\n")

                if idx > last_query_index:
                    if idx == len(messages) - 1 or reasoning_content:
                        formatted_prompt += (
                            f"<|im_start|>{role}\n<think>\n"
                            + reasoning_content.strip("\n")
                            + f"\n</think>\n\n"
                            + content.lstrip("\n")
                        )
                    else:
                        formatted_prompt += f"<|im_start|>{role}\n{content}"
                else:
                    formatted_prompt += f"<|im_start|>{role}\n{content}"
                    
                if "tool_calls" in message:
                    for tool_call in message["tool_calls"]:
                        if (tool_call == message["tool_calls"][0] and content) or tool_call != message["tool_calls"][0]:
                            formatted_prompt += "\n"
                        
                        if "function" in tool_call:
                            tool_call = tool_call["function"]
                        
                        arguments = tool_call.get("arguments", {})
                        formatted_prompt += '<tool_call>\n{"name": "'
                        formatted_prompt += tool_call["name"]
                        formatted_prompt += '", "arguments": '
                        
                        if isinstance(arguments, str):
                            formatted_prompt += arguments
                        else:
                            formatted_prompt += json.dumps(arguments)
                        
                        formatted_prompt += "}\n</tool_call>"

                formatted_prompt += "<|im_end|>\n"

            elif role == "tool":
                prev_role = messages[idx - 1]["role"] if idx > 0 else None
                next_role = messages[idx + 1]["role"] if idx < len(messages) - 1 else None

                if idx == 0 or prev_role != "tool":
                    formatted_prompt += "<|im_start|>user"

                formatted_prompt += f"\n<tool_response>\n{content}\n</tool_response>"

                if idx == len(messages) - 1 or next_role != "tool":
                    formatted_prompt += "<|im_end|>\n"

        formatted_prompt += "<|im_start|>assistant\n"
        return formatted_prompt

    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        environment_context = self._render_environment_context(
            test_entry.get("initial_config", {})
        )

        # FC models use its own system prompt, so no need to add any message

        return {
            "message": [],
            "function": functions,
            "environment_context": environment_context,
        }

    @override
    def add_first_turn_message_prompting(
        self, inference_data: dict, first_turn_message: list[dict]
    ) -> dict:
        messages = deepcopy(first_turn_message)
        environment_context = inference_data.get("environment_context", "")

        if environment_context:
            for message in messages:
                if message.get("role") == "user":
                    message["content"] = (
                        f"{environment_context}\n\n{message.get('content', '')}"
                    )
                    break

        inference_data["message"].extend(messages)
        return inference_data

    @override
    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        model_response = api_response.choices[0].text
        extracted_tool_calls = self._extract_tool_calls(model_response)

        reasoning_content = ""
        cleaned_response = model_response
        if "</think>" in model_response:
            parts = model_response.split("</think>")
            reasoning_content = parts[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
            cleaned_response = parts[-1].lstrip("\n")

        if len(extracted_tool_calls) > 0:
            model_responses_message_for_chat_history = {
                "role": "assistant",
                "content": "",
                "tool_calls": extracted_tool_calls,
            }

        else:
            model_responses_message_for_chat_history = {
                "role": "assistant",
                "content": cleaned_response,
            }
            
        model_responses_message_for_chat_history["reasoning_content"] = reasoning_content

        return {
            "model_responses": cleaned_response,
            "reasoning_content": reasoning_content,
            "model_responses_message_for_chat_history": model_responses_message_for_chat_history,
            "input_token": api_response.usage.prompt_tokens,
            "output_token": api_response.usage.completion_tokens,
        }

    @override
    def _add_assistant_message_prompting(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        inference_data["message"].append(
            model_response_data["model_responses_message_for_chat_history"],
        )
        return inference_data

    @staticmethod
    def _extract_tool_calls(input_string):
        pattern = r"<tool_call>\n(.*?)\n</tool_call>"
        matches = re.findall(pattern, input_string, re.DOTALL)

        # Process matches into a list of dictionaries
        result = []
        for match in matches:
            try:
                match = json.loads(match)
                if isinstance(match, dict) and "name" in match:
                    match.setdefault("arguments", {})
                result.append(match)
            except Exception as e:
                pass
        return result
