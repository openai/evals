import copy
import functools
import inspect
import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, OrderedDict, Text, Tuple

from evals.base import PluginSpec
from evals.plugin.specification import Api, KwargsFunctionSignature, Param, comment_content
from evals.prompt.base import OpenAICreateChatPrompt, is_chat_prompt
from evals.registry import registry
from evals.utils.misc import make_object

API_NAMESPACE = "_api_namespace"
API_NAMESPACE_DESCRIPTION = "_api_namespace_description"

API_NAME = "_api_name"
API_DESCRIPTION = "_api_description"
API_ARGS = "_api_args"
API_ARGS_TYPE = "type"
API_ARGS_OPTIONAL = "optional"
API_ARGS_DESCRIPTION = "description"
API_ARGS_DEFAULT = "default"

NAMESPACE_VALIDATION_REGEX = re.compile("[a-zA-Z]+")
API_DESCRIPTION_VALIDATION_REGEX = re.compile("[a-zA-Z0-9_]+")


def namespace(name, description: Optional[Text] = None) -> Callable[..., Any]:
    if NAMESPACE_VALIDATION_REGEX.fullmatch(name) is None:
        raise ValueError("Namespace must be a string of a-zA-Z_ characters")

    def decorator(cls) -> Any:
        setattr(cls, API_NAMESPACE, name)

        if description is None:
            setattr(cls, API_NAMESPACE_DESCRIPTION, "")
        else:
            setattr(cls, API_NAMESPACE_DESCRIPTION, description)
        return cls

    return decorator


def api_description(
    name,
    description,
    api_args: Optional[dict] = None,
    custom_encoder: Optional[json.JSONEncoder] = None,
) -> Callable[..., Any]:
    if API_DESCRIPTION_VALIDATION_REGEX.fullmatch(name) is None:
        raise ValueError("API name must be a string of a-zA-Z0-9_ characters")

    def decorator(func) -> Callable[..., Any]:
        if isinstance(func, classmethod):
            inner_func = func.__func__
        else:
            inner_func = func

        @functools.wraps(inner_func)
        def wrapper(*args, **kwargs) -> Any:
            return inner_func(*args, **kwargs)

        setattr(wrapper, API_NAME, name)
        setattr(wrapper, API_DESCRIPTION, description)

        # Validate API args
        if api_args is not None:
            for _, arg_info in api_args.items():
                if not isinstance(arg_info, dict):
                    raise ValueError("api_args must be a dictionary of dictionaries")
                if API_ARGS_TYPE not in arg_info:
                    raise ValueError(f"api_args must contain a '{API_ARGS_TYPE}' key")

        setattr(wrapper, API_ARGS, api_args)

        if isinstance(func, classmethod):
            return classmethod(wrapper)
        else:
            return wrapper

    return decorator


@dataclass
class PluginAction:
    """
    A class representing an action that can be performed by a plugin.

    Attributes:
        namespace (Text): The namespace of the plugin.
        endpoint (Text): The endpoint of the action.
        content (dict): A dictionary containing the data an action will be called with.
        include_in_conversation (bool): Whether or not to include the action in the conversation transcript.
    """

    namespace: Text
    endpoint: Text
    content: dict

    include_in_conversation: bool = True

    def invocation_message(self) -> dict:
        if not self.include_in_conversation:
            raise ValueError(
                "Cannot include action in conversation if include_in_conversation is False"
            )

        return {
            "role": "assistant",
            "recipient": f"{self.namespace}.{self.endpoint}",
            "content": json.dumps(self.content),
        }

    def load(json_data: dict) -> "PluginAction":
        return PluginAction(
            namespace=json_data["namespace"],
            endpoint=json_data["endpoint"],
            content=json_data["content"],
        )

    @classmethod
    def load_all(self, json_data: dict) -> List["PluginAction"]:
        actions = []
        for action in json_data:
            actions.append(PluginAction.load(action))
        return actions

    def metadata(self) -> dict:
        return {
            "namespace": self.namespace,
            "endpoint": self.endpoint,
            "content": self.content,
            "include_in_conversation": self.include_in_conversation,
        }


class Plugin:
    @staticmethod
    def parse_function(func: Text) -> Tuple[Text, Text]:
        if "." not in func:
            raise ValueError("Function must be namespaced")

        namespace, api_name = func.split(".")
        return namespace, api_name

    @staticmethod
    def load(plugins: Optional[List[Text]]) -> List["Plugin"]:
        if plugins is None or len(plugins) == 0:
            return []

        plugin_specifications: List[PluginSpec] = _retrieve_plugin_specifications(plugins)
        enabled_plugins: OrderedDict[Text, Plugin] = _create_ordered_plugin_instances(
            plugin_specifications
        )
        return list(enabled_plugins.values())

    @staticmethod
    def invoke(function, function_args: Optional[Dict[Text, Any]] = None) -> Dict[Text, Text]:
        source_clazz: Plugin = function.__self__

        # Validate the incoming function a bit
        namespace = getattr(source_clazz, API_NAMESPACE, None)
        api_name = getattr(function, API_NAME, None)
        if not namespace:
            raise ValueError(
                "Function must be from a class namespaced via the @namespace annotation"
            )

        if not api_name:
            raise ValueError("Function must be annotated with @api_description")

        # Partially apply the args, which should result in a function that takes no arguments
        if function_args is not None:
            function: Callable[..., Text] = functools.partial(function, **function_args)

        # Get the plugin response
        plugin_response_content = function()
        plugin_response = source_clazz.convert_response(plugin_response_content)

        assert isinstance(
            plugin_response, str
        ), f"Plugin response must be a string, not {type(plugin_response)}"

        # Build the tool response
        tool_response = {
            "role": "tool",
            "name": f"{namespace}.{api_name}",
            "content": plugin_response,
        }

        return tool_response

    @classmethod
    def namespace(cls) -> Text:
        return cls._api_namespace

    @classmethod
    def description(cls) -> Text:
        # Add namespace description and opening brace if namespace exists
        namespace = getattr(cls, API_NAMESPACE, None)
        if namespace is None:
            raise ValueError(
                "Class must be namespaced via the @namespace annotation to have a description"
            )
        # Order by the method name to ensure consistent output
        ordered_members = []
        for _, method in inspect.getmembers(cls, predicate=inspect.isroutine):
            api_name = getattr(method, API_NAME, None)
            if api_name is not None:
                ordered_members.append((api_name, method))
        ordered_members.sort(key=lambda x: x[0])

        # Construct the API payload
        signatures = []
        for api_name, method in ordered_members:
            params: List[Param] = []
            args = getattr(method, API_ARGS, {})
            if args is None:
                args = {}

            for arg_name, args in args.items():
                param = Param(
                    name=arg_name,
                    type=args[API_ARGS_TYPE],
                    required=not args.get(API_ARGS_OPTIONAL, False),
                    description=args.get(API_ARGS_DESCRIPTION, None),
                    default=args.get(API_ARGS_DEFAULT, None),
                )
                params.append(param)
            f = KwargsFunctionSignature(
                name=api_name, description=getattr(method, API_DESCRIPTION, None), params=params
            )
            signatures.append(f)

        api = Api(
            namespace=namespace,
            description_for_model=getattr(cls, API_NAMESPACE_DESCRIPTION, None),
            signatures=signatures,
        )

        return f"{comment_content(api.description_for_model)}{api.to_typescript()}"

    def convert_response(self, response_data: dict) -> Text:
        return json.dumps(response_data)

    def get_method_by_name(self, api_name: Text) -> Callable[..., Any]:
        # Iterate over class methods and return the method with a matching API name
        for _, method in inspect.getmembers(self, predicate=inspect.isroutine):
            method_api_name: Text = getattr(method, API_NAME, None)
            if method_api_name is not None and method_api_name == api_name:
                return method

        # Raise an error if the API is not found
        raise ValueError(f"Plugin '{self.namespace()} does not have API '{api_name}'")

    def metadata(self) -> dict:
        return {
            "namespace": self.namespace(),
        }


@dataclass
class PluginEvaluationContext:
    original_prompt: OpenAICreateChatPrompt
    updated_prompt: OpenAICreateChatPrompt
    actions: List[PluginAction]


def _retrieve_plugin_specifications(plugins: Optional[List[Text]]) -> List[PluginSpec]:
    plugin_specifications: List[PluginSpec] = []
    for plugin in plugins:
        plugin_spec: Optional[PluginSpec] = registry.get_plugin(plugin)
        if plugin_spec is None:
            raise ValueError(f"Unknown plugin '{plugin}' - not found in registry.")
        plugin_specifications.append(plugin_spec)
    return plugin_specifications


def _create_ordered_plugin_instances(
    plugin_specifications: List[PluginSpec],
) -> OrderedDict[Text, Plugin]:
    enabled_plugins = OrderedDict()
    for plugin_spec in plugin_specifications:
        plugin: Plugin = make_object(plugin_spec.cls, plugin_spec.args, {})()
        namespace = plugin.namespace()

        if namespace in enabled_plugins:
            raise ValueError(
                f"Duplicate plugin namespace not allowed, found '{namespace}' twice when instantiating the following plugins: {plugins}."
            )

        enabled_plugins[namespace] = plugin
    return enabled_plugins


def _invoke_plugin(
    invocation_message: Dict[Text, Text], enabled_plugins: Dict[Text, Plugin]
) -> Dict[Text, Text]:
    invocation_message_content: Optional[Text] = invocation_message.get("content")
    recipient: Optional[Text] = invocation_message.get("recipient")

    assert (
        invocation_message_content is not None
    ), f"Plugin invocation requires content - {invocation_message}"
    assert (
        recipient is not None
    ), f"Plugin invocation requires recipient to know which plugin to invoke - {invocation_message}"

    namespace, function_name = Plugin.parse_function(recipient)
    assert namespace in enabled_plugins, f"Plugin {namespace} not found"

    selected_plugin: Plugin = enabled_plugins[namespace]
    plugin_function = selected_plugin.get_method_by_name(function_name)

    try:
        function_args = json.loads(invocation_message_content)
    except Exception as e:
        raise ValueError(
            f"Could not parse plugin invocation content: {invocation_message_content}"
        ) from e

    plugin_response: Dict[Text, Text] = Plugin.invoke(plugin_function, function_args)
    return plugin_response


def evaluate_prompt_with_plugins(
    prompt: OpenAICreateChatPrompt,
    plugins: Optional[List[Plugin]],
) -> PluginEvaluationContext:
    # A list of all the PluginAction events called during this evaluation
    executed_actions = []

    # If users provided one or more plugins, we need to:
    # 1. Add the plugin descriptions to the system message
    # 2. Evaluate the last message to see if it's a plugin message
    #    - If it is, we need to call that plugin to obtain the result prior to sending to the model
    if plugins is None or len(plugins) == 0:
        return PluginEvaluationContext(
            original_prompt=prompt,
            updated_prompt=prompt,
            actions=executed_actions,
        )

    assert is_chat_prompt(
        prompt
    ), f"Plugin use requires a chat-style prompt.\n\nAttempted to use plugins: {plugins} with prompt: {prompt}"
    assert (
        len(prompt) > 0
    ), f"Plugin use requires a non-empty prompt.\n\nAttempted to use plugins: {plugins} with prompt: {prompt}"

    # Copy the prompt so that we don't modify something the user assumes is static
    result: OpenAICreateChatPrompt = [copy.deepcopy(message) for message in prompt]

    enabled_plugins: OrderedDict[Text, Plugin] = dict(
        [(plugin.namespace(), plugin) for plugin in plugins]
    )

    first_message = result[0]
    # Setup plugins system message
    plugins_system_message = "\n\n".join(
        [plugin.description() for plugin in enabled_plugins.values()]
    )
    if first_message["role"] == "system":
        first_message["content"] += f"\n\n{plugins_system_message}"
    else:
        message = {
            "role": "system",
            "content": plugins_system_message,
        }
        result.insert(0, message)

    # Evaluate each of the actions.  This does not currently support
    # assistant responses to plugin actions at each step
    # TODO: Plugin Actions
    # Look for role=plugin, try to load a PluginAction
    output = []
    for message in result:
        updated_messages = []

        if message["role"] == "plugin":
            action = PluginAction.load(message)
            executed_actions.append(action)
            invocation_message: Dict[Text, Text] = action.invocation_message()

            response: Dict[Text, Text] = _invoke_plugin(
                invocation_message=invocation_message, enabled_plugins=enabled_plugins
            )

            if action.include_in_conversation:
                # Update all the messages
                updated_messages.append(invocation_message)
                updated_messages.append(response)
        else:
            updated_messages.append(message)

        output.extend(updated_messages)

    return PluginEvaluationContext(
        original_prompt=prompt,
        updated_prompt=output,
        actions=executed_actions,
    )
