# Plugins

## Overview

Creating and evaluating AI plugins can be a challenging task, as they may interact with the environment, collaborate with other plugins, and provide domain-specific knowledge to the AI model. Moreover, the AI model should be able to use the plugin effectively under various constraints. This plugin evaluation framework aims to simplify the process of creating, evaluating, and integrating AI plugins, ensuring a balance between realism and ease of evaluation.

This framework is designed for local evaluation of AI plugins, and it's not intended for deploying Language Model plugins in production. The primary goal is to assist developers, researchers, and contributors in creating and evaluating custom plugins for their use cases, as well as understanding how the AI model responds to the installation of new plugins.

## Getting Started

To create a new plugin, follow these steps:

1. Familiarize yourself with the example plugins found in the `openai` folder, such as `openai/bird.py`, `openai/clock.py`, and `openai/text_analyzer.py`. These examples will help you understand the structure and requirements for creating a new plugin.

2. Follow the plugin creation process detailed below.

3. Submit a pull request with your new plugin and evaluation example(s).

## Preparing Samples

Evaluations with plugins enabled take an additional arguement - `plugins` - which is a list of plugins to enable. The evaluation framework will automatically instantiate the plugins from the registry for each sampling.  Please note, we currently require chat-style inputs for plugins.

Before sending a request to the model, we will iterate through the `input` messages, invoking the relevant plugins.  One plugin instance will be used per sample, so plugins will retain state throughout that request.

NOTE: By enabling plugins, the input list may be mutated.  Currently, we append the plugin's description to the end of the system message (Or insert a new system message if none was provided). This is done to ensure that the model has context to know how to use the plugin.

```JSON
{
  "plugins": [
    "openai.bird.1",
    "openai.clock.1",
  ],
  "ideal": [
    "Red-tailed hawk"
  ],
  "input": [
    {
      "role": "system",
      "content": "Answer the following questions as concisely as possible."
    },
    {
      "role": "user",
      "content": "Can you list all of the birds in my birds list which are in the Accipitridae family?"
    },
    {
      "role": "plugin",
      "namespace": "MyBirds",
      "endpoint": "listBirds",
      "content": {}
    }
  ]
}
```

## Plugin Creation

### Namespace and API Description

- Use the `@namespace` decorator to define your plugin's namespace and provide a brief description. The AI model will use this namespace to encapsulate your plugin. The description will help the model understand the goals, purposes, and limitations of your plugin.

- For each API method in your plugin, use the `@api_description` decorator to define the API's name, description, and arguments (if any). The AI model will use this information to understand the purpose of each API method and the inputs it expects. The description will also provide context for the function.

### Input and Output

- Ensure your plugin methods accept inputs as defined in the `api_args` parameter of the `@api_description` decorator.

- The output of your plugin methods should be in a format that can be easily consumed by the AI model.

### Adding to Registry

- To add a plugin to the registry, create a new YAML file in the `registry/plugins` directory under the same directory structure and name as your plugin Python file.

- The naming convention for a new registry entity is `<namespace>.<plugin_name>.<version>`. Feel free to be creative with the namespace and plugin name. The version is a monotonically increasing integer, starting at 1. For example, `openai.bird.1`.

- The YAML file should contain a description of how to instantiate your plugin. A few examples can be found at `registry/plugins/openai/example.yaml`.

### Evaluation Examples

- Create evaluation examples that involve your custom plugin. Plugins should be enabled on a per-example basis, allowing us to group examples of how the plugin changes the model's behavior with examples that might not use the plugin at all but are related requests.

- Model-graded evaluations tend to work better than string-based evaluations.

- Make sure your evaluation passes the `plugins` dictionary to the sampling command. This will ensure that the model is aware of your plugin and can use it during local evaluation.

- It is best-practice to test the behavior of the model when your plugin is not installed. This helps effectively communicate the value of your plugin when combined with the model.

- We've found evaluating the appropriateness of plugin use, the model's ability to recover from inappropriate or adversarial responses, and the ability for the model to integrate the plugin's knowledge into its own understanding of the world to be important evaluation dimensions.  You may find other dimensions to be important as well.

## Extending Existing Plugins

- Do not modify existing plugins. Instead, create a new plugin that extends the existing plugin using the plugin creation process above. When creating this new plugin, increase the version number of the plugin you are extending by 1.

- Plugins can be forked by changing the namespace, but this should be done with care. We cannot allow two plugins with the same namespace to exist in practice.

- Fixing bugs in existing plugins may be done if necessary, but please be thoughtful if the bug fix meaningfully changes the plugin's behavior. Not all plugins will be perfect, and evaluating our model's ability to recover from buggy behavior is an important part of the evaluation process.

## Evaluation Workflows

There are several factors to consider when assessing a model's performance using plugins.  We welcome additional contributions to this section, as it is not exhaustive.

### Plugin State Population

The method by which a plugin is populated with data is an important consideration when evaluating the model's behavior. There are two main ways to populate a plugin:

#### Externally Populated

In this scenario, the plugin is pre-populated with data outside of the current conversation. This allows the evaluation of the model's ability to rely on the plugin to obtain information about the state of the world. For example, a calendar plugin could be pre-populated with a meeting, and the model would then be asked to provide details about that meeting.

Assuming a plugin named `MyCalendar` with `addMeeting` and `listMeetings` endpoints, the following input would be used to populate the plugin externally.  Setting the `include_in_conversation` flag to `false` will prevent the `addMeeting` action from being recorded in the conversation, allowing the plugin to act as if it was externally populated.  By default, this flag is `true`.

```JSON
{
  "plugins": [
    "openai.calendar.1"
  ],
  "ideal": [
    "You have a meeting with the CEO at 2pm."
  ],
  "input": [
    {
      "role": "plugin",
      "namespace": "MyCalendar",
      "endpoint": "addMeeting",
      "content": {"time": "2pm", "title": "Meeting with CEO"},
      "include_in_conversation": false
    },
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "What is my next meeting?"
    },
    {
      "role": "plugin",
      "namespace": "MyCalendar",
      "endpoint": "listMeetings",
      "content": {}
    }
  ]
}
```

#### Internally Populated

If a plugin is updated within the same conversation as the evaluation, it has been internally populated. In this scenario, the model's behavior is evaluated based on its ability to accurately query the plugin for updated information rather than relying on its assumed state of the plugin. 

For instance, after adding two items to a todolist, the model should still use the todolist plugin to check the entire day's schedule instead of relying on the conversation state.  Below, we have an example of a plugin with some externally populated state ("Call a friend"), as well as two internally populated items - getting milk and picking up the dry cleaning.

The below example is a bit limited, as our local plugins code currently does not support asking the assistant which action should be next.

```JSON
{
  "plugins": [
    "openai.todo.1"
  ],
  "ideal": [
    "The list should include getting milk, picking up the dry cleaning, and calling a friend"
  ],
  "input": [
    {
      "role": "plugin",
      "namespace": "TodoList",
      "endpoint": "addItem",
      "content": {"name": "Call a friend"},
      "include_in_conversation": false
    },
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "Can you add 'Get milk' to my todo list?"
    },
    {
      "role": "plugin",
      "namespace": "TodoList",
      "endpoint": "addItem",
      "content": {"name": "Get milk"},
    },
    {
      "role": "user",
      "content": "Can you add 'Pickup the dry cleaning' to my todo list?"
    },
    {
      "role": "plugin",
      "namespace": "TodoList",
      "endpoint": "addItem",
      "content": {"name": "Pickup the dry cleaning"},
    },
    {
      "role": "user",
      "content": "What items are on my todo list?"
    },
    {
      "role": "plugin",
      "namespace": "TodoList",
      "endpoint": "listItems",
      "content": {}
    }
  ]
}
```

### Evaluation Workflows

The interaction between the user and the model is an important consideration when evaluating the model's behavior. We've outlined a few interesting workflows below.

#### User Workflow

Users will sometimes specify a dependent sequence of actions to be executed by the AI model. The evaluation focuses on whether the model invoked the correct plugin endpoints and if the final result was accurate. For example, a user might ask the model to add two items to a list and then display all the items in the list.

#### Agentic Workflow

As AI models are increasingly being used for more autonomous tasks, we aim to evaluate their ability to accomplish a single, high-level task. Here, the user assigns a task to the model, and the model is expected to perform the necessary steps to complete it using relevant plugins and information. For example, a user might ask the model to "set up a vacation," and the model would then organize the vacation details accordingly.

#### Multi-Party Workflow

This workflow evaluates the model's ability to interact and communicate effectively with multiple users or other models concurrently. The assessment focuses on whether the parties are able to collaborate and whether a model is able to use the information provided by the other parties to accomplish its task. For example, a user might ask the model to "schedule a meeting with Bob about graham cracker production", and the model would then use the relevant plugins and information to coordinate the meeting, possibly by querying Bob or an agent authorized to schedule meetings on his behalf to determine his availability.