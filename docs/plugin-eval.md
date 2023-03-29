# Plugins

## Overview

Creating and evaluating AI plugins can be a challenging task, as they may interact with the environment, collaborate with other plugins, and provide domain-specific knowledge to the AI model. Moreover, the AI model should be able to use the plugin effectively under various constraints. This plugin evaluation framework aims to simplify the process of creating, evaluating, and integrating AI plugins, ensuring a balance between realism and ease of evaluation.

This framework is designed for local evaluation of AI plugins, and it's not intended for deploying Language Model plugins in production. The primary goal is to assist developers, researchers, and contributors in creating and evaluating custom plugins for their use cases, as well as understanding how the AI model responds to the installation of new plugins.

## Evaluation Workflows

There are several factors to consider when assessing a model's performance using plugins.  We welcome additional contributions to this section, as it is not exhaustive.

### Plugin State Population

The method by which a plugin is populated with data is an important consideration when evaluating the model's behavior. There are two main ways to populate a plugin:

#### Externally Populated

In this scenario, the plugin is pre-populated with data outside of the current conversation. This allows the evaluation of the model's ability to rely on the plugin to obtain information about the state of the world. For example, a calendar plugin could be pre-populated with a meeting, and the model would then be asked to provide details about that meeting.

Assuming a plugin named `MyCalendar` with `addMeeting` and `listMeetings` endpoints, the following input would be used to populate the plugin externally.  Setting the `include_in_conversation` flag to `false` will prevent the `addMeeting` action from being recorded in the conversation, allowing the plugin to act as if it was externally populated.  By default, this flag is `true`.  In effect, this allows the plugin to be changed from a source distinct from the model during evaluation.

```JSON
{
  "plugins": [
    "openai.calendar.1",
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
      "content": "List all of my meetings"
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

#### Information Integration Workflow

We sometimes use AI models to augment existing data sources.  In this scenario, the model is expected to augment the plugin's response with its own knowledge. For example, a plugin may perform a single calculation, which the model then integrates into a larger answer.  In this case, the model is expected to use the plugin's response to provide a more complete answer.

#### Agentic Workflow

As AI models are increasingly being used for more autonomous tasks, we aim to evaluate their ability to accomplish a single, high-level task. Here, the user assigns a task to the model, and the model is expected to perform the necessary steps to complete it using relevant plugins and information. For example, a user might ask the model to "set up a vacation," and the model would then organize the vacation details accordingly.

#### Multi-Party Workflow

This workflow evaluates the model's ability to interact and communicate effectively with multiple users or other models concurrently. The assessment focuses on whether the parties are able to collaborate and whether a model is able to use the information provided by the other parties to accomplish its task. For example, a user might ask the model to "schedule a meeting with Bob about graham cracker production", and the model would then use the relevant plugins and information to coordinate the meeting, possibly by querying Bob or an agent authorized to schedule meetings on his behalf to determine his availability.

## Getting Started

To create a new plugin, follow these steps:

1. Familiarize yourself with the example plugins found in the `openai` folder, such as [openai/bird.py](evals/plugin/openai/bird.py), [openai/clock.py](evals/plugin/openai/clock.py), and [openai/text_analyzer.py](evals/plugin/openai/text_analyzer.py). These examples will help you understand the structure and requirements for creating a new plugin.

2. Follow the plugin creation process detailed below.

3. Submit a pull request with your new plugin and evaluation example(s).

## Preparing Samples

Samples which use plugins take an additional argument - `plugins` - which is a list of plugins to enable for that sample. The evaluation framework will automatically instantiate a new plugin instance from the registry for each sampling.

Before sending a request to the model, we will iterate through the `input` messages, invoking the relevant plugins.  One plugin instance will be used per sample, so plugins will retain state throughout that request.

Please note, we currently require chat-style inputs for plugins

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

NOTE: By enabling plugins, the input list may be mutated.  Currently, we append the plugin's description to the end of the system message (Or insert a new system message if none was provided). This is done to ensure that the model has context to know how to use the plugin.

For example, the system message "Answer the following questions as concisely as possible." would be transformed to the following:
```
Answer the following questions as concisely as possible.

// Use the birds plugin to get a list of all the birds you have added to your birds list
namespace MyBirds {
  
// Determine if a bird is in your list
type hasBird = (_: {
// The name of the bird you want to fetch.
name: string,
}) => any;

// API for fetching your birds.
type listBirds = () => any;
    
} // namespace MyBirds
```

## Plugin Creation

### Namespace and API Description

- Use the `@namespace` decorator to define your plugin's namespace and provide a brief description. The AI model will use this namespace to encapsulate your plugin. The description will help the model understand the goals, purposes, and limitations of your plugin.

- For each API method in your plugin, use the `@api_description` decorator to define the API's name, description, and arguments (if any). The AI model will use this information to understand the purpose of each API method and the inputs it expects. The description will also provide context for the function.

### Input and Output

- Ensure your plugin methods accept inputs as defined in the `api_args` parameter of the `@api_description` decorator.

- The output of your plugin methods should be in a format that can be easily consumed by the AI model.  We currently expect a JSON object, which the model will see as a string.

### Adding to Registry

- To add a plugin to the registry, create a new YAML file in the `registry/plugins` directory under the same directory structure and name as your plugin Python file.

- The naming convention for a new registry entity is `<namespace>.<plugin_name>.<version>`. Feel free to be creative with the namespace and plugin name. The version is a monotonically increasing integer, starting at 1. For example, `openai.bird.1`.

- The YAML file should contain a description of how to instantiate your plugin. A few examples can be found [in our example plugin registry file](evals/registry/plugins/openai/example.yaml).

### Evaluation Examples

- Create evaluation examples that involve your custom plugin. Plugins should be enabled on a per-sample basis, allowing us to group examples of how the plugin changes the model's behavior with examples that might not use the plugin at all but are related requests.

- [Model-graded evaluations](docs/build-eval.md#for-model-graded-evals-a-step-by-step-workflow) tend to work better than string-based evaluations.

- It is best-practice to test the behavior of the model when your plugin is not installed. This helps effectively communicate the value of your plugin when combined with the model.  We recommend creating a new evaluation that demonstrates the model's behavior without your plugin, and then creating a new evaluation that demonstrates the model's behavior with your plugin.  These can be combined into an `evalset` to make the connection between the two evaluations more clear.

#### Evaluation Dimensions
We've found a few initial dimensions interesting for evaluating plugin use

- **Plugin Appropriateness**: Does the plugin make sense for the task at hand?  For example, if the user is asking about the weather, it would be likely be inappropriate to use a plugin that analyzes the Github documentation.  This can be implemented by having the last `input` as an incorrect plugin call.
- **Plugin Recovery**: How well does the model recover from unexpected responses from a plugin?  For example, the [wrong Mercury example using the WolframAlpha plugin](https://writings.stephenwolfram.com/2023/03/chatgpt-gets-its-wolfram-superpowers/), where the model eventually is able to recover from the unexpected response via the plugin's error messages.  This can, partially, be implemented by having the last `input` as a plugin call that returns invalid results.
- **Knowledge Composition**: How well does the model integrate the plugin's knowledge into its own understanding of the world?  For example, filtering a list based on a property not provided by the plugin. This can be implemented by having the last `input` be a plugin call that returns some information, with a user message that asks a question about that information before.

## Extending Existing Plugins

- Do not modify existing plugins. Instead, create a new plugin that extends the existing plugin using the plugin creation process above. When creating this new plugin, increase the version number of the plugin you are extending by 1.

- Plugins can be forked by changing the namespace, but this should be done with care. We cannot allow two plugins with the same namespace to exist in practice.

- Fixing bugs in existing plugins may be done if necessary, but please be thoughtful if the bug fix meaningfully changes the plugin's behavior. Not all plugins will be perfect, and evaluating our model's ability to recover from buggy behavior is an important part of the evaluation process.