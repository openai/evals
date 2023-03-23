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

Before sending a request to the model, we will iteratively call `plugin_actions` on the relevant plugins.  One plugin instance will be used per sample, so plugins will retain state over that request.

NOTE: By enabling plugins, we may mutate the overall message chain.  Currently, we append the plugin's description to the end of the system message (Or insert a new system message if none was provided). This is done to ensure that the model has context to know how to use the plugin.

```JSON
{
  "plugins": [
    "openai.bird.1",
    "openai.clock.1",
  ],
  "ideal": [
    "Red-tailed hawk"
  ],
  "plugin_actions": [
    {
        "namespace": "MyBirds",
        "endpoint": "listBirds",
        "content": {}
    }
  ],
  "input": [
    {
      "role": "system",
      "content": "Answer the following questions as concisely as possible."
    },
    {
      "role": "user",
      "content": "Can you list all of the birds in my birds list which are in the Accipitridae family?"
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