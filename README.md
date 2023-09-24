# Smart Agent

Smart Agent is an open-source project offering a comprehensive library for deconstructing complex tasks and dispatching functions within a toolkit. The project adopts a universal Agent design philosophy, summarizing the process of handling complex tasks into four key stages: Task Planning (Planner) → Function Dispatching (Distributor) → Function Execution (Worker) → Result Synthesis (Solver).

## Key Features

- **Toolkit**: The project includes a module called `real_world`, which houses a `toolkit` containing a variety of fully customizable tools.

- **Ease of Model Use**: The project adheres to a design philosophy that separates the Agent from the operator. It deploys models as RESTful APIs, calling the LLM API interface only when necessary.

- **Memory Capabilities**: The project offers two modules, `memory` and `flash mind`, for long-term and short-term memory, respectively. (In Development)

## Architecture

The core logic of the project is built upon the following four components:

1. **Task Planning (Planner)**: Decomposes tasks based on user input.
  
2. **Function Dispatching (Distributor)**: Selects appropriate tools from the `toolkit` to execute the planned tasks.
  
3. **Function Execution (Worker)**: Executes the tasks defined in the `toolkit` and returns the results.
  
4. **Result Synthesis (Solver)**: Integrates all task plans and results to output a final conclusion.

## Usage Guide

1. **Customize Toolkit**: In the `real_world/toolkit.py` file, you can add your own callable functions following the example format provided.

2. **Model Training**: Train models for task decomposition and function dispatching using the REWOO and Trelis datasets via LORA and QLORA, respectively.
  - **Planner/Solver**: [Marcoroni-70B-v1](https://huggingface.co/AIDC-ai-business/Marcoroni-70B-v1) can be used as a base model for task planning. This model is well-suited for task decomposition as it has been trained on a large amount of COT data.
  - **Distributor**: Refer to `func_caller_trainer.py` for training the function dispatching model.
  
3. **Model Deployment**: Deploy the required models and interfaces. The current design logic is to have a shared model for Planner and Solver, and a separate model for the Distributor.
  - **Planner/Solver**: The simplest way to deploy is to install [text-generation-webui](https://github.com/oobabooga/text-generation-webui) and load [Marcoroni-70B-v1](https://huggingface.co/Panchovix/Marcoroni-70B-v1-4.65bpw-h6-exl2) using EXLLAMAv2.
  - **Distributor**: Run `python special_mind/fllama_api.py`, making sure to configure `api_config.json` beforehand.

4. **Run the Agent**: Configure the two model interfaces in `model_api_config.py`.
  - Execute `smart_agent.py` and input your command.


Smart Agent 是一个开放源代码的项目，为您提供一整套完善的库，用以解构复杂任务并调度 toolkit 中的函数。该项目采用一种通用的 Agent 设计思路，将复杂任务处理的流程概括为：任务规划（Planner）→ 函数调度（Distributor）→ 函数执行（Worker）→ 结果整合（Solver）。
## 特点

- **工具箱**: 该项目包含一个名为`real_world`的模块，其中定义了`toolkit`，内置各种完全自定义的工具。
- **模型易用性**: agent和操作者的分离的思想，将模型部署为restful api，需要的时候才会去调用接入llm的api接口。
- **记忆**: 提供`memory`和`flash mind`两个模块，分别用于长期和短期记忆。（TO DO）

## 架构

项目的核心逻辑基于以下四个部分：

1. **Planner**: 根据用户输入的指令拆分任务。
2. **Distributor**: 负责从`toolkit`中选择适当的工具来执行计划。
3. **Worker**: 负责调用工具箱中的任务，并返回任务调用的结果。
4. **Solver**: 整合所有的计划和结果，并返回一个结论。

## 使用方法
1. **Toolkit**: 在real_world/toolkit.py中，按照示例格式编写自己的调用函数。
- 示例中的search_bing模块依赖[sitedorks](https://github.com/Zarcolio/sitedorks)，search_bilibili依赖bilibili-api-python模块
2. **Train**: 根据rewoo格式的数据集和trelis数据集分别用lora和qlora分别训练拆解任务的模型和调度函数的模型.
- **planner/solver**: 可以使[用Marcoroni-70B-v1](https://huggingface.co/AIDC-ai-business/Marcoroni-70B-v1)作为基座。该模型训练时采用了大量COT数据，很适合做任务分解。如果没有现成的rewoo格式的数据集，该模型本身的zero-shot能力就很好，
- **distributor**: 训练调度函数的模型见func_caller_trainer.py
3. **deploy**: 部署需要使用的模型和接口。目前设计的逻辑是Planner和Solver共用一个模型,distributor自用一个模型。
- **planner/solver**: 最简单的方式是安装[text-generation-webui](https://github.com/oobabooga/text-generation-webui)，用exllamav2加载[Marcoroni-70B-v1](https://huggingface.co/Panchovix/Marcoroni-70B-v1-4.65bpw-h6-exl2)。
安装后运行`python server.py --loader exllamav2 --model Marcoroni-70B-v1-4.65bpw-h6-exl2 --gpu-split 21,22,23 --listen --extensions api`
- **distributor**: `python specical_mind/fllama_api.py` 运行前配置好api_config.json
4. **run**: 在model_api_config.py中配置加载的两个模型的接口。
- 运行`smart_agent.py`并输入您的指令。

