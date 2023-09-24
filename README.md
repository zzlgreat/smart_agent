# Smart Agent

Smart Agent is an open-source project offering a comprehensive library for executing user-provided instructions.

## Features

- **Toolkit**: The project contains a module named `real_world` defining a `toolkit` that incorporates a variety of handy tools.
- **Special Models**: A module named `special mind` for loading fine-tuned models.
- **Memory**: Modules `memory` and `flash mind` are provided for long-term and short-term memory, respectively.

## Architecture

The core logic of the project is based on the following four components:

1. **Planner**: Breaks down tasks based on user input.
2. **Distributor**: Responsible for selecting appropriate tools from the `toolkit` to execute the plans.
3. **Worker**: Takes charge of invoking tasks from the toolkit and returning the results.
4. **Solver**: Consolidates all plans and results, rendering a conclusion.

## Usage

Run `smart_agent.py` and input your instruction.

Smart Agent是一个开源项目，提供了一套完整的库，用于执行用户提供的指令。

## 特点

- **工具箱**: 该项目包含一个名为`real_world`的模块，其中定义了`toolkit`，内置各种实用工具。
- **特殊模型**: 一个名为`special mind`的模块，用于加载经过微调的模型。
- **记忆**: 提供`memory`和`flash mind`两个模块，分别用于长期和短期记忆。

## 架构

项目的核心逻辑基于以下四个部分：

1. **Planner**: 根据用户输入的指令拆分任务。
2. **Distributor**: 负责从`toolkit`中选择适当的工具来执行计划。
3. **Worker**: 负责调用工具箱中的任务，并返回任务调用的结果。
4. **Solver**: 整合所有的计划和结果，并返回一个结论。

## 使用方法
1. **Toolkit**: 按照real_world/toolkit.py中的格式编写自己的调用函数。
2. **Train**: 根据rewoo格式的数据集和trelis数据集分别用lora和qlora分别训练拆解任务的模型和调度函数的模型.调度函数的模型见func_caller_trainer.py
3. ****
运行`smart_agent.py`并输入您的指令。

