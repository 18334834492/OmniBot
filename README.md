# OmniBot 2.0 — Cross-Platform AI Agent Framework

**LLM-Powered Desktop Automation Agent Framework. Features: intelligent planning, visual perception, precise control, safety governance, memory, and plugin extensibility. Built on a 4-layer architecture (Decision/Perception/Execution/Infrastructure). Includes API reference (LLMClient, controllers), 20+ configs, and security mechanisms (operation interception, auto-screenshots).**

---

## 📁 File Structure

### `__init__.py` — Package Entry Point

Defines core metadata:
```python
__version__ = "2.0.0"
__author__ = "OmniBot Team"
```

---

### `__main__.py` — Python Module Entry

Provides module-level entry point for:
```bash
python -m omnibot
```

Functions:
- Imports and executes `cli.main`
- Enables Python package execution

---

### `cli.py` — Command-Line Interface

Supports multiple execution modes:

#### 1. Single Task Execution
```bash
python -m omnibot --task "打开Chrome，搜索天气，截屏保存到桌面"
```

#### 2. Checkpoint Resume
```bash
python -m omnibot --resume checkpoints/checkpoint_step3_xxx.json
```

#### 3. Interactive Mode
```bash
python -m omnibot --interactive
```

#### 4. Tool List
```bash
python -m omnibot --list-tools
```

**Core Features**:
- 📊 **Logging System**: Console + file logs (daily rotation)
- 🎨 **Rich UI**: Beautiful terminal interface (optional dependency)
- 🛡️ **Safe Mode**: Optional operation safeguards
- 💾 **Checkpoint Save**: Resume interrupted tasks
- 🧠 **Memory System**: Task history and experience recording

---

## 🚀 Quick Start

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Examples
```bash
# Execute simple task
python -m omnibot --task "打开计算器，计算 123 + 456"

# Enter interactive mode
python -m omnibot --interactive

# View available tools
python -m omnibot --list-tools
```

---

## 🧩 Core Components

### Event Bus
Publish-subscribe pattern for decoupled component communication.

### Plugin Manager
Auto-loads and manages extensible plugins.

### Memory System
Records task execution history and experiences.

### Checkpoint Manager
Supports task resume after interruption.

---

## 📝 Use Cases

1. **Office Automation**: Batch processing repetitive tasks
2. **Smart Assistant**: Natural language PC control
3. **Workflow Orchestration**: Complex task chain automation
4. **Test Validation**: GUI automated testing

---

## 🛠️ Tech Stack

- **Language**: Python 3.12+
- **Architecture**: Event-driven + Plugin-based
- **LLM**: Supports multiple language models
- **UI**: Rich (optional) terminal beautification

---

## 📄 License

TBD

---

**OmniBot Team** | Version 2.0.0

---

# OmniBot 2.0 — 全能灵控

**基于大语言模型的桌面自动化智能体框架。 具备智能规划、视觉感知、精准执行、安全可控、记忆系统与插件扩展能力。采用四层架构（决策/感知/执行/基础设施），提供LLM客户端、键鼠控制器等API参考，涵盖20余项配置详解及敏感操作拦截、自动截图等安全机制。**

---

## 📁 文件结构

### `__init__.py` — 包入口文件

定义项目核心元信息：
```python
__version__ = "2.0.0"
__author__ = "OmniBot Team"
```

---

### `__main__.py` — Python 模块入口

提供模块级入口点，支持：
```bash
python -m omnibot
```

功能：
- 导入并执行 `cli.main`
- 启用 Python 包运行模式

---

### `cli.py` — 命令行接口

支持多种执行模式：

#### 1. 单任务执行
```bash
python -m omnibot --task "打开Chrome，搜索天气，截屏保存到桌面"
```

#### 2. 断点恢复
```bash
python -m omnibot --resume checkpoints/checkpoint_step3_xxx.json
```

#### 3. 交互模式
```bash
python -m omnibot --interactive
```

#### 4. 工具列表
```bash
python -m omnibot --list-tools
```

**核心功能**：
- 📊 **日志系统**：控制台输出 + 文件日志（按日期轮转）
- 🎨 **Rich UI**：美观的终端界面（可选依赖）
- 🛡️ **安全模式**：可选的操作防护机制
- 💾 **断点保存**：支持中断后恢复执行
- 🧠 **记忆系统**：任务历史和经验记录

---

## 🚀 快速开始

### 安装依赖
```bash
pip install -r requirements.txt
```

### 运行示例
```bash
# 执行简单任务
python -m omnibot --task "打开计算器，计算 123 + 456"

# 进入交互模式
python -m omnibot --interactive

# 查看可用工具
python -m omnibot --list-tools
```

---

## 🧩 核心组件

### 事件总线
基于发布-订阅模式，实现组件解耦通信。

### 插件管理器
自动加载和管理可扩展插件。

### 记忆系统
记录任务执行历史和经验。

### 检查点管理器
支持任务中断后恢复。

---

## 📝 使用场景

1. **自动化办公**：批量处理重复性任务
2. **智能辅助**：自然语言控制电脑
3. **流程编排**：复杂任务链自动化
4. **测试验证**：GUI 自动化测试

---

## 🛠️ 技术栈

- **语言**：Python 3.12+
- **架构**：事件驱动 + 插件化
- **LLM**：支持多种大语言模型
- **UI**：Rich（可选）终端美化

---

## 📄 License

待定

---

**OmniBot Team** | Version 2.0.0
