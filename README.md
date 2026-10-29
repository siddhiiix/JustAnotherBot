# JustAnotherBot 🤖

**A personal AI-powered Discord assistant built with Python, Ollama, SQLite, and ChromaDB.**

JustAnotherBot is an ongoing project to build a customizable Discord AI assistant with a distinctive Victorian English personality, persistent memory, and Retrieval-Augmented Generation (RAG).

The goal is to create a conversational assistant that can do more than respond to messages: it should remember relevant information, interact through Discord slash commands, and retrieve information from documents.

## ✨ Features

### 🧠 AI Conversations
- Integrates with locally running language models through Ollama.
- Supports a distinctive Victorian English conversational personality.
- Uses a modular Python architecture to manage AI interactions and prompts.

### 💾 Persistent Memory
- Uses SQLite for persistent memory storage.
- Includes modules for long-term memory and automatic memory functionality.
- Provides commands for interacting with and managing memories.

### 📚 Retrieval-Augmented Generation (RAG)
- Uses ChromaDB as a vector database.
- Includes document loading, text splitting, embedding generation, and vector search components.
- Provides a RAG pipeline for retrieving relevant information from ingested documents.
- Includes commands for document ingestion and retrieval-based questions.

### ⚙️ Customization and Discord Commands
- Includes modules for personality settings and configurable behavior.
- Provides dedicated slash-command modules for help, memory management, document ingestion, and RAG queries.
- Uses a modular structure to make adding new capabilities easier.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core application logic |
| discord.py | Discord bot integration |
| Ollama | Local language model integration |
| SQLite | Persistent memory storage |
| ChromaDB | Vector storage and retrieval |
| Embeddings | Semantic document retrieval |
| Git and GitHub | Version control and project hosting |

## 📁 Project Structure

```text
JustAnotherBot/
├── ai/
│   ├── rag/
│   │   ├── document_loader.py
│   │   ├── embeddings.py
│   │   ├── ingest.py
│   │   ├── rag_pipeline.py
│   │   ├── text_splitter.py
│   │   └── vector_store.py
│   ├── auto_memory.py
│   ├── conversation.py
│   ├── long_term_memory.py
│   ├── ollama_client.py
│   ├── personality_settings.py
│   └── prompts.py
├── bot/
├── commands/
├── config.py
├── justanotherbot.py
├── clean_memory.py
└── README.md
```

*The project structure may evolve as development continues.*

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or a compatible version supported by the project's dependencies.
- A Discord account and a Discord application configured as a bot.
- Ollama installed locally.
- Git.

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/JustAnotherBot.git
cd JustAnotherBot
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Create a virtual environment

**Windows:**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

Install the Python packages required by the project.

A `requirements.txt` file should be added with the project's verified dependencies before using this as a one-command setup.

Once that file is available, run:

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a local `.env` file for the required configuration.

Add your Discord bot token and any other required settings using the variable names expected by the application.

**Never commit your `.env` file or publish your Discord bot token.**

### 5. Configure Ollama

Install Ollama and download the language model configured for the project.

For example, if the application is configured to use Qwen 2.5 3B:

```bash
ollama pull qwen2.5:3b
```

Ensure Ollama is running before starting the bot.

### 6. Start the bot

```bash
python justanotherbot.py
```

The exact configuration and dependency requirements may change as the project develops.

## 🗺️ Roadmap

Planned improvements include:

- More reliable long-term and automatic memory.
- Improved RAG retrieval and document processing.
- Additional Discord slash commands.
- Better error handling and logging.
- Automated testing and validation.
- Improved configuration and installation.
- Further customization of the assistant's personality and behavior.

## 🧪 Testing

The repository includes Python test scripts for memory functionality and different parts of the RAG pipeline.

Testing coverage and automated test execution will be expanded as development continues.

## 📌 Project Status

**In active development.**

Features and interfaces may change as new capabilities are implemented and existing components are refined.

## 👩‍💻 Author

Developed as a personal project to explore practical applications of local language models, conversational AI, persistent memory, and Retrieval-Augmented Generation.
