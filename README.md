# Agent 47

An AI-powered agent that leverages `pychrome` to interact with web pages based on instructions from a Large Language Model (LLM). It can navigate websites, fill out forms, click elements, and save page content, all driven by natural language tasks.

## Features

*   **LLM-Driven Web Automation:** Control a Chrome browser using natural language tasks.
*   **Core Web Interactions:**
    *   `navigate`: Go to specified URLs.
    *   `type_text`: Input text into fields (e.g., search bars, forms), with an option to simulate an "Enter" key press.
    *   `click_element`: Click on various web page elements.
    *   `save_page_as_pdf`: Save the current web page as a PDF document.
    *   `done`: Conclude the current task, indicating success or failure and providing a summary.
*   **Flexible Element Selection:** Identify elements using various strategies:
    *   `aria_label`: Accessibility labels.
    *   `placeholder`: Placeholder text in input fields.
    *   `text_content`: Visible text content of an element.
    *   `name`: The `name` attribute of an element.
    *   `node_type`: The HTML tag/node type (e.g., "TEXTAREA", "BUTTON").
*   **Pluggable LLM Integration:** Easily switch between different LLMs (e.g., Anthropic Claude, Google Gemini, OpenAI GPT models). Example provided with Anthropic Claude.
*   **Configurable Agent:** Adjust settings like token limits, max actions per step, and error handling.
*   **Basic Error Handling:** Includes retries for certain operations and fallback strategies for element interaction.

## Directory Structure

A brief overview of key files:

*   `agent_pychrome/main.py`: Main entry point to run the agent. Contains configuration for the task, LLM, and agent settings.
*   `agent_pychrome/agent_service.py`: Core agent logic, including LLM communication, action execution, and state management.
*   `agent_pychrome/chrome_controller.py`: Handles direct interaction with the Chrome browser using `pychrome`. Implements low-level browser control functions.
*   `agent_pychrome/agent_views.py`: Defines Pydantic models for agent actions, arguments, settings, and state.
*   `agent_pychrome/message_manager/`: Manages the conversation history and prepares messages for the LLM.
*   `agent_pychrome/memory/`: (Placeholder) Intended for future procedural memory implementation.

## Setup and Installation

1.  **Prerequisites:**
    *   Python 3.9 or higher.
    *   A running instance of Google Chrome.

2.  **Clone the Repository (or use the current directory):**
    ```bash
    # If you have it as a git repository:
    # git clone <repository_url>
    # cd agent_pychrome_directory
    ```

3.  **Install Dependencies:**
    It's recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
    Create a `requirements.txt` file with the following content:
    ```txt
    pychrome
    langchain-anthropic
    langchain-google-genai
    langchain-openai  # If using OpenAI models
    langchain-core
    pydantic
    openai # client library, often a dependency for langchain OpenAI integrations
    # Add other specific langchain provider libraries if needed
    ```
    Then install:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Environment Variables:**
    Set up the necessary API keys for your chosen LLM. For example, for Anthropic Claude:
    ```bash
    export ANTHROPIC_API_KEY="your_anthropic_api_key"
    ```
    Or for OpenAI:
    ```bash
    export OPENAI_API_KEY="your_openai_api_key"
    ```
    You can set these in your shell environment or use a `.env` file with a library like `python-dotenv` (uncomment its import in `main.py` if you use it).

5.  **Run Chrome in Debug Mode:**
    You need to start Chrome with the remote debugging port enabled. The default port used by this agent is `9222`.
    *   **Windows:**
        ```cmd
        "C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222
        ```
        (Adjust the path to your Chrome installation if necessary.)
    *   **macOS:**
        ```bash
        /Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --remote-debugging-port=9222
        ```
    *   **Linux:**
        ```bash
        google-chrome --remote-debugging-port=9222
        ```
    Close all other running instances of Chrome before starting it in debug mode to ensure it uses the specified port.

## How to Run

1.  Ensure Chrome is running in debug mode (see step 5 above).
2.  Modify the `TASK` variable in `agent_pychrome/main.py` to define what you want the agent to do.
    ```python
    # agent_pychrome/main.py
    # ...
    TASK = "Go to wikipedia.org, search for 'Large Language Models', and save the main page as LLM_wiki.pdf"
    # ...
    ```
3.  Run the agent from the root directory of the project:
    ```bash
    python agent_pychrome/main.py
    ```
    The agent will connect to the browser, and you'll see logging output in the console as it performs actions.

## Key Agent Actions

The LLM will decide which of these actions to take based on the task and current browser state. The actions are defined with Pydantic models in `agent_views.py`.

*   **`navigate`**: Navigates to a given URL.
    *   `args: { "url": "https://example.com" }`
*   **`type_text`**: Types text into a selected element.
    *   `args: { "selector": { "strategy": "placeholder", "value": "Search...", "node_name": "INPUT" }, "text_to_type": "Hello World", "press_enter": true }`
    *   `press_enter` (optional, default `false`): Simulates pressing Enter after typing.
*   **`click_element`**: Clicks on a selected element.
    *   `args: { "selector": { "strategy": "text_content", "value": "Submit Button" } }`
*   **`save_page_as_pdf`**: Saves the current page as a PDF.
    *   `args: { "output_path": "my_page.pdf" }`
    *   If `output_path` is not provided, it defaults to `web_page.pdf` in the script's execution directory.
*   **`done`**: Indicates the task is finished.
    *   `args: { "success": true, "conclusion": "Successfully retrieved and saved the information." }`

## Configuration

Several aspects of the agent can be configured in `agent_pychrome/main.py`:

*   **`CDP_URL`**: The Chrome Debugging Protocol URL (default: `http://localhost:9222`).
*   **`TASK`**: The natural language task for the agent.
*   **`MAX_STEPS`**: Maximum number of steps the agent will take before stopping.
*   **LLM Selection**:
    ```python
    # Example:
    # from langchain_anthropic import ChatAnthropic
    # llm = ChatAnthropic(model="claude-3-opus-20240229", temperature=0.1)
    # from langchain_google_genai import ChatGoogleGenerativeAI
    # llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")
    ```
*   **`AgentSettings`**:
    *   `max_input_tokens`: Max tokens for LLM input.
    *   `max_actions_per_step`: Max actions the LLM can propose in a single step.
    *   `use_vision`: (Currently placeholder functionality for screenshots) Set to `True` if you intend to use vision capabilities with a compatible LLM (requires further implementation for screenshot processing in prompts).
    *   `max_failures`: Max consecutive failures before the agent stops.

## Extending the Agent

To add new capabilities (actions) to the agent:

1.  **Define Action Models:** In `agent_pychrome/agent_views.py`:
    *   Create a Pydantic model for the action's arguments (e.g., `MyNewActionArgs`).
    *   Create a Pydantic model for the action itself (e.g., `PychromeMyNewAction`).
    *   Add your new action model to the `PychromeActionModel` Union.
2.  **Implement Action Logic:** In `agent_pychrome/chrome_controller.py`:
    *   Add a new method to `ChromeController` that performs the low-level browser interaction for your new action.
3.  **Handle Action in Service:** In `agent_pychrome/agent_service.py`:
    *   Update the `execute_action` method in `PychromeAgent` to call your new `ChromeController` method when the LLM requests your new action.
4.  **Update System Prompt:** In `agent_pychrome/agent_service.py`:
    *   Modify the `_system_message_content` to inform the LLM about the new action, its purpose, and its arguments.

## Troubleshooting

*   **"No tabs found" / Connection Issues:**
    *   Ensure Chrome is running with `--remote-debugging-port=9222`.
    *   Make sure no other Chrome instance is already using that port. Close all Chrome windows before starting it in debug mode.
    *   Verify the `CDP_URL` in `main.py` matches the port Chrome is using.
*   **LLM API Key Errors:**
    *   Double-check that your API key environment variable (e.g., `ANTHROPIC_API_KEY`) is correctly set and exported in the terminal session where you run the agent.
    *   Ensure the API key is valid and has the necessary permissions/credits.
*   **Element Not Found:**
    *   The agent relies on the LLM's ability to choose good selectors. If elements are consistently not found, the task might be too complex, or the selectors chosen by the LLM might be suboptimal for the current page structure.
    *   Review the debug logs to see which selectors are being attempted. 
