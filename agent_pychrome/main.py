 # agent_pychrome/main.py
import asyncio
import logging
import os
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Set higher logging level for pychrome to avoid overly verbose output
logging.getLogger('pychrome').setLevel(logging.WARNING)

# Assuming LangChain and OpenAI environment variables are set (e.g., OPENAI_API_KEY)
# You might need to install required packages: pip install langchain-openai python-dotenv
# from dotenv import load_dotenv
# load_dotenv()

try:
    from langchain_anthropic import ChatAnthropic
    # If using other models, import them here, e.g.:
    from langchain_google_genai import ChatGoogleGenerativeAI
    # from langchain_anthropic import ChatAnthropic
except ImportError:
    print("Please install langchain-openai: pip install langchain-openai")
    exit()

from agent_pychrome.agent_service import PychromeAgent
from agent_pychrome.agent_views import AgentSettings

async def main():
    # --- Configuration ---
    CDP_URL = "http://localhost:9222" # Default Chrome Debugging Protocol URL
    TASK = "Go to perplexity.ai, search for 'latest news on LLMs', and report the main findings."
    # TASK = "go to website = https://hackinglife.mitpress.mit.edu/   find chapter 7  open the seventh chapter and save it as pdf "
    # TASK = "Go to codeforces.com, click on enter then click on button 'Click the button to complete verification',then click on login  ."
    
    # TASK = "Go to perplexity.ai and find new findings on india pakistan war'."
    MAX_STEPS = 15
    # --- End Configuration ---
    time.sleep(4)

    # 1. Initialize the LLM
    # Replace with your preferred LLM provider and model
    try:
        # llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest") 
        # llm = ChatAnthropic(model="claude-3-sonnet-20240229")
        llm = ChatAnthropic(model="claude-3-7-sonnet-20250219", temperature=0.1) # Requires OPENAI_API_KEY
    except Exception as e:
        logging.error(f"Failed to initialize LLM: {e}")
        logging.error("Please ensure the required API keys and packages (e.g., langchain-openai) are installed and configured.")
        return

    # 2. Define Agent Settings (Optional)
    agent_settings = AgentSettings(
        max_input_tokens=120000, # Adjust based on model
        max_actions_per_step=5, # Limit actions LLM can take in one step
        use_vision=True, # Set to False if your LLM doesn't support vision or you don't want screenshots
        max_failures=3
        # Add other settings from agent_views.AgentSettings if needed
    )

    # 3. Initialize the Agent
    agent = PychromeAgent(
        task=TASK,
        llm=llm,
        cdp_url=CDP_URL,
        agent_settings=agent_settings
    )

    # 4. Run the Agent
    try:
        final_history = await agent.run(max_steps=MAX_STEPS)
        
        # Optional: Print final results or save history
        print("\n--- Agent Run Complete ---")
        if final_history:
            final_action_result = final_history.final_result()
            if final_action_result:
                 print(f"Final Result/Conclusion: {final_action_result}")
            else:
                 print("Agent did not reach a 'done' state with a conclusion.")
            
            # Example: Save history to file
            # history_file = "agent_run_history.json"
            # final_history.save_to_file(history_file)
            # print(f"Full agent history saved to {history_file}")
        else:
            print("Agent run did not produce history.")
            
    except Exception as e:
        logging.error(f"An error occurred during the agent run: {e}", exc_info=True)
    finally:
        # Ensure proper cleanup
        try:
            await agent.disconnect_browser()
        except Exception as e:
            logging.error(f"Error during cleanup: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nGracefully shutting down...")
    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
