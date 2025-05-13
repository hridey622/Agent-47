from __future__ import annotations

import asyncio
import logging
import traceback # Added for error handling in execute_action
import json
import time # Added for StepMetadata
from typing import Any, Awaitable, Callable, Dict, Generic, List, Optional, TypeVar, Union

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, HumanMessage # Added HumanMessage
from pydantic import ValidationError # Added for parsing AgentOutput

# Internal imports
from . import chrome_controller as pychrome_utils # Renamed for clarity
from .agent_views import (
    AgentSettings,
    AgentState,
    AgentOutput, # Now uses PychromeActionModel by default
    ActionResult,
    PychromeActionModel, # Import the new action model
    ElementSelector, # Import for the helper method
    AgentHistoryList,
    AgentHistory, # Added for storing history
    BrowserStateHistory, # Added for storing history state
    StepMetadata,
)
from .message_manager.message_manager_service import MessageManager, MessageManagerSettings
from .utils import extract_json_from_model_output # Import the new utility
from .chrome_controller import (
    ChromeController, find_element_in_dom, 
    create_placeholder_criteria, create_text_criteria, 
    create_aria_label_criteria, create_name_criteria,
    create_node_type_criteria # Added import
)
# from .message_manager.message_manager_views import MessageManagerState # MessageManagerState is part of AgentState
# from .prompts import AgentMessagePrompt, PlannerPrompt, SystemPrompt # To be added later
# from .memory.service import Memory, MemorySettings # To be added later

logger = logging.getLogger(__name__)

class PychromeAgent:
    def __init__(
        self,
        task: str,
        llm: BaseChatModel,
        cdp_url: str = "http://localhost:9222",
        agent_settings: Optional[AgentSettings] = None,
        initial_agent_state: Optional[AgentState] = None,
    ):
        self.task = task
        self.llm = llm
        self.settings = agent_settings or AgentSettings()
        self.state = initial_agent_state or AgentState()

        self.chrome_controller = pychrome_utils.ChromeController(cdp_url=cdp_url)

        _system_message_content = (
            "You are a helpful AI assistant that interacts with a Chrome browser via a structured API. "
            "Based on the user's task and the current state of the browser, you will decide on the next action to take. "
            "Your available actions are: navigate, type_text, click_element, save_page_as_pdf, and done. "
            "For type_text, you can optionally specify 'press_enter: true' to simulate pressing Enter after typing (e.g., for submitting a search)."
            "For save_page_as_pdf, you can optionally specify 'output_path', otherwise it defaults to 'web_page.pdf'."
            "Ensure your response strictly follows the Pydantic schema provided for AgentOutput."
        )
        _system_prompt_obj = SystemMessage(content=_system_message_content)

        mm_settings = MessageManagerSettings(
            max_input_tokens=self.settings.max_input_tokens,
            include_attributes=self.settings.include_attributes,
            message_context=self.settings.message_context,
            # sensitive_data=self.settings.sensitive_data, # If you add to AgentSettings
            # available_file_paths=self.settings.available_file_paths, # If you add to AgentSettings
        )

        self.message_manager = MessageManager(
            task=self.task,
            system_message=_system_prompt_obj,
            settings=mm_settings,
            state=self.state.message_manager_state,
        )

        self.ActionModel = PychromeActionModel
        self.AgentOutput = AgentOutput # AgentOutput in views.py now uses PychromeActionModel

        logger.info(f"PychromeAgent initialized for task: {self.task}")

    def _create_criteria_func_from_selector(self, selector: ElementSelector) -> Callable[[Dict[str, Any]], bool]:
        """Creates a criteria function for find_element_in_dom based on ElementSelector."""
        strategy = selector.strategy
        value = selector.value
        node_name_arg = selector.node_name # Use separate var to avoid conflict with node_name in criteria func
        exact_match = selector.exact_match
        
        
        if strategy == "aria_label":
            return pychrome_utils.create_aria_label_criteria(value, node_name=node_name_arg)
        elif strategy == "placeholder":
            return pychrome_utils.create_placeholder_criteria(value, node_name=node_name_arg)
        elif strategy == "text_content":
            return pychrome_utils.create_text_criteria(value, node_name=node_name_arg, exact_match=exact_match)
        elif strategy == "name":
            # Ensure create_name_criteria is imported and used correctly
            criteria_func = create_name_criteria(
                selector.value,
                node_name=selector.node_name, # Pass node_name if available
                exact_match=selector.exact_match
            )
            return criteria_func
        elif strategy == "node_type": # New strategy
            criteria_func = create_node_type_criteria(selector.value)
            return criteria_func
        elif strategy == "tag_name":
            # New strategy: Find by tag name (e.g., 'textarea', 'input')
            tag_name_upper = value.upper()
            def tag_name_criteria(node):
                return node.get('nodeName') == tag_name_upper
            # Generate a descriptive name for the criteria function for logging
            tag_name_criteria.__name__ = f"is_tag_{tag_name_upper}"
            return tag_name_criteria
        elif strategy == "css_selector":
            logger.warning("CSS selector strategy is not directly supported by find_element_in_dom. This may not work as expected.")
            def css_placeholder_criteria(node): 
                return False
            css_placeholder_criteria.__name__ = "css_selector_unsupported"
            return css_placeholder_criteria
        else:
            raise ValueError(f"Unsupported selector strategy: {strategy}")

    async def execute_action(self, action: PychromeActionModel) -> ActionResult:
        logger.info(f"Executing action: {action.action_type} with args: {action.args}")
        original_action = action # Keep a copy
        action_result = None

        try:
            if action.action_type == "type_text":
                criteria_func = self._create_criteria_func_from_selector(action.args.selector)
                description = f"input field for '{action.args.text_to_type}' (selector: {action.args.selector.strategy}='{action.args.selector.value}')"
                press_enter = action.args.press_enter if action.args.press_enter is not None else False # Get press_enter flag
                success = await asyncio.to_thread(
                    self.chrome_controller._find_and_type_pychrome,
                    criteria_func,
                    action.args.text_to_type,
                    description=description,
                    press_enter_after=press_enter # Pass flag to controller method
                )
                if success:
                    action_result_text = f"Typed '{action.args.text_to_type}' into element selected by {action.args.selector.strategy}='{action.args.selector.value}'."
                    if press_enter:
                        action_result_text += " Then pressed Enter."
                    action_result = ActionResult(extracted_content=action_result_text)
                else:
                    # --- START OF HARDCODED FALLBACK LOGIC ---
                    # Check if the original attempt was for a TEXTAREA using a specific selector
                    if action.args.selector.node_name == "TEXTAREA" and \
                       action.args.selector.strategy != "node_type": # And not already trying node_type

                        logger.warning(f"Specific selector for TEXTAREA failed. Attempting fallback to generic TEXTAREA.")
                        
                        fallback_selector = ElementSelector(strategy="node_type", value="TEXTAREA")
                        fallback_criteria_func = self._create_criteria_func_from_selector(fallback_selector)
                        fallback_description = f"input field (fallback to any TEXTAREA) for '{action.args.text_to_type}'"
                        
                        fallback_success = await asyncio.to_thread(
                            self.chrome_controller._find_and_type_pychrome,
                            fallback_criteria_func,
                            action.args.text_to_type,
                            description=fallback_description,
                            press_enter_after=press_enter # Also pass press_enter to fallback
                        )
                        if fallback_success:
                            action_result_text = f"Typed '{action.args.text_to_type}' into generic TEXTAREA (fallback)."
                            if press_enter:
                                action_result_text += " Then pressed Enter."
                            action_result = ActionResult(extracted_content=action_result_text)
                            logger.info("Fallback to generic TEXTAREA successful.")
                        else:
                            action_result = ActionResult(error=f"Failed to type '{original_action.args.text_to_type}' into element selected by {original_action.args.selector.strategy}='{original_action.args.selector.value}' AND fallback to generic TEXTAREA also failed.")
                    else:
                        action_result = ActionResult(error=f"Failed to type '{original_action.args.text_to_type}' into element selected by {original_action.args.selector.strategy}='{original_action.args.selector.value}'.")
                    # --- END OF HARDCODED FALLBACK LOGIC ---
                return action_result # type: ignore

            # ... other action types (navigate, click_element, done) ...
            # Ensure these return ActionResult as well

            elif action.action_type == "click_element":
                criteria_func = self._create_criteria_func_from_selector(action.args.selector)
                description=f"element for click (selector: {action.args.selector.strategy}='{action.args.selector.value}')"
                success = await asyncio.to_thread(
                    self.chrome_controller._find_and_click_pychrome,
                     criteria_func,
                     description=description
                )
                if success:
                    return ActionResult(extracted_content=f"Clicked element selected by {action.args.selector.strategy}='{action.args.selector.value}'.")
                else:
                    # --- POTENTIAL FALLBACK FOR CLICKING A BUTTON ---
                    if action.args.selector.node_name == "BUTTON" and \
                       action.args.selector.strategy != "node_type":
                        logger.warning(f"Specific selector for BUTTON failed. Attempting fallback to generic BUTTON.")
                        fallback_selector = ElementSelector(strategy="node_type", value="BUTTON")
                        # ... similar fallback logic as above for clicking ...
                        # This part needs to be filled in if you want this fallback
                        logger.info("Placeholder for BUTTON fallback logic.")

                    return ActionResult(error=f"Failed to click element selected by {action.args.selector.strategy}='{action.args.selector.value}'.")
            
            elif action.action_type == "navigate":
                success = await asyncio.to_thread(self.chrome_controller.navigate, action.args.url)
                if success:
                    return ActionResult(extracted_content=f"Successfully navigated to {action.args.url}")
                else:
                    return ActionResult(error=f"Failed to navigate to {action.args.url}")

            elif action.action_type == "done":
                return ActionResult(is_done=True, success=action.args.success, extracted_content=action.args.conclusion)
            
            elif action.action_type == "save_page_as_pdf":
                output_path = action.args.output_path
                success = await asyncio.to_thread(self.chrome_controller.save_page_as_pdf, output_path)
                if success:
                    return ActionResult(extracted_content=f"Successfully saved page as PDF to {output_path}")
                else:
                    return ActionResult(error=f"Failed to save page as PDF to {output_path}")
            
            else:
                action_type_str = action.action_type if isinstance(action.action_type, str) else "unknown"
                logger.error(f"Unsupported action type: {action_type_str}")
                return ActionResult(error=f"Unsupported action type: {action_type_str}")

        except Exception as e:
            action_type_str = action.action_type if isinstance(action.action_type, str) else "unknown"
            logger.error(f"Error executing action {action_type_str}: {e}\n{traceback.format_exc()}")
            return ActionResult(error=f"Error during action {action_type_str}: {str(e)}")

    async def connect_browser(self) -> bool:
        logger.info("PychromeAgent attempting to connect to browser...")
        success = await asyncio.to_thread(self.chrome_controller.connect)
        if success:
            logger.info("PychromeAgent successfully connected to browser.")
            return True
        logger.error("PychromeAgent failed to connect to browser.")
        return False

    async def disconnect_browser(self) -> None:
        logger.info("PychromeAgent disconnecting from browser...")
        await asyncio.to_thread(self.chrome_controller.disconnect)
        logger.info("PychromeAgent disconnected from browser.")

    async def get_current_browser_state(self) -> Dict[str, Any]:
        """Gathers current state from the browser for LLM context."""
        if not self.chrome_controller.tab:
            logger.warning("Browser not connected or no active tab. Returning empty state.")
            return {
                "url": None,
                "title": None,
                "dom_summary": "Browser not connected.",
                "screenshot_base64": None
            }

        current_url = await asyncio.to_thread(self.chrome_controller.get_current_url)
        page_title = await asyncio.to_thread(self.chrome_controller.get_page_title)
        
        # DOM representation - initial simple version
        # For now, we won't send the full DOM. We'll indicate its availability.
        # Future improvement: Extract interactive elements or provide a summarized DOM tree.
        # dom_root = await asyncio.to_thread(self.chrome_controller.get_dom_root)
        # dom_summary = f"Full DOM is available. Use selectors to target elements." 
        # if dom_root:
        #     # Potentially create a very short summary if needed, e.g., number of children at root
        #     dom_summary = f"DOM root available with {len(dom_root.get('children', []))} children. Use selectors to target elements."
        # else:
        #     dom_summary = "Could not retrieve DOM root."
        # For a very first pass, let's keep it even simpler:
        dom_summary = "The full HTML DOM is available. Use element selectors (aria_label, placeholder, text_content, name) to interact with specific elements. Describe the element you want to interact with clearly."

        screenshot_base64 = None
        # if self.settings.use_vision:
        #     screenshot_base64 = await asyncio.to_thread(self.chrome_controller.capture_screenshot)
        #     if not screenshot_base64:
        #         logger.warning("Failed to capture screenshot.")
        
        browser_state = {
            "url": current_url,
            "title": page_title,
            "dom_summary": dom_summary,
            "screenshot_base64": screenshot_base64 # Ensure this is None if not captured
        }
        # logger.debug(f"Current browser state: URL={current_url}, Title={page_title}, Screenshot captured: {screenshot_base64 is not None}")
        return browser_state

    def _log_llm_response(self, response: AgentOutput) -> None:
        """Utility function to log the model's response."""
        if not response or not hasattr(response, 'current_state') or not response.current_state:
            logger.info("LLM response is empty or not in expected format for logging.")
            return

        if 'Success' in response.current_state.evaluation_previous_goal:
            emoji = '👍'
        elif 'Failed' in response.current_state.evaluation_previous_goal:
            emoji = '⚠'
        else:
            emoji = '🤷'

        logger.info(f'{emoji} Eval: {response.current_state.evaluation_previous_goal}')
        logger.info(f'🧠 Memory: {response.current_state.memory}')
        logger.info(f'🎯 Next goal: {response.current_state.next_goal}')
        if response.action:
            for i, action_item in enumerate(response.action):
                action_dump = action_item.model_dump_json(exclude_unset=True) if hasattr(action_item, 'model_dump_json') else str(action_item)
                logger.info(f'🛠️  Action {i + 1}/{len(response.action)}: {action_dump}')
        else:
            logger.info("🛠️ No actions proposed.")

    async def get_llm_response(self, messages: List[BaseMessage]) -> AgentOutput:
        logger.debug(f"Sending {len(messages)} messages to LLM.")
        
        # Determine tool_calling_method to use (simplified for now)
        # In a more complex scenario, this could be based on LLM type as in original agent
        method_to_use = self.settings.tool_calling_method
        if method_to_use == 'auto':
            # For 'auto', let Langchain infer or use a default. 
            # Or, implement logic here to choose based on self.llm type.
            # For now, defaulting to None for auto, which usually means function calling for capable models.
            method_to_use = None 

        structured_llm = self.llm.with_structured_output(
            self.AgentOutput, 
            include_raw=True, 
            method=method_to_use
        )

        llm_response_dict = await structured_llm.ainvoke(messages)
        
        parsed_output: Optional[AgentOutput] = llm_response_dict.get('parsed')
        raw_output: Optional[BaseMessage] = llm_response_dict.get('raw')

        if parsed_output:
            logger.debug("LLM response parsed successfully by with_structured_output.")
        elif raw_output:
            logger.warning("Failed to parse LLM response with_structured_output. Attempting fallback.")
            try:
                if isinstance(raw_output, AIMessage) and raw_output.tool_calls:
                    logger.debug(f"Attempting to parse from tool_calls: {raw_output.tool_calls}")
                    # Assuming the first tool call's args directly map to AgentOutput schema
                    # This was a pattern in the original agent where AgentOutput itself was the tool
                    if raw_output.tool_calls[0]['name'] == self.AgentOutput.__name__ or raw_output.tool_calls[0]['name'] == 'AgentOutput':
                         # Check if args is already a dict, or needs json.loads
                        args_data = raw_output.tool_calls[0]['args']
                        if isinstance(args_data, str):
                            args_data = json.loads(args_data)
                        parsed_output = self.AgentOutput(**args_data)
                        logger.info("Successfully parsed LLM response from tool_calls arguments.")
                    else:
                        logger.warning(f"Tool call name '{raw_output.tool_calls[0]['name']}' does not match expected AgentOutput.")
                
                if not parsed_output and isinstance(raw_output.content, str):
                    logger.debug(f"Attempting to parse from raw content string: {raw_output.content[:500]}...")
                    json_content = extract_json_from_model_output(raw_output.content)
                    parsed_output = self.AgentOutput(**json_content)
                    logger.info("Successfully parsed LLM response from raw string content.")

            except (ValueError, TypeError, ValidationError, json.JSONDecodeError) as e:
                logger.error(f"Error during fallback parsing of LLM response: {e}\nRaw content was: {raw_output.content[:1000] if isinstance(raw_output.content, str) else 'N/A'}")
                raise ValueError(f"Could not parse LLM response after fallback: {e}") from e
        
        if not parsed_output:
            logger.error(f"Failed to parse LLM response. Raw response: {raw_output}")
            raise ValueError("Could not parse LLM response, and no fallback succeeded.")

        # Truncate actions if necessary
        if parsed_output.action and len(parsed_output.action) > self.settings.max_actions_per_step:
            logger.warning(f"LLM proposed {len(parsed_output.action)} actions, truncating to {self.settings.max_actions_per_step}.")
            parsed_output.action = parsed_output.action[:self.settings.max_actions_per_step]
        
        self._log_llm_response(parsed_output) # Log the final parsed output
        return parsed_output

    async def prepare_input_messages(self, state: Dict[str, Any]) -> List[BaseMessage]:
        """Prepare input messages for the LLM based on current browser state."""
        try:
            # Get existing messages from message manager
            messages = self.message_manager.get_messages()
            
            # Create state message content
            state_lines = []
            
            # Add URL and title if available
            if state.get('url'):
                state_lines.append(f"Current URL: {state['url']}")
            if state.get('title'):
                state_lines.append(f"Page Title: {state['title']}")
            
            # Add DOM summary
            if state.get('dom_summary'):
                state_lines.append(f"DOM Info: {state['dom_summary']}")
            
            # Combine state information
            state_content = "\n".join(state_lines)
            
            # Handle vision if enabled
            if self.settings.use_vision and state.get('screenshot_base64'):
                state_message = [
                    {"type": "text", "text": state_content},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{state['screenshot_base64']}"}
                    }
                ]
            else:
                state_message = state_content
            
            # Add state message to history
            state_human_message = HumanMessage(content=state_message)
            self.message_manager._add_message_with_tokens(state_human_message)
            
            # Get updated messages including the new state
            return self.message_manager.get_messages()
            
        except Exception as e:
            logger.error(f"Error preparing input messages: {e}")
            return []

    async def execute_actions(self, llm_output: AgentOutput) -> List[ActionResult]:
        """Execute a list of actions from the LLM output."""
        results = []
        for action in llm_output.action:
            try:
                result = await self.execute_action(action)
                results.append(result)
                
                # If action failed or marked as done, stop executing further actions
                if result.error or result.is_done:
                    break
                    
            except Exception as e:
                logger.error(f"Error executing action: {e}")
                results.append(ActionResult(error=str(e)))
                break
                
        return results

    async def step(self) -> bool: # Returns True if the agent is done, False otherwise
        """Executes one step of the agent's task."""
        try:
            # Get current state
            state = await self.get_current_browser_state()
            if not state:
                logger.error("Failed to get current state")
                return True

            # Prepare input messages for LLM
            input_messages = await self.prepare_input_messages(state)
            if not input_messages:
                logger.error("Failed to prepare input messages")
                return True

            # Get LLM response
            try:
                llm_output = await self.get_llm_response(input_messages)
            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt during LLM response")
                raise
            except Exception as e:
                logger.error(f"Error getting LLM response: {e}")
                self.state.consecutive_failures += 1
                return self.state.consecutive_failures >= self.settings.max_failures

            # Execute actions
            try:
                results = await self.execute_actions(llm_output)
            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt during action execution")
                raise
            except Exception as e:
                logger.error(f"Error executing actions: {e}")
                self.state.consecutive_failures += 1
                return self.state.consecutive_failures >= self.settings.max_failures

            # Update history
            self.state.history.history.append(AgentHistory(
                model_output=llm_output,
                result=results,
                state=state,
                metadata=None
            ))

            # Check if we're done
            is_done = any(result.is_done for result in results)
            if is_done:
                logger.info("Agent completed its task")
                return True

            # Reset consecutive failures on success
            self.state.consecutive_failures = 0
            self.state.n_steps += 1

            logger.info(f'🚀 Finished Step {self.state.n_steps - 1} ' + '---' * 10)
            return False

        except KeyboardInterrupt:
            raise
        except Exception as e:
            logger.error(f"Unhandled error during step: {e}", exc_info=True)
            self.state.consecutive_failures += 1
            return self.state.consecutive_failures >= self.settings.max_failures

    async def run(self, max_steps: int = 100) -> AgentHistoryList:
        """Runs the agent task for a maximum number of steps."""
        logger.info(f"--- Starting Agent Run: Task: '{self.task}' ---")
        
        try:
            connected = await self.connect_browser()
            if not connected:
                logger.error("Agent run failed: Could not connect to browser.")
                # Return history, potentially adding an error entry
                self.state.history.history.append(AgentHistory(
                    model_output=None,
                    result=[ActionResult(error="Failed to connect to browser.")],
                    state=BrowserStateHistory(raw_state_snapshot={"error": "Failed to connect"}),
                    metadata=None
                ))
                return self.state.history

            for step in range(max_steps):
                try:
                    is_done = await self.step()
                    if is_done:
                        break
                except KeyboardInterrupt:
                    logger.info("Received keyboard interrupt, stopping agent...")
                    break
                except Exception as e:
                    logger.error(f"Error during step {step}: {e}", exc_info=True)
                    self.state.consecutive_failures += 1
                    if self.state.consecutive_failures >= self.settings.max_failures:
                        logger.error(f"Too many consecutive failures ({self.state.consecutive_failures}), stopping agent.")
                        break
                    await asyncio.sleep(1)  # Brief pause before retrying
                    continue

            return self.state.history

        except Exception as e:
            logger.error(f"Fatal error during agent run: {e}", exc_info=True)
            return self.state.history
        finally:
            try:
                await self.disconnect_browser()
            except Exception as e:
                logger.error(f"Error during browser disconnect: {e}")

    # Future methods:
    # (Could add methods for pause/resume, error handling details, memory integration)