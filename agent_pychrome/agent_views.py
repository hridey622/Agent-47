# Pydantic models for agent_pychrome 
from __future__ import annotations

import json
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Type, Union

from langchain_core.language_models.chat_models import BaseChatModel
from openai import RateLimitError
from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

from agent_pychrome.message_manager.message_manager_views import MessageManagerState
# Commented out imports for types now defined as placeholders below
# from browser_use.browser.views import BrowserStateHistory
# from browser_use.controller.registry.views import ActionModel # Removed earlier
# from browser_use.dom.history_tree_processor.service import (
# 	DOMElementNode,
# 	DOMHistoryElement,
# 	HistoryTreeProcessor,
# )
# from browser_use.dom.views import SelectorMap

# Placeholder for BrowserStateHistory, DOMElementNode, DOMHistoryElement, SelectorMap if not defined elsewhere for pychrome
class BrowserStateHistory(BaseModel):
	url: Optional[str] = None
	title: Optional[str] = None
	# tabs: Optional[List[Dict[str, Any]]] = None # Simplified, can be part of raw_state_snapshot
	# interacted_element: Optional[List[Optional[Dict[str, Any]]]] = None # Simplified, can be part of raw_state_snapshot
	screenshot_base64: Optional[str] = None # Storing the base64 screenshot directly
	raw_state_snapshot: Optional[Dict[str, Any]] = None # To store the dict from get_current_browser_state

	model_config = ConfigDict(protected_namespaces=()) # Allow 'screenshot_base64'

	def to_dict(self):
		return self.model_dump()

class DOMElementNode(BaseModel): # Placeholder
	pass

class DOMHistoryElement(BaseModel): # Placeholder
	pass

class SelectorMap(BaseModel): # Placeholder
	pass

ToolCallingMethod = Literal['function_calling', 'json_mode', 'raw', 'auto']
REQUIRED_LLM_API_ENV_VARS = {
	'ChatOpenAI': ['OPENAI_API_KEY'],
	'AzureOpenAI': ['AZURE_ENDPOINT', 'AZURE_OPENAI_API_KEY'],
	'ChatBedrockConverse': ['ANTHROPIC_API_KEY'],
	'ChatAnthropic': ['ANTHROPIC_API_KEY'],
	'ChatGoogleGenerativeAI': ['GEMINI_API_KEY'],
	'ChatDeepSeek': ['DEEPSEEK_API_KEY'],
	'ChatOllama': [],
}

# --- Pychrome Specific Action Models ---
class ElementSelector(BaseModel):
	"""
	Defines how to select an element on the page.
	- strategy: The method to use for selection.
		- "aria_label": Select by the 'aria-label' attribute.
		- "placeholder": Select by the 'placeholder' attribute.
		- "text_content": Select by the text content within the element.
		- "css_selector": Select by CSS selector (Note: direct support in find_element_in_dom is limited).
		- "name": Select by the 'name' attribute.
		- "node_type": Select by the node type (e.g., "TEXTAREA", "INPUT", "BUTTON"). The 'value' field should contain the node type string.
	- value: The value to match for the given strategy (e.g., the aria-label text, placeholder text, CSS selector string, name attribute value, or node type).
	- node_name: Optional. Specifies the expected node name (e.g., "INPUT", "BUTTON", "TEXTAREA"). Useful for refining strategies like "placeholder" or "text_content". For "node_type" strategy, this field is ignored as 'value' holds the node type.
	- exact_match: Optional. If true, requires an exact match for strategies like "placeholder" or "text_content". Defaults to False (substring match).
	"""

	strategy: Literal["aria_label", "placeholder", "text_content", "css_selector", "name", "node_type"]
	value: str
	node_name: Optional[str] = None
	exact_match: bool = False

class NavigateActionArgs(BaseModel):
	url: str = Field(..., description="The URL to navigate to.")

class TypeTextActionArgs(BaseModel):
	selector: ElementSelector = Field(..., description="Selector for the input field.")
	text_to_type: str = Field(..., description="The text to type into the field.")
	press_enter: Optional[bool] = Field(default=False, description="Whether to press Enter after typing the text. Defaults to False. Useful for submitting search queries or forms.")

class ClickElementActionArgs(BaseModel):
	selector: ElementSelector = Field(..., description="Selector for the element to click.")

class SavePageAsPDFActionArgs(BaseModel):
	output_path: str = Field(default="web_page.pdf", description="The file path where the PDF should be saved. Defaults to 'web_page.pdf' in the agent's working directory if not specified.")

class CopyTextActionArgs(BaseModel):
	selector: ElementSelector = Field(..., description="Selector for the element whose text content needs to be copied to the internal clipboard.")

class PasteTextActionArgs(BaseModel):
	selector: ElementSelector = Field(..., description="Selector for the input field where the text from the internal clipboard should be pasted.")
	press_enter: Optional[bool] = Field(default=False, description="Whether to press Enter after pasting the text. Defaults to False.")

class DoneActionArgs(BaseModel):
	success: bool = Field(..., description="Whether the overall task was completed successfully.")
	conclusion: str = Field(..., description="A summary of the task outcome or findings.")

class PychromeNavigateAction(BaseModel):
	action_type: Literal["navigate"] = "navigate"
	args: NavigateActionArgs

class PychromeTypeTextAction(BaseModel):
	action_type: Literal["type_text"] = "type_text"
	args: TypeTextActionArgs

class PychromeClickElementAction(BaseModel):
	action_type: Literal["click_element"] = "click_element"
	args: ClickElementActionArgs

class PychromeSavePageAsPDFAction(BaseModel):
	action_type: Literal["save_page_as_pdf"] = "save_page_as_pdf"
	args: SavePageAsPDFActionArgs

class PychromeCopyTextAction(BaseModel):
	action_type: Literal["copy_text"] = "copy_text"
	args: CopyTextActionArgs

class PychromePasteTextAction(BaseModel):
	action_type: Literal["paste_text"] = "paste_text"
	args: PasteTextActionArgs

class PychromeDoneAction(BaseModel):
	action_type: Literal["done"] = "done"
	args: DoneActionArgs

# --- New Action Model for File Upload ---
class UploadFileActionArgs(BaseModel):
	selector: ElementSelector = Field(..., description="Selector for the file input field (<input type=\"file\">).")
	file_paths: List[str] = Field(..., description="A list of absolute paths to the files to upload.")

class PychromeUploadFileAction(BaseModel):
	action_type: Literal["upload_file"] = "upload_file"
	args: UploadFileActionArgs
# --- End New Action Model ---

PychromeActionModel = Union[
	PychromeNavigateAction,
	PychromeTypeTextAction,
	PychromeClickElementAction,
	PychromeSavePageAsPDFAction,
	PychromeCopyTextAction,
	PychromePasteTextAction,
	PychromeDoneAction,
	PychromeUploadFileAction,
]
# --- End Pychrome Specific Action Models ---


class AgentSettings(BaseModel):
	"""Options for the agent"""

	use_vision: bool = True
	use_vision_for_planner: bool = False
	save_conversation_path: Optional[str] = None
	save_conversation_path_encoding: Optional[str] = 'utf-8'
	max_failures: int = 3
	retry_delay: int = 10
	max_input_tokens: int = 128000
	validate_output: bool = False
	message_context: Optional[str] = None
	generate_gif: bool | str = False
	available_file_paths: Optional[list[str]] = None
	override_system_message: Optional[str] = None
	extend_system_message: Optional[str] = None
	include_attributes: list[str] = [
		'title',
		'type',
		'name',
		'role',
		'tabindex',
		'aria-label',
		'placeholder',
		'value',
		'alt',
		'aria-expanded',
	]
	max_actions_per_step: int = 10

	tool_calling_method: Optional[ToolCallingMethod] = 'auto'
	page_extraction_llm: Optional[BaseChatModel] = None
	planner_llm: Optional[BaseChatModel] = None
	planner_interval: int = 1  # Run planner every N steps
	is_planner_reasoning: bool = False  # type: ignore

	# Procedural memory settings
	enable_memory: bool = True
	memory_interval: int = 10
	memory_config: Optional[dict] = None


class AgentState(BaseModel):
	"""Holds all state information for an Agent"""

	agent_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
	n_steps: int = 1
	consecutive_failures: int = 0
	last_result: Optional[List['ActionResult']] = None
	history: AgentHistoryList = Field(default_factory=lambda: AgentHistoryList(history=[]))
	last_plan: Optional[str] = None
	paused: bool = False
	stopped: bool = False

	message_manager_state: MessageManagerState = Field(default_factory=MessageManagerState)

	# class Config:
	# 	arbitrary_types_allowed = True


@dataclass
class AgentStepInfo:
	step_number: int
	max_steps: int

	def is_last_step(self) -> bool:
		"""Check if this is the last step"""
		return self.step_number >= self.max_steps - 1


class ActionResult(BaseModel):
	"""Result of executing an action"""

	is_done: Optional[bool] = False
	success: Optional[bool] = None
	extracted_content: Optional[str] = None
	error: Optional[str] = None
	include_in_memory: bool = False  # whether to include in past messages as context or not


class StepMetadata(BaseModel):
	"""Metadata for a single step including timing and token information"""

	step_start_time: float
	step_end_time: float
	input_tokens: int  # Approximate tokens from message manager for this step
	step_number: int

	@property
	def duration_seconds(self) -> float:
		"""Calculate step duration in seconds"""
		return self.step_end_time - self.step_start_time


class AgentBrain(BaseModel):
	"""Current state of the agent"""

	evaluation_previous_goal: str
	memory: str
	next_goal: str


class AgentOutput(BaseModel):
	"""Output model for agent

	@dev note: this model is extended with custom actions in AgentService. You can also use some fields that are not in this model as provided by the linter, as long as they are registered in the DynamicActions model.
	"""

	model_config = ConfigDict(arbitrary_types_allowed=True)

	current_state: AgentBrain
	action: list[PychromeActionModel] = Field(
		...,
		description='List of actions to execute',
		json_schema_extra={'min_items': 1},  # Ensure at least one action is provided
	)

	# @staticmethod
	# def type_with_custom_actions(custom_actions: Type[ActionModel]) -> Type['AgentOutput']:
	# 	"""Extend actions with custom actions"""
	# 	# This method might need to be re-evaluated if we need dynamic action registration beyond PychromeActionModel
	# 	# For now, assuming PychromeActionModel is the definitive set of actions.
	# 	model_ = create_model(
	# 		'AgentOutput',
	# 		__base__=AgentOutput,
	# 		action=(
	# 			list[custom_actions],
	# 			Field(..., description='List of actions to execute', json_schema_extra={'min_items': 1}),
	# 		),
	# 		__module__=AgentOutput.__module__,
	# 	)
	# 	model_.__doc__ = 'AgentOutput model with custom actions'
	# 	return model_


class AgentHistory(BaseModel):
	"""History item for agent actions"""

	model_output: AgentOutput | None
	result: list[ActionResult]
	state: BrowserStateHistory
	metadata: Optional[StepMetadata] = None

	model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

	# @staticmethod
	# def get_interacted_element(model_output: AgentOutput, selector_map: SelectorMap) -> list[DOMHistoryElement | None]:
	# 	# This method needs to be re-evaluated for PychromeActionModel.
	# 	# Pychrome actions might not have a simple 'index' field like the previous ActionModel.
	# 	# We will need to determine how to best represent/retrieve interacted elements based on ElementSelector.
	# 	elements = []
	# 	for action in model_output.action: # action is now PychromeActionModel
	# 		# Example: if action.action_type in ['type_text', 'click_element']:
	# 		#   selector_used = action.args.selector
	# 		#   # Logic to map selector_used to a DOMHistoryElement from selector_map
	# 		#   pass 
	# 		# else:
	# 		#   elements.append(None)
	# 	return elements

	def model_dump(self, **kwargs) -> Dict[str, Any]:
		"""Custom serialization handling circular references"""

		# Handle action serialization
		model_output_dump = None
		if self.model_output:
			action_dump = []
			for act_union in self.model_output.action: # act_union is PychromeActionModel
				# Get the actual action model from the Union
				if hasattr(act_union, 'model_dump'): # Should always be true for Pydantic models
					action_dump.append(act_union.model_dump(exclude_none=True))
				else: # Fallback, should not happen
					action_dump.append(str(act_union))
			
			model_output_dump = {
				'current_state': self.model_output.current_state.model_dump(),
				'action': action_dump,  # This preserves the actual action data
			}

		return {
			'model_output': model_output_dump,
			'result': [r.model_dump(exclude_none=True) for r in self.result],
			'state': self.state.model_dump(),
			'metadata': self.metadata.model_dump() if self.metadata else None,
		}


class AgentHistoryList(BaseModel):
	"""List of agent history items"""

	history: list[AgentHistory]

	def total_duration_seconds(self) -> float:
		"""Get total duration of all steps in seconds"""
		total = 0.0
		for h in self.history:
			if h.metadata:
				total += h.metadata.duration_seconds
		return total

	def total_input_tokens(self) -> int:
		"""
		Get total tokens used across all steps.
		Note: These are from the approximate token counting of the message manager.
		For accurate token counting, use tools like LangChain Smith or OpenAI's token counters.
		"""
		total = 0
		for h in self.history:
			if h.metadata:
				total += h.metadata.input_tokens
		return total

	def input_token_usage(self) -> list[int]:
		"""Get token usage for each step"""
		return [h.metadata.input_tokens for h in self.history if h.metadata]

	def __str__(self) -> str:
		"""Representation of the AgentHistoryList object"""
		return f'AgentHistoryList(all_results={self.action_results()}, all_model_outputs={self.model_actions()})'

	def __repr__(self) -> str:
		"""Representation of the AgentHistoryList object"""
		return self.__str__()

	def save_to_file(self, filepath: str | Path) -> None:
		"""Save history to JSON file with proper serialization"""
		try:
			Path(filepath).parent.mkdir(parents=True, exist_ok=True)
			data = self.model_dump()
			with open(filepath, 'w', encoding='utf-8') as f:
				json.dump(data, f, indent=2)
		except Exception as e:
			raise e

	def model_dump(self, **kwargs) -> Dict[str, Any]:
		"""Custom serialization that properly uses AgentHistory's model_dump"""
		return {
			'history': [h.model_dump(**kwargs) for h in self.history],
		}

	@classmethod
	def load_from_file(cls, filepath: str | Path, output_model: Type[AgentOutput]) -> 'AgentHistoryList':
		"""Load history from JSON file"""
		with open(filepath, 'r', encoding='utf-8') as f:
			data = json.load(f)
		# loop through history and validate output_model actions to enrich with custom actions
		for h in data['history']:
			if h['model_output']:
				if isinstance(h['model_output'], dict):
					# Validate against the AgentOutput which now expects PychromeActionModel
					try:
						h['model_output'] = output_model.model_validate(h['model_output'])
					except ValidationError as e:
						print(f"Validation error loading history model_output: {e}") # Or log properly
						h['model_output'] = None # Or handle error more gracefully
				else:
					h['model_output'] = None
			# Placeholder for interacted_element, assuming BrowserStateHistory placeholder is used.
			if h.get('state') and 'interacted_element' not in h['state']:
				h['state']['interacted_element'] = None
		history = cls.model_validate(data)
		return history

	def last_action(self) -> None | dict:
		"""Last action in history"""
		if self.history and self.history[-1].model_output and self.history[-1].model_output.action:
			# Returns the model_dump of the last PychromeActionModel Union member
			return self.history[-1].model_output.action[-1].model_dump(exclude_none=True)
		return None

	def errors(self) -> list[str | None]:
		"""Get all errors from history, with None for steps without errors"""
		errors = []
		for h in self.history:
			step_errors = [r.error for r in h.result if r.error]

			# each step can have only one error
			errors.append(step_errors[0] if step_errors else None)
		return errors

	def final_result(self) -> None | str:
		"""Final result from history"""
		if self.history and self.history[-1].result[-1].extracted_content:
			return self.history[-1].result[-1].extracted_content
		return None

	def is_done(self) -> bool:
		"""Check if the agent is done"""
		if self.history and len(self.history[-1].result) > 0:
			last_result = self.history[-1].result[-1]
			return last_result.is_done is True
		return False

	def is_successful(self) -> bool | None:
		"""Check if the agent completed successfully - the agent decides in the last step if it was successful or not. None if not done yet."""
		if self.history and len(self.history[-1].result) > 0:
			last_result = self.history[-1].result[-1]
			if last_result.is_done is True:
				return last_result.success
		return None

	def has_errors(self) -> bool:
		"""Check if the agent has any non-None errors"""
		return any(error is not None for error in self.errors())

	def urls(self) -> list[str | None]:
		"""Get all unique URLs from history"""
		return [h.state.url if h.state.url is not None else None for h in self.history]

	def screenshots(self) -> list[str | None]:
		"""Get all screenshots from history"""
		return [h.state.screenshot_base64 if h.state.screenshot_base64 is not None else None for h in self.history]

	def action_names(self) -> list[str]:
		"""Get all action types from history"""
		action_types = []
		for h in self.history:
			if h.model_output:
				for action_union in h.model_output.action:
					action_types.append(action_union.action_type)
		return action_types

	def model_thoughts(self) -> list[AgentBrain]:
		"""Get all thoughts from history"""
		return [h.model_output.current_state for h in self.history if h.model_output]

	def model_outputs(self) -> list[AgentOutput]:
		"""Get all model outputs from history"""
		return [h.model_output for h in self.history if h.model_output]

	# get all actions with params
	def model_actions(self) -> list[dict]:
		"""Get all actions from history"""
		outputs = []

		for h in self.history:
			if h.model_output:
				for action_union in h.model_output.action:
					# For PychromeActionModel, we don't have a separate interacted_element in the action itself.
					# This might be associated via AgentHistory.state.interacted_element (which is a placeholder)
					outputs.append(action_union.model_dump(exclude_none=True))
		return outputs

	def action_results(self) -> list[ActionResult]:
		"""Get all results from history"""
		results = []
		for h in self.history:
			results.extend([r for r in h.result if r])
		return results

	def extracted_content(self) -> list[str]:
		"""Get all extracted content from history"""
		content = []
		for h in self.history:
			content.extend([r.extracted_content for r in h.result if r.extracted_content])
		return content

	def model_actions_filtered(self, include: list[str] | None = None) -> list[dict]:
		"""Get all model actions from history as JSON"""
		if include is None:
			include = []
		outputs = self.model_actions()
		result = []
		for o in outputs:
			for i in include:
				if i == list(o.keys())[0]:
					result.append(o)
		return result

	def number_of_steps(self) -> int:
		"""Get the number of steps in the history"""
		return len(self.history)


class AgentError:
	"""Container for agent error handling"""

	VALIDATION_ERROR = 'Invalid model output format. Please follow the correct schema.'
	RATE_LIMIT_ERROR = 'Rate limit reached. Waiting before retry.'
	NO_VALID_ACTION = 'No valid action found'

	@staticmethod
	def format_error(error: Exception, include_trace: bool = False) -> str:
		"""Format error message based on error type and optionally include trace"""
		message = ''
		if isinstance(error, ValidationError):
			return f'{AgentError.VALIDATION_ERROR}\nDetails: {str(error)}'
		if isinstance(error, RateLimitError):
			return AgentError.RATE_LIMIT_ERROR
		if include_trace:
			return f'{str(error)}\nStacktrace:\n{traceback.format_exc()}'
		return f'{str(error)}' 
