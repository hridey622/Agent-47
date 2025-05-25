# ChromeController and pychrome utilities 

# pychrome_utils.py
import pychrome
import time
import json
import traceback
import re # Added import for regular expressions
import base64 # Added for PDF saving
from typing import Optional, List
import inspect # Added for generating automation script
import os # Added for os.linesep


# --- DOM Traversal (Unchanged from your code) ---
def find_element_in_dom(node, criteria_func):
    # ... (your existing find_element_in_dom code)
    if not node: return None
    if criteria_func(node): return node
    children = node.get('children', [])
    for child in children:
        found = find_element_in_dom(child, criteria_func)
        if found: return found
    content_doc = node.get('contentDocument')
    if content_doc:
         found = find_element_in_dom(content_doc, criteria_func)
         if found: return found
    shadow_roots = node.get('shadowRoots', [])
    for shadow_root in shadow_roots:
         found = find_element_in_dom(shadow_root, criteria_func)
         if found: return found
    template_content = node.get('templateContent')
    if template_content:
        found = find_element_in_dom(template_content, criteria_func)
        if found: return found
    return None

# --- Core pychrome interaction functions ---
class ChromeController:

    def __init__(self, cdp_url="http://localhost:9222"):
        self.cdp_url = cdp_url
        self.browser = None
        self.tab = None
        self.action_history = [] # Added to store agent actions

    def connect(self):
        try:
            print(f"Connecting to browser at {self.cdp_url}...")
            self.browser = pychrome.Browser(url=self.cdp_url)
            tabs = self.browser.list_tab()
            if not tabs:
                print("No tabs found. Please ensure Chrome is running with remote debugging.")
                # Try to create a new tab if none exist
                try:
                    print("Attempting to create a new tab...")
                    new_tab_info = self.browser.new_tab()
                    self.tab = self.browser.get_tab(new_tab_info['id']) # Or however pychrome returns it
                    print(f"Created and selected new tab ID: {self.tab.id}")
                except Exception as e_new_tab:
                    print(f"Could not create a new tab: {e_new_tab}")
                    return False
            else:
                self.tab = tabs[0] # Or allow selection

            print(f"Starting interaction with tab ID: {self.tab.id}...")
            self.tab.start()
            self.tab.call_method("Page.enable", _timeout=2) # Good to enable Page domain
            self.tab.call_method("DOM.enable", _timeout=2)
            # Input domain is often implicitly enabled by dispatch methods, but can be explicit
            # self.tab.call_method("Input.enable", _timeout=2)
            print("Connected and domains enabled.")
            return True
        except Exception as e:
            print(f"Error connecting to Chrome: {e}")
            traceback.print_exc()
            return False

    def disconnect(self):
        if self.tab and self.tab.status == pychrome.Tab.status_started:
            print("Stopping tab connection...")
            try:
                self.tab.stop()
                print("Tab connection stopped.")
            except Exception as e:
                print(f"Error stopping tab: {e}")
        self.browser = None
        self.tab = None

    def navigate(self, url):
        if not self.tab: return False
        print(f"Navigating to: {url}...")
        self.action_history.append({'action': 'navigate', 'url': url}) # Log action
        try:
            # Wait for page load to complete
            self.tab.call_method("Page.navigate", url=url, _timeout=10)
            # self.tab.wait_event("Page.loadEventFired", timeout=20) # Commented out problematic line
            time.sleep(5) # Increased sleep to 5 seconds after navigation
            print("Navigation command sent, waited 5s.")
            # Attempt to get URL to confirm navigation context
            current_url_after_nav = self.get_current_url()
            if current_url_after_nav and url in current_url_after_nav:
                print(f"Confirmed navigation to a URL containing: {url}")
                return True
            elif current_url_after_nav:
                print(f"Navigation sent, but current URL is {current_url_after_nav}. Expected {url}.")
                # Might still be true if it's a redirect, but log it.
                return True # Let's be optimistic for now if any URL is returned
            else:
                print(f"Navigation sent, but could not confirm new URL. Assuming it might have worked or is still loading.")
                return True # Still return true, subsequent steps will show if it failed.

        except pychrome.TimeoutException:
            print(f"Timeout during navigation or page load for {url}.")
            return False
        except Exception as e:
            print(f"Error navigating: {e}")
            return False
    # --- DOM Traversal  (Unchanged from your code) ---
    def find_element_in_dom(self, node, criteria_func):
        # ... (your existing find_element_in_dom code)
        if not node: return None
        if criteria_func(node): return node

        children = node.get('children', [])
        for child in children:
            found = find_element_in_dom(child, criteria_func)
            if found: return found
        content_doc = node.get('contentDocument')
        if content_doc:
            found = find_element_in_dom(content_doc, criteria_func)
            if found: return found
        shadow_roots = node.get('shadowRoots', [])
        for shadow_root in shadow_roots:
            found = find_element_in_dom(shadow_root, criteria_func)
            if found: return found
        template_content = node.get('templateContent')
        if template_content:
            found = find_element_in_dom(template_content, criteria_func)
            if found: return found
        return None
    def get_dom_root(self):
        if not self.tab: return None
        print("Getting DOM structure...")
        try:
            result = self.tab.call_method("DOM.getDocument", depth=-1, pierce=True, _timeout=10)
            return result.get('root') if result else None
        except Exception as e:
            print(f"Error getting DOM: {e}")
            return None

    def get_element_box_model(self, node_id=None, backend_node_id=None):
        if not self.tab: return None
        params = {}
        if node_id: params['nodeId'] = node_id
        elif backend_node_id: params['backendNodeId'] = backend_node_id
        else:
            print("Error: get_element_box_model called without nodeId or backendNodeId.")
            return None

        print(f"Attempting to get box model with params: {params}") # Log parameters
        try:
            # Ensure element is in view
            print(f"Scrolling element into view if needed (params: {params})...")
            self.tab.call_method("DOM.scrollIntoViewIfNeeded", **params, _timeout=3)
            print("Scroll command sent.")
            time.sleep(0.2) # give time for scroll

            print(f"Calling DOM.getBoxModel with params: {params}...") # Log before call
            box_model_result = self.tab.call_method("DOM.getBoxModel", **params, _timeout=3)
            print(f"Raw result from DOM.getBoxModel: {box_model_result}") # Log raw result

            # Check for errors in the result itself (CDP errors don't always raise exceptions)
            if not box_model_result or 'error' in box_model_result:
                error_msg = "No result" if not box_model_result else box_model_result.get('error', {}).get('message', 'Unknown CDP error')
                print(f"Error in DOM.getBoxModel result: {error_msg}")
                return None

            model = box_model_result.get('model')
            if not model:
                print("DOM.getBoxModel result did not contain a 'model' key.")
                return None

            print("Successfully retrieved box model.")
            return model
        except pychrome.TimeoutException as e_timeout:
            print(f"Timeout error getting box model: {e_timeout}")
            return None
        except Exception as e:
            print(f"Unexpected error getting box model: {e}")
            traceback.print_exc() # Print traceback for unexpected errors
            return None

    def click_at_coordinates(self, x, y):
        if not self.tab: return False
        print(f"Simulating click at ({x}, {y})...")
        try:
            self.tab.call_method(
                "Input.dispatchMouseEvent", type="mousePressed", x=x, y=y,
                button="left", clickCount=1, _timeout=2
            )
            self.tab.call_method(
                "Input.dispatchMouseEvent", type="mouseReleased", x=x, y=y,
                button="left", clickCount=1, _timeout=2
            )
            print("Click simulated.")
            time.sleep(0.5) # Give page time to react
            return True
        except Exception as e:
            print(f"Error simulating click: {e}")
            return False

    def focus_element(self, node_id=None, backend_node_id=None):
        if not self.tab: return False
        params = {}
        if node_id: params['nodeId'] = node_id
        elif backend_node_id: params['backendNodeId'] = backend_node_id
        else: return False
        print(f"Attempting to focus element (params: {params})...")
        try:
            self.tab.call_method("DOM.focus", **params, _timeout=2)
            print("Focus command sent.")
            time.sleep(0.2)
            return True
        except Exception as e:
            print(f"Warning: Could not explicitly focus element: {e}")
            return False


    def type_text(self, text_to_type):
        if not self.tab: return False
        print(f"Typing text: '{text_to_type}'...")
        self.action_history.append({'action': 'type_text', 'text': text_to_type}) # Log action
        try:
            # One way: insert text directly (might not trigger all JS events)
            self.tab.call_method("Input.insertText", text=text_to_type, _timeout=len(text_to_type) * 0.2 + 1)

            # Alternative: simulate each key press (more realistic, slower)
            # for char in text_to_type:
            #     self.tab.call_method("Input.dispatchKeyEvent", type="keyDown", text=char, _timeout=0.1)
            #     self.tab.call_method("Input.dispatchKeyEvent", type="keyUp", text=char, _timeout=0.1)
            #     time.sleep(0.05) # Small delay between keystrokes
            print("Typing command sent.")
            time.sleep(0.5)
            return True
        except Exception as e:
            print(f"Error typing text: {e}")
            return False

    def scroll_page(self, direction="down"):
        if not self.tab: return False
        print(f"Scrolling page {direction}...")
        self.action_history.append({'action': 'scroll_page', 'direction': direction}) # Log action
        try:
            if direction == "down":
                key_code, key = 34, "PageDown" # PageDown
            elif direction == "up":
                key_code, key = 33, "PageUp" # PageUp
            else: # Scroll to bottom or top might be better with JS
                print(f"Unsupported scroll direction: {direction}")
                return False

            self.tab.call_method("Input.dispatchKeyEvent", type="keyDown", windowsVirtualKeyCode=key_code, nativeVirtualKeyCode=key_code, key=key)
            self.tab.call_method("Input.dispatchKeyEvent", type="keyUp", windowsVirtualKeyCode=key_code, nativeVirtualKeyCode=key_code, key=key)
            print(f"{key} key event sent to scroll.")
            time.sleep(0.3)
            return True
        except Exception as e:
            print(f"Error scrolling: {e}")
            return False

    def get_current_url(self):
        if not self.tab: return None
        try:
            # Method 1: From target info (might not be 100% up-to-date after redirects)
            # return self.tab.target_info.get('url')

            # Method 2: Using Runtime.evaluate (more reliable for current state)
            result = self.tab.call_method("Runtime.evaluate", expression="window.location.href", _timeout=2)
            if result and 'result' in result and result['result'].get('type') == 'string':
                return result['result'].get('value')
            return None
        except Exception as e:
            print(f"Error getting current URL: {e}")
            return None

    def get_page_title(self) -> Optional[str]:
        if not self.tab: return None
        try:
            result = self.tab.call_method("Runtime.evaluate", expression="document.title", _timeout=2)
            if result and 'result' in result and result['result'].get('type') == 'string':
                return result['result'].get('value')
            return None
        except Exception as e:
            print(f"Error getting page title: {e}")
            return None

    # def capture_screenshot(self, format: str = "png", quality: int = 80) -> Optional[str]:
    #     """Captures a screenshot of the current page.
    #
    #     Args:
    #         format (str): Image compression format (jpeg, png, webp).
    #         quality (int): Compression quality from 0 to 100 (jpeg only).
    #
    #     Returns:
    #         Optional[str]: Base64-encoded image data or None if failed.
    #     """
    #     if not self.tab: return None
    #     params = {
    #         "format": format,
    #         "captureBeyondViewport": True # Capture the full scrollable page
    #     }
    #     if format == "jpeg":
    #         params["quality"] = quality
    #     
    #     print(f"Capturing screenshot (format: {format})...")
    #     try:
    #         result = self.tab.call_method("Page.captureScreenshot", **params, _timeout=20)
    #         if result and 'data' in result:
    #             print("Screenshot captured successfully.")
    #             return result['data'] # This is base64 encoded string
    #         print("Failed to capture screenshot, no data in result.")
    #         return None
    #     except Exception as e:
    #         print(f"Error capturing screenshot: {e}")
    #         traceback.print_exc()
    #         return None

    # --- Agent Interaction Helper Methods (adapted from original agent.py) ---
    def _find_and_click_pychrome(self, criteria_func, description="element"):
        dom_root = self.get_dom_root()
        # print(f"DOM root: {dom_root}")
        # print('criteria =', criteria_func)
        if hasattr(criteria_func, '_creation_info'):
            self.action_history.append({
                'action': '_find_and_click_pychrome',
                'criteria_info': criteria_func._creation_info,
                'description': description
            }) # Log action
        else:
            # Fallback if criteria_func doesn't have _creation_info (e.g., custom lambda)
            self.action_history.append({
                'action': '_find_and_click_pychrome',
                'criteria_info': {'name': criteria_func.__name__, 'args': (), 'kwargs': {}}, # best effort
                'description': description
            })

        if not dom_root:
            print(f"Failed to get DOM to find {description}.")
            return False
        print(f"Searching for {description} using criteria: {criteria_func.__name__}...")
        target_node = self.find_element_in_dom(dom_root, criteria_func)
        print(f"Target node: {target_node}")
        if not target_node:
            print(f"Could not find {description}.")
            return False

        # --- New Check: Verify node type is suitable for clicking ---
        node_name = target_node.get('nodeName', '').upper()
        attributes = target_node.get('attributes', [])
        attr_dict = {attributes[i]: attributes[i+1] for i in range(0, len(attributes), 2)} # Convert attributes list to dict

        is_clickable = False
        clickable_tags = ['A', 'BUTTON', 'INPUT', 'SELECT', 'TEXTAREA'] # Common interactive elements
        if node_name in clickable_tags:
            is_clickable = True
        elif attr_dict.get('onclick') or attr_dict.get('role') in ('button', 'link', 'checkbox', 'radio', 'tab', 'menuitem'):
             is_clickable = True
        elif target_node.get('backendNodeId') is not None: # Assume elements with backendNodeId might be interactive
             is_clickable = True # This is a weaker signal, but better than nothing

        # Exclude elements that are typically not clicked despite having some attributes (e.g., hidden inputs)
        if node_name == 'INPUT' and attr_dict.get('type') in ('hidden', 'checkbox', 'radio'):
             is_clickable = False # Handle specific input types that might be better handled differently if needed
        # Add other exclusion rules here if necessary based on common patterns

        if not is_clickable:
            print(f"Found node with name '{node_name}' matching criteria for clicking, but it is not a commonly clickable element.")
            print(f"Node attributes: {attr_dict}")
            # Consider adding a fallback here? E.g., try JS click if coordinate click fails.
            return False
        # --- End New Check ---

        node_id = target_node.get('nodeId')
        backend_node_id = target_node.get('backendNodeId')
        print(f"Found {description}! Node ID: {node_id}, Backend Node ID: {backend_node_id}")

        box_model = self.get_element_box_model(node_id=node_id, backend_node_id=backend_node_id)
        if not box_model:
            print(f"Could not get box model for {description}. Attempting JS or key-based click fallback.")
            focused = self.focus_element(node_id=node_id, backend_node_id=backend_node_id)
            if focused:
                print("Attempting to 'click' by sending Enter key after focus...")
                self.tab.call_method("Input.dispatchKeyEvent", type="keyDown", windowsVirtualKeyCode=13, nativeVirtualKeyCode=13, key="Enter")
                time.sleep(0.1)
                self.tab.call_method("Input.dispatchKeyEvent", type="keyUp", windowsVirtualKeyCode=13, nativeVirtualKeyCode=13, key="Enter")
                time.sleep(1) # Give page time to react
                return True
            return False

        content_quad = box_model.get('content')
        if isinstance(content_quad, list) and len(content_quad) >= 2:
            width = box_model.get('width', 0)
            height = box_model.get('height', 0)
            center_x = int(content_quad[0] + width / 2)
            center_y = int(content_quad[1] + height / 2)

            self.focus_element(node_id=node_id, backend_node_id=backend_node_id) # Focus before clicking
            return self.click_at_coordinates(center_x, center_y)
        else:
            print(f"Could not extract valid coordinates for {description}. Attempting JS or key-based click fallback (as above).")
            focused = self.focus_element(node_id=node_id, backend_node_id=backend_node_id)
            if focused:
                print("Attempting to 'click' by sending Enter key after focus (coordinate fallback)...")
                self.tab.call_method("Input.dispatchKeyEvent", type="keyDown", windowsVirtualKeyCode=13, nativeVirtualKeyCode=13, key="Enter")
                time.sleep(0.1)
                self.tab.call_method("Input.dispatchKeyEvent", type="keyUp", windowsVirtualKeyCode=13, nativeVirtualKeyCode=13, key="Enter")
                time.sleep(1)
                return True
            return False

    def press_enter_key(self):
        """Simulates pressing the Enter key."""
        if not self.tab: return False
        print("Simulating Enter key press...")
        self.action_history.append({'action': 'press_enter_key'}) # Log action
        try:
            self.tab.call_method("Input.dispatchKeyEvent", type="keyDown", windowsVirtualKeyCode=13, nativeVirtualKeyCode=13, key="Enter", _timeout=1)
            time.sleep(0.05) # Small delay between down and up
            self.tab.call_method("Input.dispatchKeyEvent", type="keyUp", windowsVirtualKeyCode=13, nativeVirtualKeyCode=13, key="Enter", _timeout=1)
            print("Enter key press simulated.")
            time.sleep(0.5) # Give page time to react
            return True
        except Exception as e:
            print(f"Error simulating Enter key press: {e}")
            return False

    def _find_and_type_pychrome(self, criteria_func, text_to_type, description="input field", press_enter_after: bool = False):
        dom_root = self.get_dom_root()

        if hasattr(criteria_func, '_creation_info'):
            self.action_history.append({
                'action': '_find_and_type_pychrome',
                'criteria_info': criteria_func._creation_info,
                'text_to_type': text_to_type,
                'description': description,
                'press_enter_after': press_enter_after
            }) # Log action
        else:
            self.action_history.append({
                'action': '_find_and_type_pychrome',
                'criteria_info': {'name': criteria_func.__name__, 'args': (), 'kwargs': {}}, # best effort
                'text_to_type': text_to_type,
                'description': description,
                'press_enter_after': press_enter_after
            })

        if not dom_root:
            print(f"Failed to get DOM to find {description}.")
            return False

        print(f"Searching for {description} using criteria: {criteria_func.__name__}...")
        target_node = find_element_in_dom(dom_root, criteria_func)

        if not target_node:
            print(f"Could not find {description}.")
            return False

        # --- New Check: Verify node type is suitable for typing ---
        node_name = target_node.get('nodeName', '').upper()
        attributes = target_node.get('attributes', [])
        attr_dict = {attributes[i]: attributes[i+1] for i in range(0, len(attributes), 2)} # Convert attributes list to dict

        is_typable = False
        if node_name == 'INPUT':
            input_type = attr_dict.get('type', '').lower()
            if input_type in ('text', 'password', 'search', 'email', 'url', 'tel', '') or 'aria-multiline' in attr_dict:
                 is_typable = True
        elif node_name == 'TEXTAREA':
            is_typable = True
        elif attr_dict.get('contenteditable') == 'true':
             is_typable = True
        elif target_node.get('isContentEditable'): # Check DOM node property if available
            is_typable = True

        if not is_typable:
            print(f"Found node with name '{node_name}' matching criteria for typing, but it is not a typable element.")
            print(f"Node attributes: {attr_dict}")
            return False
        # --- End New Check ---

        node_id = target_node.get('nodeId')
        backend_node_id = target_node.get('backendNodeId')
        print(f"Found {description}! Node ID: {node_id}, Backend Node ID: {backend_node_id}")

        if node_id is None and backend_node_id is None:
            print("Found node has neither nodeId nor backendNodeId. Cannot proceed.")
            return False

        # Always attempt to click the element before typing
        box_model = self.get_element_box_model(node_id=node_id, backend_node_id=backend_node_id)
        if not box_model:
            print(f"Could not get box model for {description}. Cannot click.")
            return False

        content_quad = box_model.get('content')
        if isinstance(content_quad, list) and len(content_quad) >= 2:
            width = box_model.get('width', 0)
            height = box_model.get('height', 0)
            center_x = int(content_quad[0] + width / 2)
            center_y = int(content_quad[1] + height / 2)
            print(f"Clicking {description} at coordinates: ({center_x}, {center_y})...")
            clicked = self.click_at_coordinates(center_x, center_y)
            if not clicked:
                print(f"Failed to click {description} at ({center_x}, {center_y}).")
                return False
            time.sleep(0.3)  # Brief pause after click before typing
        else:
            print(f"Could not get valid coordinates from box model for {description}. Cannot click.")
            return False

        # Proceed to type only if click succeeded
        type_success = self.type_text(text_to_type)
        if type_success and press_enter_after:
            print(f"Successfully typed. Now pressing Enter as requested for {description}.")
            return self.press_enter_key()
        return type_success

    def save_page_as_pdf(self, output_path: str = 'weboutput.pdf'):
        """
        Saves the current page in the tab as a PDF file.

        Args:
            output_path (str): Path to save the output PDF.
        """
        if not self.tab:
            print("Error: Tab not available for saving PDF.")
            return False
        self.action_history.append({'action': 'save_page_as_pdf', 'output_path': output_path}) # Log action
        try:
            print("Ensuring Page domain is enabled for PDF generation...")
            # Page.enable is usually called at connection, but good to ensure
            # self.tab.call_method("Page.enable", _timeout=2) # Potentially redundant if enabled at start

            print(f"Calling Page.printToPDF to generate PDF data (output: {output_path})...")
            # Consider making printBackground, paperWidth, paperHeight, etc. configurable if needed
            result = self.tab.call_method(
                "Page.printToPDF", 
                printBackground=True,
                # landscape=False, 
                # displayHeaderFooter=False,
                # paperWidth=8.5, # inches
                # paperHeight=11, # inches
                # marginTop=0.4, # inches
                # marginBottom=0.4, # inches
                # marginLeft=0.4, # inches
                # marginRight=0.4, # inches
                _timeout=30 # Increased timeout for potentially large pages
            )
            
            pdf_data_b64 = result.get('data')
            if not pdf_data_b64:
                print("Failed to get PDF data from Page.printToPDF result.")
                return False

            with open(output_path, "wb") as f:
                f.write(base64.b64decode(pdf_data_b64))
            print(f"Page successfully saved as PDF to: {output_path}")
            return True

        except pychrome.TimeoutException as te:
            print(f"Timeout error while saving PDF: {te}")
            traceback.print_exc()
            return False
        except Exception as e:
            print(f"Unexpected error while saving PDF: {e}")
            traceback.print_exc()
            return False

    async def get_element_text(self, criteria_func, description="element"):
        """Finds an element and returns its innerText content."""
        if not self.tab:
            print("Error: Tab not available for getting element text.")
            return None

        print(f"Searching for {description} to get its text using criteria: {criteria_func.__name__}...")
        dom_root = self.get_dom_root()
        if not dom_root:
            print(f"Failed to get DOM to find {description} for text extraction.")
            return None

        target_node = find_element_in_dom(dom_root, criteria_func)
        if not target_node:
            print(f"Could not find {description} for text extraction.")
            return None

        node_id = target_node.get('nodeId')
        backend_node_id = target_node.get('backendNodeId')
        object_id_to_use = None

        print(f"Found {description} for text extraction. Node ID: {node_id}, Backend Node ID: {backend_node_id}")

        try:
            # First, try to get an ObjectId for the node.
            # DOM.resolveNode is useful if you only have nodeId or backendNodeId.
            if node_id:
                resolved_node = self.tab.call_method("DOM.resolveNode", nodeId=node_id, _timeout=3)
                object_id_to_use = resolved_node.get('object', {}).get('objectId')
            elif backend_node_id:
                resolved_node = self.tab.call_method("DOM.resolveNode", backendNodeId=backend_node_id, _timeout=3)
                object_id_to_use = resolved_node.get('object', {}).get('objectId')

            if not object_id_to_use:
                # Fallback or if DOM.resolveNode doesn't yield objectId directly (e.g. for pseudo-elements, though less relevant for innerText)
                # For simple nodes, trying to describe it might give an objectId in some implementations, or we can use Runtime.evaluate with a querySelector if we can build one.
                # However, the most direct path is DOM.resolveNode for regular nodes.
                print(f"Could not resolve node to an objectId for {description}.")
                # As a simpler fallback, let's try to get outerHTML and let the LLM parse if it's desperate
                # This is not ideal for clean text extraction but better than nothing.
                try:
                    html_result = self.tab.call_method("DOM.getOuterHTML", nodeId=node_id if node_id else backend_node_id, _timeout=3)
                    outer_html = html_result.get('outerHTML')
                    if outer_html:
                        print(f"Returning OuterHTML for {description} as text fallback.")
                        return outer_html # Not ideal, but a fallback
                except Exception as e_html:
                    print(f"Could not get OuterHTML as fallback for {description}: {e_html}")
                return None

            # Use Runtime.callFunctionOn to get innerText
            print(f"Calling Runtime.callFunctionOn to get innerText for objectId: {object_id_to_use}")
            function_declaration = "function() { return this.innerText; }"
            result = self.tab.call_method(
                "Runtime.callFunctionOn",
                functionDeclaration=function_declaration,
                objectId=object_id_to_use,
                returnByValue=True, # We want the actual text value
                _timeout=5
            )

            if result and 'result' in result and result['result'].get('type') == 'string':
                element_text = result['result'].get('value')
                print(f"Successfully extracted text for {description}: '{element_text[:100]}...'")
                return element_text
            else:
                error_details = result.get('exceptionDetails', 'No specific error details') if result else 'No result from callFunctionOn'
                print(f"Failed to get innerText using Runtime.callFunctionOn for {description}. Result: {result}. Error: {error_details}")
                # Check if it was a type error, indicating it might not be an element that has innerText
                if isinstance(error_details, dict) and 'SyntaxError' in error_details.get('text',''):
                    print(f"Syntax error in JS function. This is an agent dev issue.")
                elif result and result['result'].get('subtype') == 'error':
                     print(f"JS execution error: {result['result'].get('description')}")

        except pychrome.TimeoutException as te:
            print(f"Timeout error getting text for {description}: {te}")
            return None
        except Exception as e:
            print(f"Unexpected error getting text for {description}: {e}")
            traceback.print_exc()
            return None
        
        return None # Should be unreachable if logic is correct or fallbacks handle it

    # --- New Method for File Upload ---
    def upload_file(self, node_id: int, file_paths: List[str]) -> bool:
        """Selects files for a file input element using DOM.setFileInputFiles."""
        if not self.tab:
            print("Error: Tab not available for file upload.")
            return False

        if not file_paths:
            print("Error: No file paths provided for upload.")
            return False

        print(f"Attempting to upload files {file_paths} to node ID {node_id}...")
        self.action_history.append({'action': 'upload_file', 'node_id': node_id, 'file_paths': file_paths}) # Log action

        try:
            # Use DOM.setFileInputFiles to set the files for the input element
            # The node ID must be for an <input type=file> element
            self.tab.call_method("DOM.setFileInputFiles", nodeId=node_id, files=file_paths, _timeout=10)
            print(f"Successfully set file input for node ID {node_id}.")
            time.sleep(0.5) # Give browser a moment to process
            return True
        except Exception as e:
            print(f"Error setting file input for node ID {node_id}: {e}")
            traceback.print_exc()
            return False
    # --- End New Method ---

    def generate_automation_script(self, output_file_path="automated_task.py"):
        """
        Generates a Python script that replays the actions performed by the agent.
        """
        script_lines = [
            "import pychrome",
            "import time",
            "import json",
            "import traceback",
            "import re",
            "import base64",
            "from typing import Optional, List",
            "",
            "# --- DOM Traversal (Copied from chrome_controller.py) ---",
            "def find_element_in_dom(node, criteria_func):",
            "    if not node: return None",
            "    if criteria_func(node): return node",
            "    children = node.get('children', [])",
            "    for child in children:",
            "        found = find_element_in_dom(child, criteria_func)",
            "        if found: return found",
            "    content_doc = node.get('contentDocument')",
            "    if content_doc:",
            "         found = find_element_in_dom(content_doc, criteria_func)",
            "         if found: return found",
            "    shadow_roots = node.get('shadowRoots', [])",
            "    for shadow_root in shadow_roots:",
            "         found = find_element_in_dom(shadow_root, criteria_func)",
            "         if found: return found",
            "    template_content = node.get('templateContent')",
            "    if template_content:",
            "        found = find_element_in_dom(template_content, criteria_func)",
            "        if found: return found",
            "    return None",
            "",
            "# --- Criteria Functions (Copied from chrome_controller.py) ---"
        ]

        # Add criteria function definitions to the script
        # This assumes they are defined globally or can be easily extracted.
        # For a robust solution, you might need to inspect the source of these functions.
        criteria_functions_source = [
            inspect.getsource(create_placeholder_criteria),
            inspect.getsource(create_text_criteria),
            inspect.getsource(create_aria_label_criteria),
            inspect.getsource(create_name_criteria),
            inspect.getsource(create_node_type_criteria)
        ]
        for func_source in criteria_functions_source:
            script_lines.append(func_source)
            script_lines.append("")

        # Add ChromeController class definition (simplified or full)
        # For now, let's add a placeholder for the class.
        # A more robust way would be to copy the class definition,
        # or ensure the generated script can import it.
        script_lines.append("# --- ChromeController Class (Adapted from chrome_controller.py) ---")
        script_lines.append(inspect.getsource(ChromeController)) # Add the class source
        script_lines.append("")

        script_lines.append("def main():")
        script_lines.append("    controller = ChromeController()") # Assuming default CDP URL
        script_lines.append("    if not controller.connect():")
        script_lines.append("        print('Failed to connect to Chrome. Exiting.')")
        script_lines.append("        return")
        script_lines.append("")
        script_lines.append("    try:")

        for action_item in self.action_history:
            action = action_item['action']
            script_lines.append(f"        print(f'Performing action: {action} with params {action_item}')") # Debug print in generated script

            if action == 'navigate':
                script_lines.append(f"        controller.navigate(url='{action_item['url']}')")
            elif action == 'type_text':
                # Escape quotes in text_to_type for the generated script
                text_to_type_escaped = action_item['text'].replace("'", "\\\\'")
                script_lines.append(f"        controller.type_text(text_to_type='{text_to_type_escaped}')")
            elif action == 'press_enter_key':
                script_lines.append("        controller.press_enter_key()")
            elif action == 'scroll_page':
                script_lines.append(f"        controller.scroll_page(direction='{action_item['direction']}')")
            elif action == 'save_page_as_pdf':
                script_lines.append(f"        controller.save_page_as_pdf(output_path='{action_item['output_path']}')")
            elif action == '_find_and_click_pychrome' or action == '_find_and_type_pychrome':
                criteria_info = action_item['criteria_info']
                func_name = criteria_info['name']
                args_repr = ', '.join([repr(arg) for arg in criteria_info.get('args', [])])
                kwargs_repr = ', '.join([f"{k}={repr(v)}" for k, v in criteria_info.get('kwargs', {}).items()])
                
                # Filter out empty strings from args_repr and kwargs_repr before joining
                criteria_params_list = [repr_ for repr_ in [args_repr, kwargs_repr] if repr_]
                criteria_params = ', '.join(criteria_params_list)

                script_lines.append(f"        criteria_func = {func_name}({criteria_params})")
                
                if action == '_find_and_click_pychrome':
                    script_lines.append(f"        controller._find_and_click_pychrome(criteria_func, description='{action_item['description']}')")
                elif action == '_find_and_type_pychrome':
                    text_to_type_escaped = action_item['text_to_type'].replace("'", "\\\\'")
                    script_lines.append(f"        controller._find_and_type_pychrome(criteria_func, text_to_type='{text_to_type_escaped}', description='{action_item['description']}', press_enter_after={action_item['press_enter_after']})")
            script_lines.append("        time.sleep(1) # Add a small delay between actions")


        script_lines.append("    except Exception as e:")
        script_lines.append("        print(f'An error occurred during automation: {e}')")
        script_lines.append("        traceback.print_exc()")
        script_lines.append("    finally:")
        script_lines.append("        print('Automation script finished. Disconnecting...')")
        script_lines.append("        controller.disconnect()")
        script_lines.append("")
        script_lines.append("if __name__ == '__main__':")
        script_lines.append("    import inspect # Ensure inspect is imported for the script if class source is used")
        script_lines.append("    main()")

        try:
            with open(output_file_path, 'w', encoding='utf-8') as f:
                f.write(os.linesep.join(script_lines))
            print(f"Automation script successfully generated: {output_file_path}")
        except Exception as e:
            print(f"Error writing automation script: {e}")
            traceback.print_exc()

# --- Example Criteria Function (can be moved or defined by agent) ---
def check_contenteditable(node, attr_dict):
    """Check if node has contenteditable="true"."""
    if node.get('nodeName') != 'DIV':
        return False
    if attr_dict.get('contenteditable') == 'true':
        print('Found contenteditable element')
        print(f"Node attributes: {attr_dict}")
        return True
    return False

def check_textarea(node, attr_dict):
    """Check if node is a textarea."""
    if node.get('nodeName', '').upper() == 'TEXTAREA':
        print('Found textarea element')
        print(f"Node attributes: {attr_dict}")
        return True
    return False

def check_text_input(node, attr_dict):
    """Check if node is an input[type="text"] or input without type."""
    if node.get('nodeName', '').upper() == 'INPUT':
        input_type = attr_dict.get('type', '').lower()
        if input_type == 'text' or input_type == '':
            print('Found text input element')
            print(f"Node attributes: {attr_dict}")
            return True
    return False

def check_textbox_role(node, attr_dict):
    """Check if node has role="textbox"."""
    if attr_dict.get('role') == 'textbox':
        print('Found element with role="textbox"')
        print(f"Node attributes: {attr_dict}")
        return True
    return False

def check_aria_label(node, attr_dict):
    """Check if node has aria-label."""
    if 'aria-label' in attr_dict:
        print('Found element with aria-label')
        print(f"Node attributes: {attr_dict}")
        return True
    return False

def check_placeholder(node, attr_dict):
    """Check if node has placeholder."""
    if 'placeholder' in attr_dict:
        print('Found element with placeholder')
        print(f"Node attributes: {attr_dict}")
        return True
    return False

def check_input_prompt(node, attr_dict):
    """Check if node has input/prompt in class/id/name."""
    class_value = attr_dict.get('class', '').lower()
    id_value = attr_dict.get('id', '').lower()
    name_value = attr_dict.get('name', '').lower()
    
    if 'input' in class_value or 'input' in id_value or 'prompt' in name_value:
        print('Found element with input/prompt in class/id/name')
        print(f"Node attributes: {attr_dict}")
        return True
    return False

def create_placeholder_criteria(placeholder_text, node_name=None):
    """Creates criteria function for element with exact placeholder match (case-insensitive)."""
    def check_all_nodes_for_criteria(node, criteria_func, attr_dict):
        """Helper function to check all nodes against a specific criteria"""
        if not node:
            return False
            
        # Check current node
        if criteria_func(node, attr_dict):
            return True
            
        # Check children
        children = node.get('children', [])
        for child in children:
            child_attrs = {}
            try:
                for i in range(0, len(child.get('attributes', [])), 2):
                    child_attrs[child['attributes'][i]] = child['attributes'][i+1]
            except IndexError:
                continue
                
            if check_all_nodes_for_criteria(child, criteria_func, child_attrs):
                return True
                
        # Check content document
        content_doc = node.get('contentDocument')
        if content_doc:
            if check_all_nodes_for_criteria(content_doc, criteria_func, attr_dict):
                return True
                
        # Check shadow roots
        shadow_roots = node.get('shadowRoots', [])
        for shadow_root in shadow_roots:
            if check_all_nodes_for_criteria(shadow_root, criteria_func, attr_dict):
                return True
                
        # Check template content
        template_content = node.get('templateContent')
        if template_content:
            if check_all_nodes_for_criteria(template_content, criteria_func, attr_dict):
                return True
                
        return False

    def criteria(node):
        # Get attributes from node
        attributes = node.get('attributes', [])
        
        # Convert attributes list to dict for easier lookup
        attr_dict = {}
        try:
            for i in range(0, len(attributes), 2):
                print('attributes ==', attributes[0])
                attr_dict[attributes[i]] = attributes[i+1]
        except IndexError:
            print(f"Warning: Malformed attributes list for node ID {node.get('nodeId')}")
            return False

        # Check all nodes for each criteria in sequence
        print("Checking all nodes for contenteditable...")
        if check_all_nodes_for_criteria(node, check_contenteditable, attr_dict):
            return True
            
        print("Checking all nodes for textarea...")
        if check_all_nodes_for_criteria(node, check_textarea, attr_dict):
            return True
            
        print("Checking all nodes for text input...")
        if check_all_nodes_for_criteria(node, check_text_input, attr_dict):
            return True
            
        print("Checking all nodes for textbox role...")
        if check_all_nodes_for_criteria(node, check_textbox_role, attr_dict):
            return True
            
        print("Checking all nodes for aria label...")
        if check_all_nodes_for_criteria(node, check_aria_label, attr_dict):
            return True
            
        print("Checking all nodes for placeholder...")
        if check_all_nodes_for_criteria(node, check_placeholder, attr_dict):
            return True
            
        print("Checking all nodes for input prompt...")
        if check_all_nodes_for_criteria(node, check_input_prompt, attr_dict):
            return True

        return False

    # Safely generate name
    base_name = f"with_exact_placeholder_{placeholder_text.replace(' ', '_')}"
    criteria.__name__ = f"is_{node_name.lower()}_{base_name}" if node_name else f"element_{base_name}"
    criteria._creation_info = {
        'name': 'create_placeholder_criteria',
        'args': (placeholder_text,),
        'kwargs': {'node_name': node_name}
    } # Log creation info
    return criteria

def create_text_criteria(text_content, node_name=None, exact_match=False):
    def criteria(node):
        if node_name and node.get('nodeName') != node_name.upper():
            return False
        node_text = ""
        if node.get('nodeType') == 3: 
            node_text = node.get('nodeValue', '').strip()
        elif 'children' in node:
            for child in node['children']:
                if child.get('nodeType') == 3:
                    node_text += child.get('nodeValue', '').strip() + " "
            node_text = node_text.strip()
        if exact_match:
            return text_content.lower() == node_text.lower()
        else:
            return text_content.lower() in node_text.lower()
    base_name = f"contains_text_{text_content.replace(' ', '_')}"
    criteria.__name__ = f"is_{node_name.lower()}_{base_name}" if node_name else f"element_{base_name}"
    criteria._creation_info = {
        'name': 'create_text_criteria',
        'args': (text_content,),
        'kwargs': {'node_name': node_name, 'exact_match': exact_match}
    } # Log creation info
    return criteria

def create_aria_label_criteria(label_text, node_name=None):
    def criteria(node):
        if node_name and node.get('nodeName') != node_name.upper(): return False
        attributes = node.get('attributes', [])
        try:
            for i in range(0, len(attributes), 2):
                if attributes[i] == 'aria-label' and \
                   attributes[i+1].strip().lower() == label_text.lower():
                    return True
        except IndexError:
            pass
        return False
    base_name = f"has_aria_label_{label_text.replace(' ', '_').replace('.', '_').replace(':', '_')}"
    criteria.__name__ = f"is_{node_name.lower()}_with_aria_label_{label_text.replace(' ', '_').replace('.', '_').replace(':', '_')}"
    if node_name:
        criteria.__name__ = f"is_{node_name.lower()}_with_aria_label_{label_text.replace(' ', '_').replace('.', '_').replace(':', '_')}"
    criteria._creation_info = {
        'name': 'create_aria_label_criteria',
        'args': (label_text,),
        'kwargs': {'node_name': node_name}
    } # Log creation info
    return criteria

def create_name_criteria(name_value, node_name=None, exact_match=False):
    """Creates a criteria function to find an element by its 'name' attribute."""
    def criteria(node):
        if node_name and node.get('nodeName') != node_name.upper(): 
            return False
        attributes = node.get('attributes', [])
        try:
            for i in range(0, len(attributes), 2):
                if attributes[i] == 'name' and attributes[i+1] == name_value:
                    return True
        except IndexError:
            pass 
        return False
    base_name = f"has_name_{name_value.replace(' ', '_')}"
    criteria.__name__ = f"is_{node_name.lower()}_{base_name}" if node_name else f"element_{base_name}"
    criteria._creation_info = {
        'name': 'create_name_criteria',
        'args': (name_value,),
        'kwargs': {'node_name': node_name, 'exact_match': exact_match}
    } # Log creation info
    return criteria 

def create_node_type_criteria(node_type_to_find: str):
    """Creates a criteria function to find an element by its nodeName."""
    def criteria(node):
        return node.get('nodeName', '').upper() == node_type_to_find.upper()
    criteria.__name__ = f"is_node_type_{node_type_to_find.upper().replace(' ', '_')}"
    criteria._creation_info = {
        'name': 'create_node_type_criteria',
        'args': (node_type_to_find,),
        'kwargs': {}
    } # Log creation info
    return criteria 
