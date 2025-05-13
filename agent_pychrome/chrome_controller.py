# ChromeController and pychrome utilities 

# pychrome_utils.py
import pychrome
import time
import json
import traceback
import re # Added import for regular expressions
import base64 # Added for PDF saving
from typing import Optional


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
        print('criteria =', criteria_func)
        if not dom_root:
            print(f"Failed to get DOM to find {description}.")
            return False
        print(f"Searching for {description} using criteria: {criteria_func.__name__}...")
        target_node = self.find_element_in_dom(dom_root, criteria_func)
        print(f"Target node: {target_node}")
        if not target_node:
            print(f"Could not find {description}.")
            return False

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
        if not dom_root:
            print(f"Failed to get DOM to find {description}.")
            return False

        print(f"Searching for {description} using criteria: {criteria_func.__name__}...")
        target_node = find_element_in_dom(dom_root, criteria_func)

        if not target_node:
            print(f"Could not find {description}.")
            return False

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

# --- Example Criteria Function (can be moved or defined by agent) ---
def create_placeholder_criteria(placeholder_text, node_name=None):
    """Creates criteria function for element with exact placeholder match (case-insensitive)."""
    def criteria(node):
        # Check node_name first if provided
        # attributes = node.get('attributes', [])
        # print('A1,A2',attributes[i],attributes[i+1],'node name = ',node.get('nodeName'), 'node id =',node.get('nodeId'))
        # if node.get('nodeId') > 300:
        #     return False
        if node.get('nodeName') != 'TEXTAREA':
            return False
        print("NODE =",node.get('nodeName'))
        attributes = node.get('attributes', [])
        try:
        
            for i in range(0, len(attributes), 2):
                # Exact match for placeholder after stripping and lowercasing
                # print(placeholder_text.lower())
                if attributes[i] == 'placeholder':
                    dom_placeholder_original = attributes[i+1]
                    
                    # Normalize DOM placeholder value
                    norm_dom_placeholder = dom_placeholder_original.lower()
                    norm_dom_placeholder = norm_dom_placeholder.replace("...", "…") # Normalize three dots to ellipsis char
                    norm_dom_placeholder = re.sub(r'\s+', ' ', norm_dom_placeholder).strip() # Normalize whitespace

                    # Normalize search placeholder text
                    norm_search_placeholder = placeholder_text.lower()
                    norm_search_placeholder = norm_search_placeholder.replace("...", "…") # Normalize three dots to ellipsis char
                    norm_search_placeholder = re.sub(r'\s+', ' ', norm_search_placeholder).strip() # Normalize whitespace
                    
                    # Updated debug prints for clarity
                    print(f"+_+_+_== Original DOM placeholder (stripped, lower): '{dom_placeholder_original.strip().lower()}'")
                    print(f"----=== Normalized DOM placeholder: '{norm_dom_placeholder}'")
                    print(f"____=== Original search placeholder (lower): '{placeholder_text.lower()}'")
                    print(f"----=== Normalized search placeholder: '{norm_search_placeholder}'")

                    if norm_dom_placeholder.startswith(norm_search_placeholder):
                        # Corrected 'attribute' to 'attributes' in the success print and improved clarity
                        print(f"MATCH FOUND: Node ID {node.get('nodeId')}, Attribute '{attributes[i]}'='{attributes[i+1]}'")
                        return True
        except IndexError:
            print('ERROR')
            pass
        return False
    # Safely generate name
    base_name = f"with_exact_placeholder_{placeholder_text.replace(' ', '_')}"
    criteria.__name__ = f"is_{node_name.lower()}_{base_name}" if node_name else f"element_{base_name}"
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
    return criteria 

def create_node_type_criteria(node_type_to_find: str):
    """Creates a criteria function to find an element by its nodeName."""
    def criteria(node):
        return node.get('nodeName', '').upper() == node_type_to_find.upper()
    criteria.__name__ = f"is_node_type_{node_type_to_find.upper().replace(' ', '_')}"
    return criteria 