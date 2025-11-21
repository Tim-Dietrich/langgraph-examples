from typing import TypedDict

from langgraph.graph import END, StateGraph
from langchain_core.messages import (
    AIMessage,
    FunctionMessage,
)
from enum import Enum
import requests
from typing import Dict, List, Any, Optional
from IPython.display import Image, display
from PIL import Image as PILImage
import logging

from rapidfuzz import fuzz, process

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# LOGGING with colorformatter


class ColorFormatter(logging.Formatter):
    def format(self, record):
        if record.levelno == logging.INFO:
            return f"\033[92m{record.msg}\033[0m"  # green
        elif record.levelno == logging.WARNING:
            return f"\033[93m{record.msg}\033[0m"  # yellow
        elif record.levelno == logging.ERROR:
            return f"\033[91m{record.msg}\033[0m"  # red


formatter = ColorFormatter()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)


class ChatbotState(TypedDict):
    """
    Messages have the type "list". The `add_messages` function
    in the annotation defines how this state key should be updated
    (in this case, it appends messages to the list, rather than overwriting them)
    """
    input: str
    slots: dict
    messages: list
    active_order: bool
    ended: bool


class Nodes(Enum):
    ENTRY = "entry"
    CHECKER = "checker"
    ORDER_FORM = "order_form"
    RETRIEVAL = "retrieval"
    END = "end"


class OrderSlots(Enum):
    PIZZA_NAME = "pizza_name"
    PIZZA_COUNT = "pizza_count"
    CUSTOMER_ADDRESS = "customer_address"


class CheckerNode:
    """
    This node checks whether user input is valid
    """
    
    def __init__(self, keywords=None):
        if keywords is None:
            keywords = ["order", "pizza"]
        self.keywords = keywords

    def invoke(self, state: ChatbotState) -> dict[str, list | dict | bool]:
        """
        Checks whether the input is a valid request for pizza order
        """
        if state['active_order']:
            return {
                "messages": state["messages"],
                "slots": state["slots"],
                "active_order": state["active_order"],
                "ended": state["ended"]
            }

        _input = state['input'].lower()
        if not all(keyword in _input for keyword in self.keywords):
            state['messages'].append(AIMessage(
                content="Invalid order. Please specify a pizza order. Try writing 'I want to order a pizza'."))
            return {
                "messages": state["messages"],
                "slots": state["slots"],
                "active_order": state["active_order"],
                "ended": state["ended"]
            }
        else:
            state['active_order'] = True
            # state['messages'].append(AIMessage(content="Your pizza order is valid."))
            return {
                "messages": state["messages"],
                "slots": state["slots"],
                "active_order": state["active_order"],
                "ended": state["ended"]
            }

    def route(self, state: ChatbotState) -> str:
        """
        Routes to the next node
        """
        if state['active_order']:
            logger.info("Routing to retrieval node")
            return Nodes.RETRIEVAL.value
        else:
            logger.info("Routing to end node")
            return END


class OrderNode:
    """
    Collects the slots for the pizza order
    """

    def __init__(self):
        pass

    def invoke(self, state: ChatbotState) -> dict[str, list | dict | bool] | None:
        """
        Returns fallback message
        """

        required_slots = [OrderSlots.PIZZA_NAME, OrderSlots.PIZZA_COUNT, OrderSlots.CUSTOMER_ADDRESS]
        missing_slots = [
            slot.value for slot in required_slots if slot.value not in state['slots'].keys()]

        if not missing_slots:
            state['messages'].append(AIMessage(
                "Thank you for providing all the details. Your order is being processed!"))
            state['ended'] = True
            logger.info("Order completed")
            return {
                "messages": state["messages"],
                "slots": state["slots"],
                "ended": state["ended"]
            }

        next_slot = missing_slots[0]
        last_message = state["messages"][-1] if len(state["messages"]) > 0 else None

        if next_slot == OrderSlots.PIZZA_NAME.value:
            if last_message is None or last_message.content != OrderSlots.PIZZA_NAME.value:
                print("DEBUG: pizza name missing, appending")
                state['messages'].append(AIMessage("What pizza would you like to order?"))
                state["messages"].append(FunctionMessage(content=OrderSlots.PIZZA_NAME, name=OrderSlots.PIZZA_NAME.value))
                return {
                    "messages": state["messages"],
                    "slots": state["slots"],
                    "ended": state["ended"]
                }

        elif next_slot == OrderSlots.PIZZA_COUNT.value:
            if last_message is None or last_message.content != OrderSlots.PIZZA_COUNT.value:
                state['messages'].append(AIMessage("How many Pizza's do you want to order?"))
                state["messages"].append(FunctionMessage(content=OrderSlots.PIZZA_COUNT, name=OrderSlots.PIZZA_COUNT.value))
                return {
                    "messages": state["messages"],
                    "slots": state["slots"],
                    "ended": state["ended"]
                }

        elif next_slot == OrderSlots.CUSTOMER_ADDRESS.value:
            if last_message is None or last_message.content != OrderSlots.CUSTOMER_ADDRESS.value:
                state['messages'].append(AIMessage("What is your delivery address?"))
                state["messages"].append(FunctionMessage(content=OrderSlots.CUSTOMER_ADDRESS, name=OrderSlots.CUSTOMER_ADDRESS.value))
                return {
                    "messages": state["messages"],
                    "slots": state["slots"],
                    "ended": state["ended"]
                }
        return None

class RetrievalNode:
    """
    This node extracts the information from user input
    """

    def __init__(self):
        pass

    def invoke(self, state: ChatbotState) -> dict[str, list | dict | bool] | None:
        """
        Extracts the information from user input
        """
        last_message = state["messages"][-1] if len(
            state["messages"]) > 0 else "No message"

        if not state['active_order'] or not isinstance(last_message, FunctionMessage):
            logger.info("Not active order or not function message.")
            return {
                "messages": state["messages"],
                "slots": state["slots"],
                "ended": state["ended"]
            }

        _input = state['input'].lower().strip()

        if last_message.content == OrderSlots.PIZZA_NAME.value:
            available_pizzas = api_client.list_pizzas()
            pizza_names = [pizza['name'] for pizza in available_pizzas]

            # Use fuzzy matching to find the best match
            best_match = process.extractOne(
                _input,
                pizza_names,
                scorer=fuzz.partial_ratio,
                score_cutoff=70
            )

            if best_match:
                matched_pizza_name = best_match[0]
                confidence_score = best_match[1]

                matched_pizza = next((pizza for pizza in available_pizzas if pizza['name'] == matched_pizza_name), None)

                if confidence_score >= 70:
                    # print("DEBUG: matched slots = ", matched_pizza)
                    print(f"-- Chatbot: For your request of '{_input}', I found: '{matched_pizza["name"]}' and added it to your order.")
                    state['slots'][OrderSlots.PIZZA_NAME.value] = matched_pizza['name']

            else:
                # No match found at all
                available_names = ", ".join(pizza_names)
                state['messages'].append(AIMessage(content=f"I couldn't find '{_input}' on our menu. Available pizzas are: {available_names}"))
                state["messages"].append(FunctionMessage(content=OrderSlots.PIZZA_NAME, name=OrderSlots.PIZZA_NAME.value))

            return {
                "messages": state["messages"],
                "slots": state["slots"],
                "ended": state["ended"]
            }
        elif last_message.content == OrderSlots.PIZZA_COUNT.value:
            try:
                pizza_count = int(_input)
                if pizza_count < 1:
                    state['messages'].append(AIMessage(content="Please enter a positive number of pizzas (at least 1)."))
                    state["messages"].append(FunctionMessage(content=OrderSlots.PIZZA_COUNT, name=OrderSlots.PIZZA_COUNT.value))
                    return {
                        "messages": state["messages"],
                        "slots": state["slots"],
                        "ended": state["ended"]
                    }
                
                state['slots'][OrderSlots.PIZZA_COUNT.value] = pizza_count
                return {
                    "messages": state["messages"],
                    "slots": state["slots"],
                    "ended": state["ended"]
                }
            except ValueError:
                state['messages'].append(AIMessage(content="Please enter a valid number for the pizza count."))
                state["messages"].append(FunctionMessage(content=OrderSlots.PIZZA_COUNT, name=OrderSlots.PIZZA_COUNT.value))
                return {
                    "messages": state["messages"],
                    "slots": state["slots"],
                    "ended": state["ended"]
                }
        elif last_message.content == OrderSlots.CUSTOMER_ADDRESS.value:
            logger.info("Customer address collected: %s" % _input)
            state['slots'][OrderSlots.CUSTOMER_ADDRESS.value] = _input
            return {
                "messages": state["messages"],
                "slots": state["slots"],
                "ended": state["ended"]
            }
        return None

class PizzaAPIClient:
    # Communicates with https://demos.swe.htwk-leipzig.de/pizza-api/docs
    def __init__(self, base_url: str = "https://demos.swe.htwk-leipzig.de/pizza-api"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'accept': 'application/json',
            'Content-Type': 'application/json'
        })
        # Disable SSL verification for outdated certificates
        self.session.verify = False

    #   GET /pizza - List available pizzas
    def list_pizzas(self) -> List[Dict[str, Any]]:
        try:
            response = self.session.get(f"{self.base_url}/pizza")
            response.raise_for_status()
            print(response.json())
            return response.json()
        except requests.RequestException as e:
            print(f"Error fetching pizzas: {e}")
            return []

    #     POST /address/validate - Validate delivery address
    def validate_address(self, address: str) -> Dict[str, Any]:
        try:
            response = self.session.post(
                f"{self.base_url}/address/validate",
                json = {"address": address}
            )
            response.raise_for_status()
            print(response.json())
            return response.json()
        except requests.RequestException as e:
            print(f"Error validating address: {e}")
            return {"valid": False, "error": str(e)}

    #     POST /order - Create a new pizza order
    def create_order(self, pizza_name: str, address: str) -> Dict[str, Any]:
        try:
            response = self.session.post(
                f"{self.base_url}/order",
                json = {
                    "pizza_name": pizza_name,
                    "delivery_address": address
                }
            )
            response.raise_for_status()
            print(response.json())
            return response.json()
        except requests.RequestException as e:
            print(f"Error creating order: {e}")
            return {"success": False, "error": str(e)}

    #     GET /order/{order_id} - Get order status
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        try:
            response = self.session.get(f"{self.base_url}/order/{order_id}")
            response.raise_for_status()
            print(response.json())
            return response.json()
        except requests.RequestException as e:
            print(f"Error fetching order status: {e}")
            return {"status": "unknown", "error": str(e)}

def display_graph():
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from io import BytesIO

    png_data = graph.get_graph().draw_mermaid_png()
    img = mpimg.imread(BytesIO(png_data))

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)
    ax.axis('off')
    plt.tight_layout()
    plt.show()

# def test_routing_function(state: ChatbotState) -> str:
#     return Nodes.CONFIRMATION

if __name__ == "__main__":
    # Initialize nodes
    order_node = OrderNode()
    checker_node = CheckerNode()
    retrieval_node = RetrievalNode()
    api_client = PizzaAPIClient()

    workflow = StateGraph(ChatbotState)
    workflow.add_node(Nodes.CHECKER.value, checker_node.invoke)
    workflow.add_node(Nodes.RETRIEVAL.value, retrieval_node.invoke)
    workflow.add_node(Nodes.ORDER_FORM.value, order_node.invoke)

    workflow.add_conditional_edges(
        Nodes.CHECKER.value,
        checker_node.route,
        {
            Nodes.RETRIEVAL.value: Nodes.RETRIEVAL.value,
            END: END,
        }
    )
    workflow.add_edge(Nodes.RETRIEVAL.value, Nodes.ORDER_FORM.value)
    workflow.add_edge(Nodes.ORDER_FORM.value, END)

    workflow.set_entry_point(Nodes.CHECKER.value)
    graph = workflow.compile()

    # display_graph()

    # START DIALOGUE: first message
    print("-- Chatbot: ", "Hi! I am a pizza bot. I can help you order a pizza. What would you like to order?")
    user_input = input("-> Your response: ")
    outputs = graph.invoke({"input": user_input, "slots": {}, "messages": [
    ], "active_order": False, "ended": False})

    while True:
        print("-- Chatbot: ", [m.content for m in outputs["messages"]
              if isinstance(m, AIMessage)][-1])  # print chatbot response
        user_input = input("-> Your response: ")

        outputs = graph.invoke({"input": user_input, "slots": outputs["slots"], "messages": outputs[
                               "messages"], "active_order": outputs["active_order"], "ended": outputs["ended"]})

        # check if the conversation has ended
        if outputs["ended"]:
            print("-- Chatbot: ", [m.content for m in outputs["messages"]
                  if isinstance(m, AIMessage)][-1])  # print chatbot response
            break