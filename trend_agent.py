import os
import time
import json
import base64
import datetime
import requests
from typing import TypedDict, List, Optional, Dict, Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END, MessageGraph
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import StructuredTool
from langsmith import traceable as _traceable # optional; no-op if not configured
from pprint import pprint

# Load environment variables from .env if present
load_dotenv()

# LangSmith tracing decorator (optional). If unavailable, becomes a no-op.
try:
    from langsmith import traceable as _traceable  # type: ignore
except Exception:
    def _traceable(*dargs, **dkwargs):
        # Fallback no-op decorator
        if dargs and callable(dargs[0]):
            return dargs[0]
        def wrapper(f):
            return f
        return wrapper
traceable = _traceable


# --------- Configuration helpers ---------

def getenv_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.lower() in {"1", "true", "yes", "on"}

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

# --- Define API clients ---
PRINTIFY_API_KEY = os.getenv("PRINTIFY_API_KEY")
SHOPIFY_STORE_URL = os.getenv("SHOPIFY_STORE_URL")
if not PRINTIFY_API_KEY or not SHOPIFY_STORE_URL:
    raise ValueError("Missing required environment variables: PRINTIFY_API_KEY, SHOPIFY_STORE_URL")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Missing required environment variable: OPENAI_API_KEY")

_openai_client = OpenAI(api_key=OPENAI_API_KEY)


# --------- State definition ---------

class PodState(TypedDict, total=False):
    # Inputs / context
    trend: str
    design_prompt: str
    keywords: List[str]
    seo_title: str
    seo_description: str
    # Outputs from agents
    image_path: str
    image_url: str
    created_products: List[Dict[str, Any]]
    blueprints_to_create: List[Dict[str, Any]]
    # Diagnostics
    logs: List[str]


# --------- Logging utility ---------

def log(state: PodState, message: str) -> PodState:
    now = datetime.datetime.now().isoformat()
    log_message = f"[{now}] {message}"
    print(log_message)
    state["logs"] = state.get("logs", []) + [log_message]
    return state


# --------- Agentic Functions ---------

@traceable(name="identify_trend_node")
def identify_trend_node(state: PodState):
    # This function is a placeholder and should be replaced by a real trend-identifying agent.
    # For now, it just passes through a hardcoded trend.
    state = log(state, f"Identified trend: {state['trend']}")
    return state

@traceable(name="generate_design")
def generate_design(state: PodState):
    state = log(state, f"Generating design for trend: {state['trend']}")
    try:
        response = _openai_client.images.generate(
            model="dall-e-3",
            prompt=state["design_prompt"],
            size="1024x1024",
            quality="standard",
            n=1,
            response_format="url",
            style="vivid"
        )
        image_url = response.data[0].url
        state = log(state, f"Generated image URL: {image_url}")
        state["image_url"] = image_url
        return state
    except Exception as e:
        state = log(state, f"Error generating design: {e}")
        return state


@traceable(name="select_products")
def select_products(state: PodState) -> PodState:
    state = log(state, "Selecting 5 diverse products from Printify catalog...")
    # This is a hardcoded list for demonstration. In a real-world scenario,
    # you would use a tool to query the Printify API for blueprints.
    selected_blueprints = [
        {"blueprint_id": 6, "print_provider_id": 43, "title": "Unisex Heavy Cotton Tee"},
        {"blueprint_id": 36, "print_provider_id": 72, "title": "Unisex Pullover Hoodie"},
        {"blueprint_id": 68, "print_provider_id": 1, "title": "Mug 11oz"},
        {"blueprint_id": 232, "print_provider_id": 10, "title": "Tote Bag"},
        {"blueprint_id": 384, "print_provider_id": 1, "title": "Square Stickers"},
    ]
    state = log(state, "Using a predefined, validated list of blueprints.")
    state["blueprints_to_create"] = selected_blueprints
    state["created_products"] = []
    return state


def get_print_provider_variants(blueprint_id: int, print_provider_id: int) -> List[Dict[str, Any]]:
    """Fetches all valid variants for a given blueprint and print provider."""
    try:
        url = f"https://api.printify.com/v1/catalog/blueprints/{blueprint_id}/print_providers/{print_provider_id}/variants.json"
        headers = {"Authorization": f"Bearer {PRINTIFY_API_KEY}"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        variants = response.json()["variants"]
        return [v for v in variants if v["is_enabled"]]
    except requests.exceptions.RequestException as e:
        print(f"Error fetching variants for blueprint {blueprint_id}: {e}")
        return []

@traceable(name="create_single_product")
def create_single_product(state: PodState) -> PodState:
    if not state.get("blueprints_to_create"):
        return state

    product_info = state["blueprints_to_create"].pop(0)
    blueprint_id = product_info["blueprint_id"]
    print_provider_id = product_info["print_provider_id"]
    title = product_info["title"]
    image_url = state["image_url"]

    state = log(state, f"Creating product for blueprint: {title} (ID: {blueprint_id})")

    try:
        # Step 1: Upload the image to Printify via URL, as per the API spec
        # This is the corrected payload.
        upload_url = "https://api.printify.com/v1/uploads/images.json"
        headers = {"Authorization": f"Bearer {PRINTIFY_API_KEY}", "Content-Type": "application/json"}
        upload_payload = {
            "file_name": "design.png",
            "url": image_url
        }
        
        state = log(state, "Uploading image to Printify...")
        upload_response = requests.post(upload_url, headers=headers, json=upload_payload)
        upload_response.raise_for_status()
        image_id = upload_response.json()["id"]
        state = log(state, f"Image uploaded to Printify with ID: {image_id}")
    except requests.exceptions.RequestException as e:
        state = log(state, f"Error uploading image to Printify: {e}. Skipping this product.")
        return state

    try:
        # Step 2: Get available variants and construct payload
        state = log(state, "Searching for a valid variant for the blueprint...")
        valid_variants = get_print_provider_variants(blueprint_id, print_provider_id)
        if not valid_variants:
            state = log(state, "No valid variants found. Skipping this product.")
            return state

        # Construct the variants payload for the new product
        variants_payload = []
        for v in valid_variants:
            variants_payload.append({
                "id": v["id"],
                "price": v["price"],
                "is_enabled": True
            })

        # Step 3: Create the product
        create_url = f"https://api.printify.com/v1/shops/24250990/products.json"
        
        # Construct the full product payload as per the Printify documentation
        product_payload = {
            "title": state.get("seo_title", f"Custom {title}"),
            "description": state.get("seo_description", "A unique and humorous design created just for you!"),
            "blueprint_id": blueprint_id,
            "print_provider_id": print_provider_id,
            "variants": variants_payload,
            "print_areas": [
                {
                    "variant_ids": [v["id"] for v in valid_variants],
                    "placeholders": [
                        {
                            "position": "front",
                            "images": [
                                {
                                    "id": image_id,
                                    "x": 0.5,
                                    "y": 0.5,
                                    "scale": 1,
                                    "angle": 0
                                }
                            ]
                        }
                    ]
                }
            ]
        }
        
        state = log(state, "Creating Printify product...")
        create_response = requests.post(create_url, headers=headers, json=product_payload)
        create_response.raise_for_status()
        
        printify_product_id = create_response.json()["id"]
        store_url = f"https://{SHOPIFY_STORE_URL}/admin/apps/printify/products/{printify_product_id}"

        state["created_products"] = state.get("created_products", []) + [{
            "blueprint_id": blueprint_id,
            "printify_product_id": printify_product_id,
            "store_url": store_url
        }]
        state = log(state, f"Successfully created Printify product with ID: {printify_product_id}")
    except requests.exceptions.RequestException as e:
        state = log(state, f"Error creating product on Printify: {e}. Skipping this product.")
    
    return state


def should_continue(state: PodState) -> str:
    if state.get("blueprints_to_create"):
        return "continue"
    else:
        return "end"


# --------- Build Graph ---------

def build_workflow() -> Any:
    """Builds the LangGraph workflow for the autonomous agent."""
    graph = StateGraph(PodState)
    graph.add_node("identify_trend", identify_trend_node)
    graph.add_node("generate_design", generate_design)
    graph.add_node("select_products", select_products)
    graph.add_node("create_single_product", create_single_product)

    # Define the graph flow
    graph.add_edge(START, "identify_trend")
    graph.add_edge("identify_trend", "generate_design")
    graph.add_edge("generate_design", "select_products")
    
    # Add a conditional edge for the product creation loop
    graph.add_conditional_edges(
        "select_products",
        should_continue,
        {
            "continue": "create_single_product",
            "end": END,
        }
    )
    graph.add_conditional_edges(
        "create_single_product",
        should_continue,
        {
            "continue": "create_single_product",
            "end": END,
        }
    )

    return graph.compile()


@traceable(name="run_example")
def run_example() -> PodState:
    app = build_workflow()
    initial: PodState = {
        "trend": "funny gym cats",
        "logs": [],
        "created_products": [],
        "blueprints_to_create": [],
        "design_prompt": "Create a 1024x1024 vector/flat graphic depicting a humorous scene of a cat lifting weights, preferably a chonky cat struggling with a tiny dumbbell, with a puzzled or determined expression. The composition should be high-contrast and centered, featuring bold, flat colors, and include a punny slogan like 'Working on my Meowscles' integrated into the design attractively. Ensure the graphic is versatile enough to be used on both light and dark backgrounds with no background included in the design itself.",
        "keywords": ["gym cat meme shirts", "hilarious cat workout apparel", "feline fitness humor tees", "comic cat gym gear", "fit felines funny attire"],
        "seo_title": "Working on my Meowscles - Funny Gym Cat Apparel",
        "seo_description": "Unleash your inner fitness feline with our 'Working on my Meowscles' tees! Perfect for gym enthusiasts who love a good chuckle, these shirts blend humorous cat parodies with fitness motivation. Whether you're lifting weights or chasing after laser pointers, do it in style with our unique, pun-inspired gym wear. Designed for laughs but made for workouts, it's the purr-fect way to stand out at the gym or anywhere your fitness journey takes you.",
    }
    return app.invoke(initial)


if __name__ == "__main__":
    pprint(run_example())
