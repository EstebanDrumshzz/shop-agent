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
PRINTIFY_SHOP_ID = os.getenv("PRINTIFY_SHOP_ID")
if not PRINTIFY_API_KEY or not SHOPIFY_STORE_URL or not PRINTIFY_SHOP_ID:
    raise ValueError("Missing required environment variables: PRINTIFY_API_KEY, SHOPIFY_STORE_URL, PRINTIFY_SHOP_ID")

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
    # Dynamically identify a relevant trend and generate design + SEO inputs
    # If AUTO_TREND is false and a trend was provided, keep it.
    if (not getenv_bool("AUTO_TREND", True)) and state.get("trend"):
        state = log(state, f"Identified trend (provided): {state['trend']}")
        return state

    niche = os.getenv("STORE_NICHE", "general print-on-demand")
    today = datetime.datetime.now().date().isoformat()

    system = (
        "You are a product trend research assistant for a print-on-demand store. "
        "Choose a timely, non-infringing, non-trademarked topic that is safe to sell. "
        "Avoid copyrighted IP, brands, celebrities, or sensitive/controversial topics. "
        "Output strict JSON with keys: trend, design_prompt, keywords, seo_title, seo_description."
    )

    user = (
        f"Store niche: {niche}. Today's date: {today}.\n"
        "Return a single best idea as JSON.\n"
        "Constraints:\n"
        "- The design_prompt must fully describe a 1024x1024 flat/vector-style composition without backgrounds.\n"
        "- The design should work on light and dark products.\n"
        "- keywords: 5-8 SEO phrases.\n"
        "- seo_title: <= 70 chars.\n"
        "- seo_description: 140-160 chars.\n"
    )

    try:
        resp = _openai_client.chat.completions.create(
            model=os.getenv("OPENAI_TREND_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            temperature=float(os.getenv("TREND_TEMPERATURE", "0.7")),
        )
        content = resp.choices[0].message.content or "{}"
        data = json.loads(content)
        # Fill state with agent outputs (with safe fallbacks)
        trend = data.get("trend") or state.get("trend") or "timely minimalist graphic"
        design_prompt = data.get("design_prompt") or state.get("design_prompt") or (
            "Create a 1024x1024 high-contrast flat/vector graphic, centered composition, no background."
        )
        keywords = data.get("keywords") or state.get("keywords") or [
            "trending graphic tee", "minimalist vector art", "unique gift idea", "fun style shirt", "cool design apparel"
        ]
        seo_title = data.get("seo_title") or state.get("seo_title") or f"{trend} - Graphic Apparel"
        seo_description = data.get("seo_description") or state.get("seo_description") or (
            "Stand out with a unique, high-contrast vector design. Perfect for gifts and everyday style."
        )

        state["trend"] = trend
        state["design_prompt"] = design_prompt
        state["keywords"] = keywords
        state["seo_title"] = seo_title
        state["seo_description"] = seo_description
        state = log(state, f"Identified trend (auto): {trend}")
        return state
    except Exception as e:
        # Fall back to provided or default trend if agent fails
        state = log(state, f"Trend agent failed: {e}. Falling back to existing/default inputs.")
        if not state.get("trend"):
            state["trend"] = "funny gym cats"
        if not state.get("design_prompt"):
            state["design_prompt"] = (
                "Create a 1024x1024 vector/flat graphic with a humorous subject, high-contrast, centered, "
                "bold flat colors, and no background. Include a short punny slogan integrated nicely."
            )
        if not state.get("keywords"):
            state["keywords"] = [
                "funny meme shirts", "humor graphic tee", "punny design apparel",
                "unique gift shirt", "vector art t-shirt"
            ]
        if not state.get("seo_title"):
            state["seo_title"] = f"{state['trend']} - Graphic Apparel"
        if not state.get("seo_description"):
            state["seo_description"] = (
                "A unique, pun-inspired vector design that pops on any color. Great for gifts and daily wear."
            )
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
    """Fetches all valid variants for a given blueprint and print provider.
    Be tolerant to schema differences (some responses may not include `is_enabled`).
    """
    try:
        url = f"https://api.printify.com/v1/catalog/blueprints/{blueprint_id}/print_providers/{print_provider_id}/variants.json"
        headers = {"Authorization": f"Bearer {PRINTIFY_API_KEY}"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        variants = data.get("variants", data if isinstance(data, list) else [])
        cleaned: List[Dict[str, Any]] = []
        for v in variants:
            if not isinstance(v, dict):
                continue
            vid = v.get("id")
            if vid is None:
                continue
            # If key exists, enforce it; otherwise include by default
            if "is_enabled" in v and not v.get("is_enabled", False):
                continue
            cleaned.append(v)
        return cleaned
    except Exception as e:
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
        # Step 1: Download the image first and then upload it as a file
        state = log(state, f"Downloading image from DALL-E URL: {image_url}")
        image_response = requests.get(image_url)
        # This is the new, crucial line to ensure the download was successful
        image_response.raise_for_status() 
        
        # New check: ensure the content type is actually an image
        content_type = image_response.headers.get('Content-Type', '')
        if 'image' not in content_type:
            raise ValueError(f"Downloaded content is not an image. Content-Type: {content_type}")

        image_data = image_response.content
        
        # Step 2: Upload the image to Printify
        upload_url = "https://api.printify.com/v1/uploads/images.json"
        # Determine file extension from content type
        ext = "png"
        if "jpeg" in content_type or "jpg" in content_type:
            ext = "jpg"
        elif "webp" in content_type:
            ext = "webp"
        file_name = f"design.{ext}"
        # Printify expects JSON with file_name and base64-encoded contents
        b64_contents = base64.b64encode(image_data).decode("utf-8")
        payload = {
            "file_name": file_name,
            "contents": b64_contents,
        }
        headers = {
            "Authorization": f"Bearer {PRINTIFY_API_KEY}",
            "X-Printify-Shop-Id": PRINTIFY_SHOP_ID,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        state = log(state, f"Uploading image to Printify as JSON (file_name: {file_name}, size: {len(image_data)} bytes)...")
        upload_response = requests.post(upload_url, headers=headers, json=payload, timeout=60)
        if upload_response.status_code >= 400:
            try:
                err_text = upload_response.text
            except Exception:
                err_text = "<no body>"
            raise requests.HTTPError(f"{upload_response.status_code} {upload_response.reason}: {err_text}")
        image_id = upload_response.json()["id"]
        state = log(state, f"Image uploaded to Printify with ID: {image_id}")
    except Exception as e:
        state = log(state, f"Error uploading image to Printify: {e}. Skipping this product.")
        return state

    try:
        # Step 3: Get available variants and construct payload
        state = log(state, "Searching for a valid variant for the blueprint...")
        valid_variants = get_print_provider_variants(blueprint_id, print_provider_id)
        if not valid_variants:
            state = log(state, "No valid variants found. Skipping this product.")
            return state

        # Construct the variants payload for the new product
        default_price_cents = int(os.getenv("PRINTIFY_DEFAULT_PRICE_CENTS", "1999"))
        variants_payload: List[Dict[str, Any]] = []
        for v in valid_variants:
            price_cents = default_price_cents
            # If the catalog provides cost, try to apply a margin
            cost_val = v.get("cost") if isinstance(v, dict) else None
            try:
                if cost_val is not None:
                    cost_int = int(round(float(cost_val)))
                    price_cents = max(default_price_cents, int(cost_int * 1.6))
            except Exception:
                pass
            variants_payload.append({
                "id": v["id"],
                "price": price_cents,
                "is_enabled": True
            })
        # Respect Printify's max enabled variants (100)
        # Sort deterministically by variant id, enable first 100, disable the rest
        variants_payload.sort(key=lambda x: x["id"])  # type: ignore[index]
        if len(variants_payload) > 100:
            for idx, vp in enumerate(variants_payload):
                vp["is_enabled"] = idx < 100
            state = log(state, f"Limiting enabled variants to 100 (from {len(variants_payload)}).")
        enabled_ids = [vp["id"] for vp in variants_payload if vp.get("is_enabled")]
        state = log(state, f"Prepared {len(variants_payload)} variant entries; enabling {len(enabled_ids)}; example price: {variants_payload[0]['price']} cents")

        # Step 4: Create the product
        create_url = f"https://api.printify.com/v1/shops/{PRINTIFY_SHOP_ID}/products.json"
        
        # Construct the full product payload as per the Printify documentation
        product_payload = {
            "title": state.get("seo_title", f"Custom {title}"),
            "description": state.get("seo_description", "A unique and humorous design created just for you!"),
            "blueprint_id": blueprint_id,
            "print_provider_id": print_provider_id,
            "variants": variants_payload,
            "print_areas": [
                {
                    "variant_ids": [vp["id"] for vp in variants_payload if vp["is_enabled"]],
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
        headers = {
            "Authorization": f"Bearer {PRINTIFY_API_KEY}",
            "X-Printify-Shop-Id": PRINTIFY_SHOP_ID,
            "Accept": "application/json",
        }
        create_response = requests.post(create_url, headers=headers, json=product_payload, timeout=60)
        if create_response.status_code >= 400:
            try:
                err_text = create_response.text
            except Exception:
                err_text = "<no body>"
            raise requests.HTTPError(f"{create_response.status_code} {create_response.reason}: {err_text}")
        
        printify_product_id = create_response.json()["id"]
        store_url = f"https://{SHOPIFY_STORE_URL}/admin/apps/printify/products/{printify_product_id}"
        state = log(state, f"Successfully created Printify product with ID: {printify_product_id}")

        # Publish the product to Shopify via Printify
        publish_url = f"https://api.printify.com/v1/shops/{PRINTIFY_SHOP_ID}/products/{printify_product_id}/publish.json"
        publish_payload = {
            "title": True,
            "description": True,
            "images": True,
            "variants": True,
            "tags": True,
            "key_features": True,
            "shipping_template": True,
        }
        state = log(state, "Publishing product to Shopify...")
        publish_response = requests.post(publish_url, headers=headers, json=publish_payload, timeout=60)
        published = False
        if publish_response.status_code >= 400:
            try:
                publish_err = publish_response.text
            except Exception:
                publish_err = "<no body>"
            state = log(state, f"Error publishing product: {publish_response.status_code} {publish_response.reason}: {publish_err}")
        else:
            published = True
            state = log(state, "Product publish requested successfully.")

        state["created_products"] = state.get("created_products", []) + [{
            "blueprint_id": blueprint_id,
            "printify_product_id": printify_product_id,
            "store_url": store_url,
            "published": published,
        }]
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
        # Let the trend agent decide by default (AUTO_TREND=true). Provide empty context and logs.
        "logs": [],
        "created_products": [],
        "blueprints_to_create": [],
        # Optionally you can prefill 'trend' to force a theme when AUTO_TREND=false
    }
    return app.invoke(initial)


if __name__ == "__main__":
    pprint(run_example())
