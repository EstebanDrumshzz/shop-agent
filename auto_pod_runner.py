import json
from typing import Any, Dict, Optional

from dotenv import load_dotenv

from trend_agent import identify_trend
from pod_workflow import build_workflow, PodState

load_dotenv()


def run_auto_pod(seed: Optional[str] = None, max_iterations: int = 2) -> Dict[str, Any]:
    """Run end-to-end: identify trend -> generate design -> create + list product.

    Returns the combined result with provenance.
    """
    trend_pkg = identify_trend(seed=seed, max_iterations=max_iterations)

    app = build_workflow()
    initial: PodState = {
        "trend": trend_pkg.get("trend", ""),
        "design_prompt": trend_pkg.get("design_prompt", ""),
        "keywords": trend_pkg.get("keywords", []),
        "seo_title": trend_pkg.get("seo_title", ""),
        "seo_description": trend_pkg.get("seo_description", ""),
        "logs": [],
    }
    final_state: PodState = app.invoke(initial)

    return {
        "trend": trend_pkg.get("trend"),
        "keywords": trend_pkg.get("keywords"),
        "design_prompt": trend_pkg.get("design_prompt"),
        "seo_title": trend_pkg.get("seo_title"),
        "seo_description": trend_pkg.get("seo_description"),
        "sources": trend_pkg.get("sources"),
        "pod": final_state,
    }


if __name__ == "__main__":
    result = run_auto_pod()
    print(json.dumps(result, indent=2))
