import os
import json
import logging
from fastapi import APIRouter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/stats", tags=["Test Statistics"])

# In-memory cache for test stats
_test_stats = {}

def load_test_stats():
    """Loads the pre-generated test statistics into memory."""
    global _test_stats

    # Determine project root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    stats_file = os.path.join(project_root, "data_sample", "test_stats.json")

    if os.path.exists(stats_file):
        with open(stats_file, "r") as f:
            _test_stats = json.load(f)
        logger.info(f"✅ Loaded test statistics from {stats_file}")
    else:
        logger.warning(f"⚠️ Test statistics file not found: {stats_file}")
        logger.warning("  Run 'python scripts/generate_test_stats.py' to generate it.")

@router.get("/test_stats")
async def get_test_stats():
    """
    Returns the precomputed precision-recall curves, threshold statistics,
    buffer strategy metrics, and feature importance for cost and charge models.
    """
    if not _test_stats:
        return {"error": "Test statistics not loaded. Run generate_test_stats.py first."}
    
    return _test_stats
