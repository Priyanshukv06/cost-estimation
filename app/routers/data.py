"""
Data router — serves random test data samples and field dropdown options.
"""

import os
import json
import random
import logging
from fastapi import APIRouter
from app.models import RandomPatientResponse, FieldOptionsResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/data", tags=["Sample Data"])

# In-memory cache for sample data and field options
_sample_data: list[dict] = []
_field_options: dict[str, list[str]] = {}


def load_sample_data():
    """Loads the pre-generated sample data and field options into memory."""
    global _sample_data, _field_options

    # Determine project root: data.py is in app/routers/, so go up 3 levels
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    sample_file = os.path.join(project_root, "data_sample", "test_sample.json")
    options_file = os.path.join(project_root, "data_sample", "field_options.json")

    if os.path.exists(sample_file):
        with open(sample_file, "r") as f:
            _sample_data = json.load(f)
        logger.info(f"✅ Loaded {len(_sample_data)} sample records from {sample_file}")
    else:
        logger.warning(f"⚠️ Sample data file not found: {sample_file}")
        logger.warning("  Run 'python scripts/prepare_sample_data.py' to generate it.")

    if os.path.exists(options_file):
        with open(options_file, "r") as f:
            _field_options = json.load(f)
        logger.info(f"✅ Loaded field options for {len(_field_options)} fields")
    else:
        logger.warning(f"⚠️ Field options file not found: {options_file}")


@router.get("/random", response_model=RandomPatientResponse)
async def get_random_patient():
    """
    Returns a random patient record from the test dataset.
    Includes actual Total Cost and Total Charge values for comparison.
    Use the 'Randomize' button in the frontend to get a new sample.
    """
    if not _sample_data:
        return RandomPatientResponse(patient={"error": "No sample data loaded. Run prepare_sample_data.py first."})

    patient = random.choice(_sample_data)
    return RandomPatientResponse(patient=patient)


@router.get("/random/batch")
async def get_random_patients(count: int = 5):
    """Returns multiple random patient records."""
    if not _sample_data:
        return {"patients": [], "error": "No sample data loaded."}

    count = min(count, 50)  # Cap at 50
    patients = random.sample(_sample_data, min(count, len(_sample_data)))
    return {"patients": patients}


@router.get("/field-options", response_model=FieldOptionsResponse)
async def get_field_options():
    """
    Returns all unique values for each input field.
    Used to populate dropdown selectors in the frontend UI.
    """
    if not _field_options:
        return FieldOptionsResponse(field_options={"error": ["No field options loaded. Run prepare_sample_data.py first."]})

    return FieldOptionsResponse(field_options=_field_options)
