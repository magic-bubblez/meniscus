"""Quick diagnostic: does the real Gemini call actually work?

Run it with:

    cd ~/Desktop/meniscus && .venv/bin/python check_model.py
"""

from dotenv import load_dotenv

load_dotenv()

from models import ExtractionResult
from providers import get_model

print("Building model and making one test call to Gemini...\n")

try:
    result = get_model().generate_structured(
        "Extract key concepts from: learning about JWT auth tokens",
        ExtractionResult,
    )
    print("SUCCESS — the model works.")
    print("Entities it extracted:", [e.name for e in result.entities])
    print("\nNow run:  .venv/bin/men process")
except Exception:
    import traceback

    traceback.print_exc()
    print("\n^^^ That traceback (last line especially) is the real error.")
    print("Common causes:")
    print("  429 / RESOURCE_EXHAUSTED  ->  rate/quota limit; wait a minute and retry")
    print("  API key not valid / 400   ->  fix GEMINI_API_KEY in your .env")
    print("  connection / SSL error    ->  network or proxy")
