# Compatibility shim: allow `uvicorn speaker_id.api:app`
from speakerid.api import app  # noqa: F401
