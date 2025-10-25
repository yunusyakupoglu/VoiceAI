# Compatibility shim: allow `python -m speaker_id.cli`
from speakerid.cli import app, main

if __name__ == "__main__":
    main()
