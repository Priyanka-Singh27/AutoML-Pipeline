import os
import sys

def run_mode():
    mode = os.environ.get("AUTOML_MODE", "cli").lower()
    
    if mode == "api":
        import uvicorn
        uvicorn.run("api.app:app", host="0.0.0.0", port=8000)
    else:
        # Pass everything to main.py's click CLI
        from main import main
        # sys.argv is automatically forwarded to click when main() is called without args
        sys.argv[0] = 'main.py'
        main()

if __name__ == "__main__":
    run_mode()
