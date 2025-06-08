import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dotenv import load_dotenv
load_dotenv()
from src.utils.logging import setup_logging
from gui.main import App
def main():
    setup_logging()
    app = App()
    app.MainLoop()

if __name__ == "__main__":
    main() 