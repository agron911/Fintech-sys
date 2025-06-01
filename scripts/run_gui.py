from src.utils.logging import setup_logging
from gui.main import App

def main():
    setup_logging()
    app = App()
    app.MainLoop()

if __name__ == "__main__":
    main() 