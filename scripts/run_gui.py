from investment_system.src.utils.logging import setup_logging
from investment_system.gui.main import App

def main():
    setup_logging()
    app = App()
    app.MainLoop()

if __name__ == "__main__":
    main() 