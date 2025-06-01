import threading
import wx

def run_in_thread(fn):
    """Decorator to run handler in a background thread and manage button states."""
    def wrapper(self, event):
        def task():
            try:
                fn(self, event)
            except Exception as e:
                wx.PostEvent(self, self.UpdateOutputEvent(message=f"Error in {fn.__name__}: {e}\n"))
                import traceback
                wx.PostEvent(self, self.UpdateOutputEvent(message=f"Traceback: {traceback.format_exc()}\n"))
            finally:
                wx.CallAfter(lambda: self.enable_buttons(True))
        if not getattr(self, 'buttons_enabled', True):
            return
        self.enable_buttons(False)
        threading.Thread(target=task, daemon=True).start()
    return wrapper 