# controls.py (or put at the top of your script)
import threading, time
try:
    import tkinter as tk
except Exception:
    tk = None  # headless fallback

class RunController:
    def __init__(self, make_gui: bool = True):
        self.paused = False
        self.stopped = False
        self._lock = threading.Lock()
        self._gui_thread = None
        if make_gui and tk is not None:
            self._start_gui()

    def toggle_pause(self):
        with self._lock:
            self.paused = not self.paused

    def stop(self):
        with self._lock:
            self.stopped = True

    def is_paused(self) -> bool:
        with self._lock:
            return self.paused

    def is_stopped(self) -> bool:
        with self._lock:
            return self.stopped

    # --- minimal Tk GUI in a background thread ---
    def _start_gui(self):
        def run():
            root = tk.Tk()
            root.title("Robot Control")
            state = tk.StringVar(value="Pause")
            def on_pause():
                self.toggle_pause()
                state.set("Resume" if self.is_paused() else "Pause")
            def on_stop():
                self.stop()
                root.destroy()
            tk.Button(root, textvariable=state, width=12, command=on_pause).pack(padx=10, pady=(10,6))
            tk.Button(root, text="STOP", fg="white", bg="red", width=12, command=on_stop).pack(padx=10, pady=(0,10))
            root.mainloop()
        self._gui_thread = threading.Thread(target=run, daemon=True)
        self._gui_thread.start()
