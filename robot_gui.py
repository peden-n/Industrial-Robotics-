import tkinter as tk
from typing import List, Any


class SimpleRobotGUI:
    def __init__(self):
        self.stopped = False

        # UI
        self.root = tk.Tk()
        self.root.title('Robot control panel')
        self.root.geometry("280x120")

        self.status_var = tk.StringVar(value="Status: RUNNING")
        tk.Label(self.root, textvariable=self.status_var, font=("Arial", 12)).pack(pady=(12, 6))

        self.btn = tk.Button(
            self.root,
            text="EMERGENCY STOP",
            font=("Arial", 14, "bold"),
            bg="red", fg="white",
            activebackground="darkred",
            command=self.toggle_emergency,
            height=2
        )
        self.btn.pack(padx=10, pady=8, fill="x")

        # Optional hotkeys
        self.root.bind("<space>", lambda _e: self.toggle_emergency())

    def run(self):
        self.root.mainloop()

    def toggle_emergency(self):
        if not self.stopped:
            self.stopped = True
            self.btn.config(text="RESUME", bg="green", activebackground="darkgreen")
            self.status_var.set("Status: STOPPED")
        else:
            self.stopped = False
            self.btn.config(text="EMERGENCY STOP", bg="red", activebackground="darkred")
            self.status_var.set("Status: RUNNING")
