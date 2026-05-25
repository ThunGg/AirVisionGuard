import tkinter as tk
from PIL import Image, ImageTk, ImageGrab, ImageFilter
import threading

class ScreenLocker:
    def __init__(self):
        self.root = tk.Tk()
        self.root.withdraw() # Hide initially
        self.root.attributes('-fullscreen', True)
        self.root.attributes('-topmost', True)
        self.root.configure(background='black')
        self.root.overrideredirect(True)
        
        # Block common exits (doesn't block Ctrl+Alt+Del)
        self.root.protocol("WM_DELETE_WINDOW", self.disable_event)
        self.root.bind("<Escape>", self.disable_event)
        self.root.bind("<Alt-F4>", self.disable_event)
        self.root.bind("<Alt-Tab>", self.disable_event)
        self.root.bind("<FocusOut>", self.on_focus_out)

        self.canvas = tk.Canvas(self.root, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.bg_image_tk = None
        self.is_locked = False

    def disable_event(self, event=None):
        return "break"

    def on_focus_out(self, event=None):
        if self.is_locked:
            self.root.lift()
            self.root.focus_force()

    def lock(self):
        if not self.is_locked:
            # Capture screen, blur it
            screen = ImageGrab.grab()
            blurred_screen = screen.filter(ImageFilter.GaussianBlur(radius=80))
            self.bg_image_tk = ImageTk.PhotoImage(blurred_screen)
            
            self.canvas.create_image(0, 0, image=self.bg_image_tk, anchor="nw")
            
            # Add text overlay
            w, h = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
            self.canvas.create_text(w//2, h//2, text="Screen Locked", fill="white", font=("Arial", 50, "bold"))
            self.canvas.create_text(w//2, h//2 + 80, text="Waiting for authorized face...", fill="white", font=("Arial", 20))

            self.root.deiconify()
            self.root.lift()
            self.root.focus_force()
            self.is_locked = True

    def unlock(self):
        if self.is_locked:
            self.root.withdraw()
            self.is_locked = False

    def update(self):
        self.root.update()
