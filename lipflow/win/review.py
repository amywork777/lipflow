"""Focused Windows candidate window; number keys never reach the target editor."""
import tkinter as tk


class Review:
    def __init__(self, root):
        self.root, self.win = root, None
        self.callback = None

    def show(self, choices, reason, callback):
        self.dismiss()
        self.callback = callback
        self.win = tk.Toplevel(self.root)
        self.win.title('Lipflow · Choose / 选择识别结果')
        self.win.attributes('-topmost', True)
        tk.Label(self.win, text=reason, wraplength=680, justify='left').pack(padx=20, pady=15)
        for i, (title, text) in enumerate(choices):
            tk.Button(self.win, text=f'{i+1}. {title}: {text}', wraplength=680,
                      command=lambda t=text: self.finish(t)).pack(fill='x', padx=20, pady=8)
            self.win.bind(str(i+1), lambda event, t=text: self.finish(t))
        tk.Button(self.win, text='Cancel / 重说 (Esc)', command=lambda: self.finish(None)).pack(pady=15)
        self.win.bind('<Escape>', lambda event: self.finish(None))
        self.win.protocol('WM_DELETE_WINDOW', lambda: self.finish(None))
        self.win.lift()
        self.win.focus_force()

    def finish(self, text):
        callback = self.callback
        self.dismiss()
        if callback:
            callback(text)

    def dismiss(self):
        self.callback = None
        if self.win is not None:
            self.win.destroy()
            self.win = None
