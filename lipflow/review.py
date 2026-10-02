"""Focused, floating macOS candidate panel. Number keys stay inside this window."""
import objc
from Foundation import NSObject
from AppKit import (NSPanel, NSButton, NSTextField, NSMakeRect, NSBackingStoreBuffered,
                    NSWindowStyleMaskTitled, NSFloatingWindowLevel, NSApplication)


class Review(NSObject):
    def init(self):
        self = objc.super(Review, self).init()
        self.panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, 720, 380), NSWindowStyleMaskTitled, NSBackingStoreBuffered, False)
        self.panel.setTitle_("Lipflow · Choose / 选择识别结果")
        self.panel.setLevel_(NSFloatingWindowLevel)
        self.panel.setReleasedWhenClosed_(False)
        self.callback = None
        return self

    @objc.python_method
    def show(self, choices, reason, callback):
        self.choices, self.callback = choices, callback
        root = self.panel.contentView()
        for view in list(root.subviews()):
            view.removeFromSuperview()
        label = NSTextField.wrappingLabelWithString_(reason)
        label.setFrame_(NSMakeRect(20, 286, 680, 72))
        root.addSubview_(label)
        for i, (title, text) in enumerate(choices):
            button = NSButton.buttonWithTitle_target_action_(f"{i+1}. {title}: {text}", self, "choose:")
            button.setFrame_(NSMakeRect(20, 228-i*58, 680, 52))
            button.setTag_(i)
            button.setKeyEquivalent_(str(i+1))
            button.setKeyEquivalentModifierMask_(0)
            button.setToolTip_(text)
            root.addSubview_(button)
        cancel = NSButton.buttonWithTitle_target_action_("Cancel / 重说 (Esc)", self, "cancel:")
        cancel.setFrame_(NSMakeRect(20, 20, 240, 38))
        cancel.setKeyEquivalent_("\x1b")
        root.addSubview_(cancel)
        self.panel.center()
        self.panel.makeKeyAndOrderFront_(None)
        app = NSApplication.sharedApplication()
        app.activate() if hasattr(app, 'activate') else app.activateIgnoringOtherApps_(True)

    def choose_(self, sender):
        callback, self.callback = self.callback, None
        text = self.choices[sender.tag()][1]
        self.panel.orderOut_(None)
        if callback:
            callback(text)

    def cancel_(self, sender):
        callback, self.callback = self.callback, None
        self.panel.orderOut_(None)
        if callback:
            callback(None)

    @objc.python_method
    def dismiss(self):
        self.callback = None
        self.panel.orderOut_(None)
