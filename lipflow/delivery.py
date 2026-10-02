"""Shared decision and focus checks for both front ends."""
from .cleanup import basic_cleanup
from .confidence import Quality, assess


def quality_for(rec, rois):
    import numpy as np
    pixels = getattr(rec, 'mouth_pixels', [])
    return Quality(rec.face_ratio, float(np.mean(rois)), float(np.std(rois)),
                   float(np.median(pixels)) if pixels else 0.0)


def choose_result(hypotheses, greedy, quality, cleaner, ctx, context, policy='review', min_margin=0.5):
    decision = assess(hypotheses, greedy, quality, policy, min_margin,
                      language=getattr(cleaner, 'language', 'en'))
    if decision.action == 'retry':
        return decision, None, []
    candidates = [h.text for h in hypotheses]
    result = cleaner.process(candidates, context=context, names=ctx.names if ctx else None)
    choices = []
    for label, text in [('Cleanup / 纠错建议', result.proposed),
                        ('Raw / 原始识别', basic_cleanup(result.raw)),
                        *[('Alternative / 其他候选', basic_cleanup(c)) for c in candidates[1:]]]:
        if text and text not in [t for _, t in choices]:
            choices.append((label, text))
        if len(choices) == 3:
            break
    return decision, result, choices


def target_is_current(ctx):
    if ctx is None or not ctx.target:
        return False
    import sys
    if sys.platform == 'win32':
        import ctypes
        from ctypes import wintypes
        u = ctypes.WinDLL('user32')
        u.GetForegroundWindow.restype = wintypes.HWND
        return u.GetForegroundWindow() == ctx.target
    from AppKit import NSWorkspace
    app = NSWorkspace.sharedWorkspace().frontmostApplication()
    if app is None or app.processIdentifier() != ctx.target:
        return False
    if ctx.element is not None:
        from .context import _ax
        from ApplicationServices import AXUIElementCreateApplication, CFEqual
        focused = _ax(AXUIElementCreateApplication(ctx.target), 'AXFocusedUIElement')
        return focused is not None and bool(CFEqual(focused, ctx.element))
    return True


def restore_target(ctx):
    """Called only after choosing a focused review window; never for a stale automatic result."""
    if ctx is None or not ctx.target:
        return False
    import sys
    if sys.platform == 'win32':
        import ctypes
        from ctypes import wintypes
        u = ctypes.WinDLL('user32')
        u.SetForegroundWindow.argtypes = [wintypes.HWND]
        return bool(u.SetForegroundWindow(ctx.target))
    from AppKit import NSRunningApplication, NSApplicationActivateIgnoringOtherApps
    app = NSRunningApplication.runningApplicationWithProcessIdentifier_(ctx.target)
    return bool(app and app.activateWithOptions_(NSApplicationActivateIgnoringOtherApps))
