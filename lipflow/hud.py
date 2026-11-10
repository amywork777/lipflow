"""The floating pill at the bottom of the screen, plus the live mouth video above it.

States: listening (mouth video + lip-motion meter + live words), reading (animated
waveform), done (the text that was pasted, fades out), error/hint (fades out).
Native look: frosted HUD material, SF Symbols, dark appearance.
All methods must be called on the main thread (use AppHelper.callAfter).
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import objc
import Quartz
from AppKit import (
    NSAnimationContext, NSAppearance, NSBackingStoreBuffered, NSBezierPath, NSBitmapImageRep, NSColor, NSFont,
    NSFontWeightMedium, NSFontWeightSemibold, NSImage, NSImageScaleProportionallyUpOrDown,
    NSImageSymbolConfiguration, NSImageView, NSLineBreakByTruncatingHead, NSLineBreakByTruncatingTail, NSMakeRect,
    NSPanel, NSScreen, NSStatusWindowLevel, NSTextField, NSTimer, NSView, NSVisualEffectView,
    NSWindowCollectionBehaviorCanJoinAllSpaces, NSWindowCollectionBehaviorFullScreenAuxiliary,
    NSWindowStyleMaskBorderless, NSWindowStyleMaskNonactivatingPanel,
)
from Foundation import NSData, NSObject

W, H = 420, 60
VW, VH = 232, 144
BARS = 16
ACCENT = (1.0, 0.33, 0.45)   # lip pink
GREEN = (0.30, 0.85, 0.55)
AMBER = (1.0, 0.72, 0.25)
HUD_MATERIAL = 13            # NSVisualEffectMaterialHUDWindow
BEHIND_WINDOW, ACTIVE = 0, 1

STATES = {
    #           symbol                          tint    title colour
    "listening": ("mouth.fill",                  ACCENT),
    "reading":   ("waveform",                    (1, 1, 1)),
    "done":      ("checkmark",                   GREEN),
    "error":     ("exclamationmark.triangle.fill", AMBER),
}


def _rgb(c, a=1.0):
    return NSColor.colorWithSRGBRed_green_blue_alpha_(c[0], c[1], c[2], a)


def _cg(c, a=1.0):
    return Quartz.CGColorCreateSRGB(c[0], c[1], c[2], a)


def symbol(name: str, size: float = 15, weight=NSFontWeightSemibold) -> NSImage:
    img = NSImage.imageWithSystemSymbolName_accessibilityDescription_(name, None)
    cfg = NSImageSymbolConfiguration.configurationWithPointSize_weight_(size, weight)
    return img.imageWithSymbolConfiguration_(cfg)


def _nsimage(bgr: np.ndarray) -> NSImage:
    ok, buf = cv2.imencode(".bmp", bgr)  # BMP: no compression cost at 30 fps
    rep = NSBitmapImageRep.imageRepWithData_(NSData.dataWithBytes_length_(buf.tobytes(), len(buf)))
    img = NSImage.alloc().initWithSize_(rep.size())
    img.addRepresentation_(rep)
    return img


def _panel(frame) -> NSPanel:
    p = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
        frame, NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel, NSBackingStoreBuffered, False)
    p.setLevel_(NSStatusWindowLevel)
    p.setOpaque_(False)
    p.setBackgroundColor_(NSColor.clearColor())
    p.setHasShadow_(True)
    p.setIgnoresMouseEvents_(True)
    p.setHidesOnDeactivate_(False)
    p.setAppearance_(NSAppearance.appearanceNamed_("NSAppearanceNameDarkAqua"))
    p.setCollectionBehavior_(NSWindowCollectionBehaviorCanJoinAllSpaces | NSWindowCollectionBehaviorFullScreenAuxiliary)
    return p


def _glass(frame, radius: float, scrim: float = 0.8):
    """(outer view, content view). Liquid Glass (NSGlassEffectView, macOS 26+) when available,
    otherwise the frosted HUD material.

    Liquid Glass adapts to what's behind it and turns light over a white page, so a tint alone
    can't keep white text readable. The content view carries its own dark scrim: the glass still
    refracts at the edges, but text always sits on a dark surface, on any background."""
    content = NSView.alloc().initWithFrame_(NSMakeRect(0, 0, frame.size.width, frame.size.height))
    content.setWantsLayer_(True)
    content.layer().setCornerRadius_(radius)
    content.layer().setMasksToBounds_(True)
    content.layer().setBackgroundColor_(_cg((0.06, 0.06, 0.09), scrim))
    try:
        Glass = objc.lookUpClass("NSGlassEffectView")
    except objc.nosuchclass_error:
        Glass = None
    if Glass is not None:
        g = Glass.alloc().initWithFrame_(frame)
        g.setStyle_(0)  # NSGlassEffectViewStyleRegular
        g.setCornerRadius_(radius)
        g.setWantsLayer_(True)
        g.layer().setCornerRadius_(radius)
        g.layer().setMasksToBounds_(True)
        g.setTintColor_(_rgb((0.05, 0.05, 0.08), 0.5))
        g.setContentView_(content)
        return g, content
    v = NSVisualEffectView.alloc().initWithFrame_(frame)
    v.setMaterial_(HUD_MATERIAL)
    v.setBlendingMode_(BEHIND_WINDOW)
    v.setState_(ACTIVE)
    # Behind-window blur is composited separately by WindowServer. A layer's
    # cornerRadius clips our content, but the effect needs its own alpha mask.
    mask = NSImage.alloc().initWithSize_(frame.size)
    mask.lockFocus()
    try:
        NSColor.whiteColor().setFill()
        NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
            NSMakeRect(0, 0, frame.size.width, frame.size.height), radius, radius).fill()
    finally:
        mask.unlockFocus()
    v.setMaskImage_(mask)
    v.setWantsLayer_(True)
    v.layer().setCornerRadius_(radius)
    v.layer().setMasksToBounds_(True)
    v.addSubview_(content)
    return v, content


def _label(parent, frame, size, weight, color):
    t = NSTextField.alloc().initWithFrame_(frame)
    t.setBezeled_(False)
    t.setDrawsBackground_(False)
    t.setEditable_(False)
    t.setSelectable_(False)
    t.setTextColor_(color)
    t.setFont_(NSFont.systemFontOfSize_weight_(size, weight))
    parent.addSubview_(t)
    return t


class MeterView(NSView):
    """Lip-motion meter while listening; a travelling wave while reading."""

    def initWithFrame_(self, frame):
        self = objc.super(MeterView, self).initWithFrame_(frame)
        if self is None:
            return None
        self.levels = [0.0] * BARS
        self.mode = "idle"
        self.phase = 0.0
        return self

    def drawRect_(self, rect):
        if self.mode not in ("listening", "reading"):
            return
        b = self.bounds()
        step = b.size.width / BARS
        mid = b.size.height / 2
        for i, lv in enumerate(self.levels):
            if self.mode == "reading":
                lv = 0.22 + 0.22 * math.sin(self.phase * 2.0 - i * 0.5)
                color = _rgb((1, 1, 1), 0.55 + 0.35 * max(0.0, math.sin(self.phase * 2.0 - i * 0.5)))
            else:
                color = _rgb(ACCENT, 0.45 + 0.55 * lv)
            h = 3 + lv * (b.size.height - 6)
            color.setFill()
            NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                NSMakeRect(i * step + (step - 2.5) / 2, mid - h / 2, 2.5, h), 1.25, 1.25).fill()


class HUD(NSObject):
    def init(self):
        self = objc.super(HUD, self).init()
        if self is None:
            return None
        screen = NSScreen.mainScreen().visibleFrame()
        x = screen.origin.x + (screen.size.width - W) / 2
        y = screen.origin.y + 28

        # -- pill --------------------------------------------------------------------
        self.panel = _panel(NSMakeRect(x, y, W, H))
        outer, self.glass = _glass(NSMakeRect(0, 0, W, H), H / 2)
        self.panel.setContentView_(outer)

        badge = 36
        self.badge = NSView.alloc().initWithFrame_(NSMakeRect(12, (H - badge) / 2, badge, badge))
        self.badge.setWantsLayer_(True)
        self.badge.layer().setCornerRadius_(badge / 2)
        self.glass.addSubview_(self.badge)
        self.icon = NSImageView.alloc().initWithFrame_(NSMakeRect(0, 0, badge, badge))
        self.icon.setImageScaling_(0)  # NSImageScaleNone: keep the symbol's point size
        self.badge.addSubview_(self.icon)

        tx = 12 + badge + 12
        self.meter = MeterView.alloc().initWithFrame_(NSMakeRect(W - 18 - BARS * 5, (H - 28) / 2, BARS * 5, 28))
        self.glass.addSubview_(self.meter)
        self.title = _label(self.glass, NSMakeRect(tx, H / 2 + 2, W - tx - 20, 15), 10.5, NSFontWeightSemibold,
                            _rgb((1, 1, 1), 0.68))
        self.body = _label(self.glass, NSMakeRect(tx, H / 2 - 19, W - tx - 20, 19), 14, NSFontWeightMedium,
                           _rgb((1, 1, 1), 0.95))

        # -- mouth video ---------------------------------------------------------------
        self.video_panel = _panel(NSMakeRect(x + (W - VW) / 2, y + H + 10, VW, VH))
        vouter, vglass = _glass(NSMakeRect(0, 0, VW, VH), 22, scrim=0.7)
        self.video_panel.setContentView_(vouter)
        self.video = NSImageView.alloc().initWithFrame_(NSMakeRect(6, 6, VW - 12, VH - 12))
        self.video.setImageScaling_(NSImageScaleProportionallyUpOrDown)
        self.video.setWantsLayer_(True)
        self.video.layer().setCornerRadius_(16)
        self.video.layer().setMasksToBounds_(True)
        self.video.layer().setBorderWidth_(1.5)
        self.video.layer().setBorderColor_(_cg(ACCENT, 0.75))
        vglass.addSubview_(self.video)
        # "REC" tag in the corner
        tag = NSView.alloc().initWithFrame_(NSMakeRect(14, VH - 32, 44, 18))
        tag.setWantsLayer_(True)
        tag.layer().setCornerRadius_(9)
        tag.layer().setBackgroundColor_(_cg((0, 0, 0), 0.55))
        vglass.addSubview_(tag)
        dot = NSView.alloc().initWithFrame_(NSMakeRect(7, 6, 6, 6))
        dot.setWantsLayer_(True)
        dot.layer().setCornerRadius_(3)
        dot.layer().setBackgroundColor_(_cg(ACCENT))
        tag.addSubview_(dot)
        self.rec_dot = dot
        _label(tag, NSMakeRect(15, 1, 28, 14), 9.5, NSFontWeightSemibold, _rgb((1, 1, 1), 0.9)).setStringValue_("REC")

        self._hide_timer = None
        self._anim_timer = None
        return self

    # -- public (main thread) ------------------------------------------------------
    @objc.python_method
    def show(self, mode: str, title: str, body: str = "", hide_after: float | None = None):
        self._cancel_hide()
        sym, tint = STATES.get(mode, STATES["reading"])
        self.icon.setImage_(symbol(sym))
        self.icon.setContentTintColor_(_rgb(tint))
        self.badge.layer().setBackgroundColor_(_cg(tint, 0.16))
        self.title.setStringValue_(title.upper())
        self.body.setStringValue_(body)
        live = mode in ("listening", "reading")
        self.body.cell().setLineBreakMode_(NSLineBreakByTruncatingHead if live else NSLineBreakByTruncatingTail)
        tx = self.body.frame().origin.x
        width = W - tx - 20 - ((BARS * 5 + 14) if live else 0)
        self.body.setFrameSize_((width, self.body.frame().size.height))
        self.meter.mode = mode
        if mode != "listening":
            self.meter.levels = [0.0] * BARS
        self.meter.setNeedsDisplay_(True)
        if live:
            self._start_anim()
        else:
            self._stop_anim()
        self.panel.setAlphaValue_(1.0)
        self.panel.orderFrontRegardless()
        if mode == "listening":
            self.video_panel.setAlphaValue_(1.0)
            self.video_panel.orderFrontRegardless()
        else:
            self.video_panel.orderOut_(None)
            self.video.setImage_(None)
        if hide_after:
            self._hide_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                hide_after, self, "fadeOut:", None, False)

    @objc.python_method
    def set_text(self, body: str):
        self.body.setStringValue_(body)

    @objc.python_method
    def set_frame(self, video_bgr, level: float):
        if video_bgr is not None and self.meter.mode == "listening":
            self.video.setImage_(_nsimage(video_bgr))
        lv = self.meter.levels
        lv.pop(0)
        lv.append(float(max(0.0, min(1.0, level))))

    @objc.python_method
    def hide(self):
        self._cancel_hide()
        self._stop_anim()
        self.panel.orderOut_(None)
        self.video_panel.orderOut_(None)
        self.video.setImage_(None)

    # -- timers --------------------------------------------------------------------
    def fadeOut_(self, timer):
        self._hide_timer = None
        NSAnimationContext.beginGrouping()
        NSAnimationContext.currentContext().setDuration_(0.3)
        self.panel.animator().setAlphaValue_(0.0)
        self.video_panel.animator().setAlphaValue_(0.0)
        NSAnimationContext.endGrouping()
        self._hide_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            0.35, self, "orderOut:", None, False)

    def orderOut_(self, timer):
        self._hide_timer = None
        self.hide()

    def tick_(self, timer):
        self.meter.phase += 0.18
        self.meter.setNeedsDisplay_(True)
        # breathe the REC dot
        self.rec_dot.layer().setOpacity_(0.55 + 0.45 * (0.5 + 0.5 * math.sin(self.meter.phase * 1.4)))

    @objc.python_method
    def _start_anim(self):
        if self._anim_timer is None:
            self._anim_timer = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
                1 / 30, self, "tick:", None, True)

    @objc.python_method
    def _stop_anim(self):
        if self._anim_timer is not None:
            self._anim_timer.invalidate()
            self._anim_timer = None

    @objc.python_method
    def _cancel_hide(self):
        if self._hide_timer is not None:
            self._hide_timer.invalidate()
            self._hide_timer = None
