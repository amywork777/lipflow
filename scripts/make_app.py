"""Build ~/Applications/Lipflow.app: a small bundle that launches this checkout's venv.

Why a bundle: macOS grants Camera / Input Monitoring / Accessibility per app. Run from a
terminal, Lipflow borrows the terminal's permissions (and asks for them in the terminal's name);
as an app it gets its own entries called "Lipflow", starts from Launchpad/Spotlight, and can be
added to Login Items.

    uv run python scripts/make_app.py [--dest /Applications]
"""
from __future__ import annotations

import argparse
import os
import platform
import plistlib
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def draw_icon(path_png: str, size: int = 1024):
    """Pink squircle with a white SF Symbols mouth."""
    from AppKit import (NSBezierPath, NSBitmapImageRep, NSCalibratedRGBColorSpace, NSColor, NSGradient,
                        NSGraphicsContext, NSImage, NSImageSymbolConfiguration, NSMakeRect, NSPNGFileType)
    rep = NSBitmapImageRep.alloc().initWithBitmapDataPlanes_pixelsWide_pixelsHigh_bitsPerSample_samplesPerPixel_hasAlpha_isPlanar_colorSpaceName_bytesPerRow_bitsPerPixel_(
        None, size, size, 8, 4, True, False, NSCalibratedRGBColorSpace, 0, 0)
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.setCurrentContext_(NSGraphicsContext.graphicsContextWithBitmapImageRep_(rep))
    inset = size * 0.1
    r = NSMakeRect(inset, inset, size - 2 * inset, size - 2 * inset)
    shape = NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(r, size * 0.18, size * 0.18)
    NSGradient.alloc().initWithStartingColor_endingColor_(
        NSColor.colorWithSRGBRed_green_blue_alpha_(1.0, 0.45, 0.55, 1),
        NSColor.colorWithSRGBRed_green_blue_alpha_(0.78, 0.16, 0.38, 1)).drawInBezierPath_angle_(shape, -90)
    NSColor.colorWithWhite_alpha_(1, 0.25).setStroke()
    shape.setLineWidth_(size * 0.006)
    shape.stroke()
    cfg = NSImageSymbolConfiguration.configurationWithPointSize_weight_(size * 0.36, 6)  # semibold
    sym = NSImage.imageWithSystemSymbolName_accessibilityDescription_("mouth.fill", None).imageWithSymbolConfiguration_(cfg)
    tinted = NSImage.alloc().initWithSize_(sym.size())
    tinted.lockFocus()
    sym.drawInRect_(NSMakeRect(0, 0, sym.size().width, sym.size().height))
    NSColor.whiteColor().set()
    from AppKit import NSCompositingOperationSourceAtop, NSRectFillUsingOperation
    NSRectFillUsingOperation(NSMakeRect(0, 0, sym.size().width, sym.size().height), NSCompositingOperationSourceAtop)
    tinted.unlockFocus()
    w, h = tinted.size().width, tinted.size().height
    tinted.drawInRect_(NSMakeRect((size - w) / 2, (size - h) / 2, w, h))
    NSGraphicsContext.restoreGraphicsState()
    rep.representationUsingType_properties_(NSPNGFileType, None).writeToFile_atomically_(path_png, True)


def make_icns(dest_icns: str):
    tmp = tempfile.mkdtemp()
    base = os.path.join(tmp, "base.png")
    draw_icon(base)
    iconset = os.path.join(tmp, "Lipflow.iconset")
    os.makedirs(iconset)
    for s in (16, 32, 128, 256, 512):
        for scale in (1, 2):
            px = s * scale
            name = f"icon_{s}x{s}{'@2x' if scale == 2 else ''}.png"
            subprocess.run(["sips", "-z", str(px), str(px), base, "--out", os.path.join(iconset, name)],
                           check=True, capture_output=True)
    subprocess.run(["iconutil", "-c", "icns", iconset, "-o", dest_icns], check=True)
    shutil.rmtree(tmp)


def build(dest_dir: str = "/Applications") -> str:
    app = os.path.join(os.path.expanduser(dest_dir), "Lipflow.app")
    if os.path.exists(app):
        shutil.rmtree(app)
    macos = os.path.join(app, "Contents", "MacOS")
    res = os.path.join(app, "Contents", "Resources")
    os.makedirs(macos)
    os.makedirs(res)
    make_icns(os.path.join(res, "Lipflow.icns"))
    plistlib.dump({
        "CFBundleName": "Lipflow",
        "CFBundleDisplayName": "Lipflow",
        "CFBundleIdentifier": "app.lipflow.Lipflow",
        "CFBundleVersion": "1",
        "CFBundleShortVersionString": "0.1",
        "CFBundlePackageType": "APPL",
        "CFBundleExecutable": "Lipflow",
        "CFBundleIconFile": "Lipflow",
        "LSUIElement": True,  # menu-bar app: no Dock icon
        "LSMinimumSystemVersion": "13.0",
        "NSCameraUsageDescription": "Lipflow reads your lips from the camera to type what you mouth. "
                                    "Video never leaves this Mac.",
        "NSMicrophoneUsageDescription": "Whisper mode listens to a soft whisper while you hold the key, "
                                        "to read your lips more accurately. Audio never leaves this Mac.",
        "NSHighResolutionCapable": True,
    }, open(os.path.join(app, "Contents", "Info.plist"), "wb"))
    launcher = os.path.join(macos, "Lipflow")
    _compile_launcher(launcher)
    # Ad-hoc signature so Accessibility / Input Monitoring bind to this app, not to
    # whatever Python the old shell launcher exec'd. Unsigned bundles record a grant
    # the running process can never satisfy, and setup's Continue button stays off.
    subprocess.run(["codesign", "--force", "--sign", "-", "--identifier", "app.lipflow.Lipflow", app],
                   check=True)
    subprocess.run(["touch", app])
    return app


def _compile_launcher(dest: str) -> None:
    """Mach-O main executable that loads the venv in-process. See lipflow_launcher.c."""
    py = os.path.realpath(os.path.join(ROOT, ".venv", "bin", "python"))
    pyhome = os.path.dirname(os.path.dirname(py))
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    include = os.path.join(pyhome, "include", version)
    lib = os.path.join(pyhome, "lib")
    site = os.path.join(ROOT, ".venv", "lib", version, "site-packages")
    src = os.path.join(os.path.dirname(__file__), "lipflow_launcher.c")
    if not os.path.isfile(os.path.join(include, "Python.h")):
        raise SystemExit(f"no Python headers at {include}; run ./setup.sh first")
    if not os.path.isdir(site):
        raise SystemExit(f"no venv site-packages at {site}; run ./setup.sh first")
    subprocess.run([
        "clang", "-O2", "-arch", platform.machine(),  # match the venv's libpython
        f"-I{include}",
        f"-DLIPFLOW_ROOT={root_c(ROOT)}",
        f"-DLIPFLOW_PYHOME={root_c(pyhome)}",
        f"-DLIPFLOW_SITE={root_c(site)}",
        src, f"-L{lib}", f"-l{version}",
        f"-Wl,-rpath,{lib}",
        "-o", dest,
    ], check=True)


def root_c(path: str) -> str:
    """A C string literal, including the quotes clang expects in -DNAME=value."""
    return '"' + path.replace("\\", "\\\\").replace('"', '\\"') + '"'


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="/Applications")
    print(build(ap.parse_args().dest))
