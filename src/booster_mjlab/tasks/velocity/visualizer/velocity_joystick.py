"""2D joystick panel widget for driving the velocity command from Viser.

Viser has no joystick component, so a script in a hidden ``srcdoc`` iframe
(added through ``gui.add_html``) builds the pad directly in the panel DOM,
where it inherits Viser's font and Mantine theme variables.

Scripts inside the panel cannot talk to the server directly; instead the
widget writes its state into a hidden Viser text input (a native ``<input>``
React forwards to Python), which is the only GUI element that client-side
script can update and have synced back. The toggles are regular Viser
checkboxes that the script reads from the DOM.

The pad writes into the per-axis sliders that mjlab's velocity command creates
in its ``create_gui`` and switches on that GUI's "Enable" override, so the
sliders, the pad and the applied command always agree.
"""

from __future__ import annotations

import html
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import viser


@dataclass(kw_only=True)
class VelocityJoystickCfg:
    """Configuration of the panel joystick."""

    pad_size_px: int = 112
    """Diameter of the pad in the control panel."""
    return_to_center: bool = True
    """Initial state of the widget's "Recenter" toggle."""
    steer: bool = False
    """Initial state of the widget's "Steer" toggle. When on, sideways pad
    motion commands yaw rate instead of lateral velocity, like a car."""


# Value the hidden text input starts with; the widget script uses it to find
# the input element in the panel DOM.
_STATE_TOKEN = "__velocity_joystick__"


class VelocityJoystick:
    """Panel joystick that drives the velocity sliders."""

    def __init__(
        self,
        cfg: VelocityJoystickCfg,
        server: viser.ViserServer,
        enabled: viser.GuiCheckboxHandle,
        sliders: list[viser.GuiSliderHandle],
        on_change: Callable[[], None] | None = None,
    ) -> None:
        self._cfg = cfg
        self._override = enabled
        self._vx, self._vy, self._wz = sliders
        self._on_change = on_change

        with server.gui.add_folder("Joystick"):
            server.gui.add_html(_iframe_html(cfg.pad_size_px, _STATE_TOKEN))
            # Hidden by the widget script; carries "vx,vy,wz" as fractions of
            # each slider's max in [-1, 1].
            self._state = server.gui.add_text("joystick_state", _STATE_TOKEN)
            # The script reads these two by position; keep them in this order.
            server.gui.add_checkbox(
                "Recenter",
                cfg.return_to_center,
                hint="Snap back to zero when the pad or turn bar is released.",
            )
            server.gui.add_checkbox(
                "Steer",
                cfg.steer,
                hint="Sideways pad motion turns instead of strafing.",
            )

        @self._state.on_update
        def _(_) -> None:
            self._apply(self._state.value)

    def _apply(self, state: str) -> None:
        try:
            fx, fy, fz = (float(v) for v in state.split(","))
        except ValueError:
            return
        self._override.value = True
        self._vx.value = _clip(fx) * self._vx.max
        self._vy.value = _clip(fy) * self._vy.max
        self._wz.value = _clip(fz) * self._wz.max
        if self._on_change is not None:
            self._on_change()


def _clip(x: float) -> float:
    return max(-1.0, min(1.0, x))


def _iframe_html(pad_size_px: int, token: str) -> str:
    """HTML for the panel: a hidden iframe whose ``srcdoc`` holds the script."""
    doc = _WIDGET_DOC.replace("__TOKEN__", token).replace(
        "__PAD_PX__", str(pad_size_px)
    )
    return (
        f'<iframe srcdoc="{html.escape(doc, quote=True)}" '
        'style="display:none"></iframe>'
    )


_WIDGET_DOC = r"""<!doctype html>
<html><head><meta charset="utf-8"></head><body>
<script>
(function () {
  var TOKEN = "__TOKEN__";
  var PAD_PX = __PAD_PX__;
  var SVG_NS = "http://www.w3.org/2000/svg";
  var frame = window.frameElement;
  if (!frame) return;
  var doc = window.parent.document;
  // Html component div; its parent holds the text input and checkboxes.
  var host = frame.parentElement;
  var container = host.parentElement;

  var state = { vx: 0, vy: 0, wz: 0 };
  var active = null; // "pad" | "bar" | null

  // ---- DOM helpers; styles use Mantine's theme variables. ----
  function el(tag, style, parent) {
    var e = doc.createElement(tag);
    if (style) e.style.cssText = style;
    if (parent) parent.appendChild(e);
    return e;
  }
  function svg(tag, attrs, parent) {
    var e = doc.createElementNS(SVG_NS, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  var PRIMARY = "var(--mantine-primary-color-filled, #228be6)";
  var BORDER = "var(--mantine-color-default-border, #ced4da)";
  var labelBoxes = [];
  // Mirrors Viser's labeled input row: dimmed label column, then the input.
  function row(label) {
    var r = el("div", "display:flex;align-items:center;padding:0 var(--mantine-spacing-xs, 10px) 0.5em", host);
    var box = el("div", "width:6.6em;flex-shrink:0;padding-right:var(--mantine-spacing-xs, 10px);box-sizing:border-box", r);
    var t = el("div", "color:var(--mantine-color-dimmed, #868e96);font-size:0.875em;font-weight:450;line-height:1.375em;letter-spacing:-0.75px", box);
    t.textContent = label;
    labelBoxes.push(box);
    return el("div", "flex-grow:1;min-width:0", r);
  }

  // ---- Pad: up = +vx (forward), left = +vy (strafe) or +wz (steer). ----
  var pad = svg("svg", {
    viewBox: "-1.1 -1.1 2.2 2.2", width: PAD_PX, height: PAD_PX,
    style: "display:block;touch-action:none;cursor:crosshair;overflow:visible",
  }, row("move"));
  var thin = { fill: "none", stroke: BORDER, "vector-effect": "non-scaling-stroke" };
  svg("circle", { r: 1, fill: "var(--mantine-color-default-hover, #f8f9fa)", stroke: BORDER, "vector-effect": "non-scaling-stroke" }, pad);
  svg("circle", Object.assign({ r: 0.5, "stroke-dasharray": "2 3" }, thin), pad);
  svg("path", Object.assign({ d: "M-1 0H1M0 -1V1" }, thin), pad);
  svg("path", { d: "M0 -1.1L-0.07 -1.01H0.07Z", fill: PRIMARY }, pad);
  var stem = svg("line", { x1: 0, y1: 0, stroke: PRIMARY, "stroke-width": 2, "stroke-opacity": 0.5, "vector-effect": "non-scaling-stroke" }, pad);
  var knob = svg("circle", { r: 0.13, fill: PRIMARY }, pad);

  // ---- Turn bar: styled like Viser's slider; left = +wz (turn left). ----
  var bar = el("div", "position:relative;height:1.1em;touch-action:none;cursor:pointer", row("turn"));
  el("div", "position:absolute;left:0;right:0;top:50%;height:4px;margin-top:-2px;border-radius:2px;background:" + BORDER, bar);
  el("div", "position:absolute;left:50%;top:50%;width:1px;height:8px;margin-top:-4px;background:var(--mantine-color-dimmed, #868e96)", bar);
  var fill = el("div", "position:absolute;top:50%;height:4px;margin-top:-2px;border-radius:2px;background:" + PRIMARY, bar);
  var thumb = el("div", "position:absolute;top:50%;width:0.5rem;height:0.75rem;margin:-0.375rem 0 0 -0.25rem;border-radius:2px;background:" + PRIMARY, bar);

  // ---- Viser elements this widget pairs with. ----
  var input = null;
  function findInput() {
    if (input && input.isConnected) return input;
    input = null;
    var inputs = container.querySelectorAll("input");
    for (var i = 0; i < inputs.length; i++) {
      if (inputs[i].value === TOKEN || inputs[i].dataset.velocityJoystick === "1") {
        input = inputs[i];
        input.dataset.velocityJoystick = "1";
        adoptRow(input);
        return input;
      }
    }
    return null;
  }
  function adoptRow(inputEl) {
    var node = inputEl;
    while (node && node.parentElement !== container) node = node.parentElement;
    if (!node) return;
    // Match Viser's label column width, which depends on folder depth.
    var label = node.querySelector("label");
    var box = label && label.parentElement && label.parentElement.parentElement;
    if (box && box.style.width) labelBoxes.forEach(function (b) { b.style.width = box.style.width; });
    node.style.display = "none";
  }
  function toggles() {
    var boxes = container.querySelectorAll("input[type=checkbox]");
    return { recenter: boxes[0], steer: boxes[1] };
  }
  function steering() { var s = toggles().steer; return !!(s && s.checked); }
  function recentering() { var r = toggles().recenter; return !r || r.checked; }

  var lastSent = "";
  function send() {
    var e = findInput();
    if (!e) return;
    var msg = state.vx.toFixed(3) + "," + state.vy.toFixed(3) + "," + state.wz.toFixed(3);
    if (msg === lastSent) return;
    lastSent = msg;
    // Go through the native setter so React sees the change, then fire the
    // 'input' event React listens to for controlled inputs.
    var setter = Object.getOwnPropertyDescriptor(window.parent.HTMLInputElement.prototype, "value").set;
    setter.call(e, msg);
    e.dispatchEvent(new Event("input", { bubbles: true }));
  }
  // Retry until the panel has rendered the text input.
  var tries = 0;
  var poll = setInterval(function () {
    if (findInput() || ++tries > 100) clearInterval(poll);
  }, 100);

  function draw() {
    var steer = steering();
    var side = steer ? state.wz : state.vy;
    var kx = -side, ky = -state.vx;
    stem.setAttribute("x2", kx); stem.setAttribute("y2", ky);
    knob.setAttribute("cx", kx); knob.setAttribute("cy", ky);
    // Greyed out in steer mode, where the pad owns yaw.
    bar.style.opacity = steer ? 0.35 : 1;
    bar.style.pointerEvents = steer ? "none" : "auto";
    var pos = 50 - state.wz * 50;
    thumb.style.left = pos + "%";
    fill.style.left = Math.min(pos, 50) + "%";
    fill.style.width = Math.abs(pos - 50) + "%";
  }

  // ---- Input. ----
  function clamp(v) { return Math.max(-1, Math.min(1, v)); }
  function applyPointer(ev) {
    if (active === "pad") {
      var rect = pad.getBoundingClientRect();
      var s = 1.1 / (rect.width / 2);
      var dx = (ev.clientX - rect.left - rect.width / 2) * s;
      var dy = (ev.clientY - rect.top - rect.height / 2) * s;
      var n = Math.hypot(dx, dy);
      if (n > 1) { dx /= n; dy /= n; }
      state.vx = -dy;
      if (steering()) { state.wz = -dx; state.vy = 0; } else { state.vy = -dx; }
    } else if (active === "bar") {
      var r = bar.getBoundingClientRect();
      state.wz = clamp(1 - 2 * (ev.clientX - r.left) / r.width);
    }
    draw(); send();
  }
  function grab(target, name) {
    target.addEventListener("pointerdown", function (ev) {
      active = name;
      target.setPointerCapture(ev.pointerId);
      applyPointer(ev);
    });
    target.addEventListener("pointermove", function (ev) { if (active === name) applyPointer(ev); });
    target.addEventListener("pointerup", release);
    target.addEventListener("pointercancel", release);
    target.addEventListener("dblclick", function () { state.vx = 0; state.vy = 0; state.wz = 0; draw(); send(); });
  }
  function release() {
    if (!active) return;
    if (recentering()) {
      if (active === "pad") { state.vx = 0; state.vy = 0; if (steering()) state.wz = 0; } else { state.wz = 0; }
    }
    active = null;
    draw(); send();
  }
  grab(pad, "pad");
  grab(bar, "bar");
  container.addEventListener("change", function (ev) {
    if (ev.target !== toggles().steer) return;
    // Sideways input changes meaning; drop it rather than carry it over.
    state.vy = 0; state.wz = 0; draw(); send();
  });
  draw();
})();
</script></body></html>
"""
