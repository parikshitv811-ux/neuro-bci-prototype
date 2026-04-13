"""
BCI Level 2 — Autonomous Execution Engine
==========================================
Translates Claude's structured action plan into OS-level commands.
Supports: Windows (Win32/UIAutomation), Linux (xdotool/AT-SPI),
          macOS (AppleScript/Accessibility API), and simulated mode.
"""

import time, sys, json, subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass
class ExecutionResult:
    success: bool
    action: str
    step: int
    message: str
    simulated: bool = True
    timestamp: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()


class ExecutionEngine:
    """
    Executes structured action plans from the Claude agent.
    
    Modes:
        simulate=True  — logs actions without touching the OS (safe default)
        simulate=False — real OS interaction via PyAutoGUI / subprocess
    
    Platform detection is automatic.
    """

    def __init__(self, simulate: bool = True):
        self.simulate = simulate
        self._platform = sys.platform
        self._pyautogui = None
        self._log: list = []

        if not simulate:
            try:
                import pyautogui
                pyautogui.FAILSAFE = True
                pyautogui.PAUSE = 0.05
                self._pyautogui = pyautogui
                print("[Execution] PyAutoGUI loaded — real OS execution enabled")
            except ImportError:
                print("[Execution] PyAutoGUI not found — falling back to simulation")
                self.simulate = True

    def run_plan(self, plan: dict) -> list[ExecutionResult]:
        """Execute all steps in an action plan. Returns list of results."""
        results = []
        steps = plan.get("steps", [])
        if not steps:
            return [ExecutionResult(True, "idle", 0, "No steps to execute", self.simulate)]

        for step in steps:
            result = self._dispatch_step(step)
            results.append(result)
            self._log.append(result)
            if not result.success:
                print(f"  [Execution] Step {step['step']} FAILED: {result.message}")
                break
            time.sleep(0.1)  # brief inter-step pause

        return results

    def _dispatch_step(self, step: dict) -> ExecutionResult:
        cmd = step.get("os_command", "")
        params = step.get("params", {})
        desc = step.get("description", cmd)
        step_n = step.get("step", 0)

        dispatch_map = {
            "open_app":   self._open_app,
            "type_text":  self._type_text,
            "click":      self._click,
            "scroll":     self._scroll,
            "hotkey":     self._hotkey,
            "send_email": self._send_email,
            "idle":       self._idle,
        }

        handler = dispatch_map.get(cmd, self._unknown)
        return handler(step_n, desc, params)

    def _open_app(self, step_n, desc, params) -> ExecutionResult:
        app = params.get("app", "terminal")
        if self.simulate:
            print(f"  [SIM] Open app: {app}")
            return ExecutionResult(True, "open_app", step_n, f"Simulated open: {app}", True)

        try:
            if self._platform.startswith("linux"):
                cmds = {
                    "terminal": ["xterm"],
                    "browser": ["xdg-open", "https://"],
                    "email_client": ["thunderbird"],
                    "vscode": ["code"],
                }
                cmd = cmds.get(app, [app])
                subprocess.Popen(cmd)
            elif self._platform == "darwin":
                subprocess.Popen(["open", "-a", app])
            elif self._platform == "win32":
                import os; os.startfile(app)
            return ExecutionResult(True, "open_app", step_n, f"Opened: {app}", False)
        except Exception as e:
            return ExecutionResult(False, "open_app", step_n, str(e), False)

    def _type_text(self, step_n, desc, params) -> ExecutionResult:
        text = params.get("text", "")
        if self.simulate:
            print(f"  [SIM] Type: {repr(text[:50])}")
            return ExecutionResult(True, "type_text", step_n, f"Simulated type: {repr(text[:30])}", True)
        try:
            self._pyautogui.typewrite(text, interval=0.03)
            return ExecutionResult(True, "type_text", step_n, f"Typed: {len(text)} chars", False)
        except Exception as e:
            return ExecutionResult(False, "type_text", step_n, str(e), False)

    def _click(self, step_n, desc, params) -> ExecutionResult:
        button = params.get("button", "left")
        x = params.get("x"); y = params.get("y")
        if self.simulate:
            pos = f"({x},{y})" if x and y else "current position"
            print(f"  [SIM] Click {button} at {pos}")
            return ExecutionResult(True, "click", step_n, f"Simulated {button} click at {pos}", True)
        try:
            if x and y:
                self._pyautogui.click(x, y, button=button)
            else:
                self._pyautogui.click(button=button)
            return ExecutionResult(True, "click", step_n, f"{button} click executed", False)
        except Exception as e:
            return ExecutionResult(False, "click", step_n, str(e), False)

    def _scroll(self, step_n, desc, params) -> ExecutionResult:
        direction = params.get("direction", "down")
        amount = params.get("amount", 3)
        clicks = -amount if direction == "down" else amount
        if self.simulate:
            print(f"  [SIM] Scroll {direction} x{amount}")
            return ExecutionResult(True, "scroll", step_n, f"Simulated scroll {direction}", True)
        try:
            self._pyautogui.scroll(clicks)
            return ExecutionResult(True, "scroll", step_n, f"Scrolled {direction} x{amount}", False)
        except Exception as e:
            return ExecutionResult(False, "scroll", step_n, str(e), False)

    def _hotkey(self, step_n, desc, params) -> ExecutionResult:
        keys = params.get("keys", [])
        if self.simulate:
            print(f"  [SIM] Hotkey: {'+'.join(keys)}")
            return ExecutionResult(True, "hotkey", step_n, f"Simulated hotkey: {'+'.join(keys)}", True)
        try:
            self._pyautogui.hotkey(*keys)
            return ExecutionResult(True, "hotkey", step_n, f"Hotkey: {'+'.join(keys)}", False)
        except Exception as e:
            return ExecutionResult(False, "hotkey", step_n, str(e), False)

    def _send_email(self, step_n, desc, params) -> ExecutionResult:
        to = params.get("to", "")
        subject = params.get("subject", "(no subject)")
        body = params.get("body", "")
        if self.simulate:
            print(f"  [SIM] Send email → {to}: '{subject}'")
            print(f"        Body preview: {body[:80]}")
            return ExecutionResult(True, "send_email", step_n,
                                   f"Simulated email to {to}", True)
        # Real: open mailto link (cross-platform)
        try:
            import urllib.parse, webbrowser
            url = f"mailto:{urllib.parse.quote(to)}?subject={urllib.parse.quote(subject)}&body={urllib.parse.quote(body)}"
            webbrowser.open(url)
            return ExecutionResult(True, "send_email", step_n, f"Opened email client for {to}", False)
        except Exception as e:
            return ExecutionResult(False, "send_email", step_n, str(e), False)

    def _idle(self, step_n, desc, params) -> ExecutionResult:
        return ExecutionResult(True, "idle", step_n, "Idle — no action", self.simulate)

    def _unknown(self, step_n, desc, params) -> ExecutionResult:
        return ExecutionResult(False, "unknown", step_n, f"Unknown command: {desc}", self.simulate)

    def export_log(self) -> list:
        return [
            {
                "step": r.step,
                "action": r.action,
                "success": r.success,
                "message": r.message,
                "simulated": r.simulated,
                "ts": r.timestamp,
            }
            for r in self._log
        ]
