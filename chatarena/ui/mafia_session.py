"""Server-side UI sessions. Gradio state holds only an opaque session identifier."""
from __future__ import annotations

import threading
import time
import uuid
from typing import Dict

from ..backends.human import Human, HumanBackendError


class GameSession:
    def __init__(self, arena):
        self.arena = arena
        self.controller = arena.discussion
        self.humans = [p.name for p in arena.players if isinstance(p.backend, Human)]
        if len(self.humans) > 1:
            arena.close()
            raise ValueError("The shared browser supports one human seat; use AI for other seats")
        self.human = self.humans[0] if self.humans else None
        self.lock = self.controller.lock if self.controller else threading.RLock()
        self.status = "paused"
        self.error = None
        self.last_access = time.monotonic()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._single_steps = 0
        if self.controller:
            self.controller.realtime = True
            self.controller.pause()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def start(self):
        with self.lock:
            self.error = None
            if self.controller:
                self.controller.resume()
            else:
                self._single_steps = 1
            self.status = "running"
            self._wake.set()

    def pause(self):
        with self.lock:
            if self.controller:
                self.controller.pause()
            self.status = "paused"

    def submit(self, text):
        if not self.human:
            raise ValueError("This is a spectator session; configure one Human player to participate")
        with self.lock:
            if self.controller:
                self.arena.current_timestep = self.controller.submit_human(self.human, text)
            else:
                if self.arena.environment.get_next_player() != self.human:
                    raise ValueError("Wait for your turn")
                self.arena.current_timestep = self.arena.environment.step(self.human, text)
            self._wake.set()

    def snapshot(self):
        with self.lock:
            self.last_access = time.monotonic()
            env = self.arena.environment
            # Never send other players' private observations to a human browser.
            messages = list(env.get_observation(self.human))
            views = {p.name: list(env.get_observation(p.name))
                     for p in self.arena.players if not self.human or p.name == self.human}
            return {"messages": messages, "views": views,
                    "human": self.human, "status": self.status, "error": self.error,
                    "terminal": env.is_terminal()}

    def _run(self):
        while not self._stop.is_set():
            if time.monotonic() - self.last_access > 1800:
                self.arena.close()
                self._stop.set()
                return
            try:
                if self.controller:
                    status = self.controller.tick()
                    with self.lock:
                        self.status = status
                        self.error = self.controller.error
                        self.arena.current_timestep = self.arena.environment.timestep()
                elif self._single_steps:
                    self._single_steps = 0
                    try:
                        result = self.arena.step()
                        with self.lock:
                            self.status = "terminal" if result.terminal else "paused"
                    except HumanBackendError:
                        self.status = "waiting_human"
            except Exception as exc:
                with self.lock:
                    if self.controller:
                        self.controller.pause()
                    self.status = "error"
                    self.error = f"{type(exc).__name__}: {exc}"
            self._wake.wait(0.03)
            self._wake.clear()

    def close(self):
        self._stop.set()
        self._wake.set()
        self.arena.close()


class SessionRegistry:
    def __init__(self):
        self.lock = threading.Lock()
        self.sessions: Dict[str, GameSession] = {}

    def add(self, arena):
        with self.lock:
            for key, session in list(self.sessions.items()):
                if session._stop.is_set():
                    del self.sessions[key]
            key = uuid.uuid4().hex
            self.sessions[key] = GameSession(arena)
            return key

    def get(self, key):
        with self.lock:
            return self.sessions.get(key)

    def remove(self, key):
        with self.lock:
            session = self.sessions.pop(key, None)
        if session:
            session.close()


sessions = SessionRegistry()
