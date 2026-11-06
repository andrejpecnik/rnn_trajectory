"""Playable pygame front-end.

Modes
-----
play   - you (green) against a computer tank (red) that aims with a trajectory predictor
watch  - two computer tanks, each aiming with a different predictor (live comparison)
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path

import numpy as np

from . import config as C
from .ai import AIController
from .aiming import intercept_point
from .arena import Match
from .behaviors import GAME_BEHAVIORS, MixedBehavior
from .data import save_recording
from .predictors import load_predictor

HUD_H = 56
WIN_W, WIN_H = int(C.ARENA_W), int(C.ARENA_H) + HUD_H

BG = (30, 34, 40)
GRID = (38, 43, 50)
OBST = (92, 98, 108)
OBST_EDGE = (130, 136, 146)
HUD_BG = (20, 22, 27)
TEXT = (225, 228, 232)
DIM = (150, 156, 165)
BULLET = (255, 220, 120)
TEAM = [(76, 175, 96), (214, 78, 64)]
TEAM_WATCH = [(70, 140, 230), (235, 150, 50)]

PREDICTOR_ORDER = ("static", "cv", "ctrv", "mlp", "lstm", "gru")

# enemy settings in "play" mode: (reload time [s], random aim error std [rad])
DIFFICULTY = {"easy": (1.4, 0.07), "normal": (1.0, 0.03), "hard": (C.RELOAD_TIME, 0.0)}


class HumanController:
    """Controller fed from the keyboard and mouse."""

    def __init__(self):
        self.throttle = 0.0
        self.turn = 0.0
        self.aim: float | None = None
        self.fire = False

    def reset(self):
        self.fire = False

    def control(self, tank, target_tracker, world, dt):
        return self.throttle, self.turn, self.aim, self.fire


class GameApp:
    def __init__(self, mode: str = "play", enemy: str = "gru", blue: str = "gru", red: str = "cv",
                 seed: int | None = None, autopilot: bool = False, record: bool = False,
                 headless: bool = False, speed: int = 1, assist: str = "gru", difficulty: str = "normal"):
        if headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            os.environ["SDL_AUDIODRIVER"] = "dummy"
        import pygame  # imported lazily so that the rest of the package works without a display
        self.pg = pygame
        pygame.init()
        pygame.display.set_caption("RNN Tank Arena - trajectory prediction")
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        self.clock = pygame.time.Clock()
        self.font = self._font(17)
        self.font_small = self._font(14)
        self.font_big = self._font(40, bold=True)
        self.overlay = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        self.background = self._render_background()

        self.mode = mode
        self.seed = seed if seed is not None else int(time.time()) % 100000
        self.rng = np.random.default_rng(self.seed)
        self._predictors: dict = {}
        self.speed = max(1, speed)
        self.paused = False
        self.show_prediction = True
        self.show_assist = False
        self.help_timer = 6.0 if mode == "play" and not autopilot else 0.0
        self.recording = record
        self.colors = TEAM if mode == "play" else TEAM_WATCH
        self.difficulty = difficulty

        if mode == "play":
            self.human = None if autopilot else HumanController()
            reload_time, noise = DIFFICULTY[difficulty]
            c0 = self.human or AIController(MixedBehavior(self.rng, pool=GAME_BEHAVIORS), self.predictor(enemy))
            c1 = AIController(MixedBehavior(self.rng, pool=GAME_BEHAVIORS), self.predictor(enemy), aim_noise=noise,
                              reload_time=reload_time, seed=self.seed)
            self.names = ["YOU" if self.human else "AUTOPILOT", "ENEMY"]
        elif mode == "watch":
            self.human = None
            c0 = AIController(MixedBehavior(self.rng, pool=GAME_BEHAVIORS), self.predictor(blue))
            c1 = AIController(MixedBehavior(self.rng, pool=GAME_BEHAVIORS), self.predictor(red))
            self.names = ["BLUE", "RED"]
        else:
            raise ValueError(mode)
        self.match = Match([c0, c1], seed=self.seed)
        if record:
            self.match.trackers[0].start_recording()
        self.assist_predictor = self.predictor(assist)
        self.assist_prediction = None
        self.assist_point = None
        self._assist_sample = -1

    # ------------------------------------------------------------------ helpers
    def _font(self, size: int, bold: bool = False):
        pg = self.pg
        try:
            f = pg.font.SysFont("segoeui,dejavusans,arial", size, bold=bold)
            if f is not None:
                return f
        except Exception:  # pragma: no cover - depends on installed fonts
            pass
        return pg.font.Font(None, int(size * 1.3))

    def predictor(self, name: str):
        if name not in self._predictors:
            self._predictors[name] = load_predictor(name)
        return self._predictors[name]

    def to_screen(self, x: float, y: float) -> tuple[int, int]:
        return int(round(x)), int(round(y + HUD_H))

    def _render_background(self):
        pg = self.pg
        surf = pg.Surface((WIN_W, WIN_H))
        surf.fill(BG)
        for x in range(0, WIN_W, 50):
            pg.draw.line(surf, GRID, (x, HUD_H), (x, WIN_H))
        for y in range(HUD_H, WIN_H, 50):
            pg.draw.line(surf, GRID, (0, y), (WIN_W, y))
        from .sim import World
        for o in World().obstacles:
            r = pg.Rect(int(o.x), int(o.y + HUD_H), int(o.w), int(o.h))
            pg.draw.rect(surf, OBST, r, border_radius=4)
            pg.draw.rect(surf, OBST_EDGE, r, width=2, border_radius=4)
        pg.draw.rect(surf, HUD_BG, (0, 0, WIN_W, HUD_H))
        pg.draw.line(surf, (60, 65, 75), (0, HUD_H - 1), (WIN_W, HUD_H - 1), 2)
        return surf

    # ------------------------------------------------------------------ input
    def handle_events(self) -> bool:
        pg = self.pg
        for ev in pg.event.get():
            if ev.type == pg.QUIT:
                return False
            if ev.type == pg.KEYDOWN:
                if ev.key == pg.K_ESCAPE:
                    return False
                if ev.key == pg.K_p:
                    self.paused = not self.paused
                elif ev.key == pg.K_v:
                    self.show_prediction = not self.show_prediction
                elif ev.key == pg.K_t:
                    self.show_assist = not self.show_assist
                elif ev.key == pg.K_F1 or ev.key == pg.K_h:
                    self.help_timer = 0.0 if self.help_timer > 0 else 1e9
                elif ev.key == pg.K_r:
                    self.match.reset_round()
                elif ev.key in (pg.K_PLUS, pg.K_KP_PLUS, pg.K_EQUALS):
                    self.speed = min(16, self.speed * 2)
                elif ev.key in (pg.K_MINUS, pg.K_KP_MINUS):
                    self.speed = max(1, self.speed // 2)
                elif pg.K_1 <= ev.key <= pg.K_6 and self.mode == "play":
                    name = PREDICTOR_ORDER[ev.key - pg.K_1]
                    self.match.controllers[1].predictor = self.predictor(name)
                    self.match.trackers[0].errors.clear()
        if self.human:
            keys = pg.key.get_pressed()
            self.human.throttle = float(keys[pg.K_w] or keys[pg.K_UP]) - float(keys[pg.K_s] or keys[pg.K_DOWN])
            self.human.turn = float(keys[pg.K_d] or keys[pg.K_RIGHT]) - float(keys[pg.K_a] or keys[pg.K_LEFT])
            mx, my = pg.mouse.get_pos()
            tank = self.match.tanks[0]
            self.human.aim = math.atan2(my - HUD_H - tank.y, mx - tank.x)
            self.human.fire = bool(pg.mouse.get_pressed()[0] or keys[pg.K_SPACE])
        return True

    # ------------------------------------------------------------------ update
    def update(self, frame_dt: float):
        if self.help_timer > 0 and self.help_timer < 1e8:
            self.help_timer = max(0.0, self.help_timer - frame_dt)
        if self.paused:
            return
        for _ in range(self.speed):
            self.match.step(C.SIM_DT)
        if self.show_assist:
            self._update_assist()

    def _update_assist(self):
        """Aim assist for the player: predict the enemy with the RNN and mark the intercept point."""
        tracker = self.match.trackers[1]
        if tracker.samples == self._assist_sample or not tracker.pos:
            return
        self._assist_sample = tracker.samples
        pos, head, rays = tracker.history()
        self.assist_prediction = self.assist_predictor.predict(pos, head, rays)
        me = self.match.tanks[0]
        self.assist_point, _ = intercept_point((me.x, me.y), pos[-1], self.assist_prediction)

    # ------------------------------------------------------------------ drawing
    def draw(self):
        pg = self.pg
        s = self.screen
        s.blit(self.background, (0, 0))
        self.overlay.fill((0, 0, 0, 0))
        m = self.match

        if self.show_prediction:
            for i, ctrl in enumerate(m.controllers):
                if isinstance(ctrl, AIController) and ctrl.prediction is not None and not m.round_over:
                    if self.mode == "play" and i == 0:
                        continue  # autopilot: show only the enemy's view to keep the screen clean
                    self._draw_path(ctrl.prediction, self.colors[i], ctrl.aim_point)
        if self.show_assist and self.assist_prediction is not None and not m.round_over:
            self._draw_path(self.assist_prediction, (120, 255, 140), self.assist_point, assist=True)

        for i, t in enumerate(m.tanks):
            if t.alive:
                self._draw_tank(t, self.colors[i])
        for b in m.bullets:
            pg.draw.circle(s, BULLET, self.to_screen(b.x, b.y), int(C.BULLET_RADIUS))
        for e in m.explosions:
            k = e.t / e.duration
            r = max(2, int(10 + 26 * e.size * k))
            alpha = int(220 * (1 - k))
            pg.draw.circle(self.overlay, (255, 170, 60, alpha), self.to_screen(e.x, e.y), r)
            pg.draw.circle(self.overlay, (255, 240, 180, alpha), self.to_screen(e.x, e.y), max(1, r // 2))
        s.blit(self.overlay, (0, 0))
        self._draw_hud()
        if m.round_over:
            self._draw_banner()
        if self.help_timer > 0:
            self._draw_help()
        if self.paused:
            self._center_text("PAUSED", self.font_big, WIN_H // 2)

    def _draw_path(self, pred: np.ndarray, color, aim_point=None, assist: bool = False):
        pg = self.pg
        n = len(pred)
        for k in range(1, n, 2):
            alpha = int(230 * (1 - k / n)) + 25
            pg.draw.circle(self.overlay, (*color, alpha), self.to_screen(*pred[k]), 3 if not assist else 2)
        if aim_point is not None:
            x, y = self.to_screen(*aim_point)
            c = (*color, 230)
            pg.draw.circle(self.overlay, c, (x, y), 10, 2)
            pg.draw.line(self.overlay, c, (x - 15, y), (x - 5, y), 2)
            pg.draw.line(self.overlay, c, (x + 5, y), (x + 15, y), 2)
            pg.draw.line(self.overlay, c, (x, y - 15), (x, y - 5), 2)
            pg.draw.line(self.overlay, c, (x, y + 5), (x, y + 15), 2)

    def _draw_tank(self, t, color):
        pg = self.pg
        s = self.screen
        cx, cy = self.to_screen(t.x, t.y)
        ch, sh = math.cos(t.heading), math.sin(t.heading)

        def tr(px, py):
            return (cx + px * ch - py * sh, cy + px * sh + py * ch)

        dark = tuple(int(c * 0.55) for c in color)
        for side in (-1, 1):  # tracks
            pg.draw.polygon(s, (40, 42, 46), [tr(-22, side * 17), tr(22, side * 17),
                                             tr(22, side * 10), tr(-22, side * 10)])
        body = [tr(-19, -12), tr(19, -12), tr(23, 0), tr(19, 12), tr(-19, 12)]
        pg.draw.polygon(s, color, body)
        pg.draw.polygon(s, dark, body, 2)
        bx = cx + math.cos(t.turret) * C.MUZZLE_LENGTH
        by = cy + math.sin(t.turret) * C.MUZZLE_LENGTH
        pg.draw.line(s, dark, (cx, cy), (bx, by), 7)
        pg.draw.line(s, (200, 200, 200), (cx, cy), (bx, by), 3)
        pg.draw.circle(s, dark, (cx, cy), 10)
        pg.draw.circle(s, color, (cx, cy), 7)
        # health bar
        w = 40
        frac = max(0, t.hp) / C.TANK_HP
        pg.draw.rect(s, (60, 20, 20), (cx - w // 2, cy - 32, w, 5))
        pg.draw.rect(s, (90, 220, 110) if frac > 0.4 else (240, 180, 60), (cx - w // 2, cy - 32, int(w * frac), 5))
        if t.reload > 0:
            pg.draw.rect(s, (70, 70, 80), (cx - w // 2, cy - 25, int(w * t.reload / C.RELOAD_TIME), 2))

    def _text(self, txt, pos, font=None, color=TEXT, anchor="topleft"):
        img = (font or self.font).render(txt, True, color)
        rect = img.get_rect(**{anchor: pos})
        self.screen.blit(img, rect)
        return rect

    def _center_text(self, txt, font, y, color=TEXT):
        img = font.render(txt, True, color)
        self.screen.blit(img, img.get_rect(center=(WIN_W // 2, y)))

    def _draw_hud(self):
        pg = self.pg
        m = self.match
        for i, (name, anchor_x) in enumerate(((self.names[0], 14), (self.names[1], WIN_W - 14))):
            t = m.tanks[i]
            label = name
            if self.mode == "watch" or (self.mode == "play" and i == 1) or not self.human:
                label += f"  [{m.controllers[i].predictor.label}]"
            anchor = "topleft" if i == 0 else "topright"
            r = self._text(label, (anchor_x, 6), color=self.colors[i], anchor=anchor)
            px = r.left if i == 0 else r.right - C.TANK_HP * 16
            for k in range(C.TANK_HP):
                col = self.colors[i] if k < t.hp else (60, 63, 70)
                pg.draw.rect(self.screen, col, (px + k * 16, 32, 12, 12), border_radius=2)
            acc = f"accuracy {m.accuracy(i):.0%} ({m.hits[i]}/{m.shots[i]})"
            self._text(acc, (px + C.TANK_HP * 16 + 8, 30) if i == 0 else (px - 8, 30),
                       font=self.font_small, color=DIM, anchor="topleft" if i == 0 else "topright")
        self._center_text(f"{m.score[0]} : {m.score[1]}", self.font_big, 22)
        err_tracker = m.trackers[0] if self.mode == "play" else None
        info = [f"difficulty {self.difficulty}"] if self.mode == "play" else []
        if err_tracker is not None and err_tracker.mean_error is not None:
            info.append(f"enemy prediction error {err_tracker.mean_error:5.1f} px")
        if self.mode == "watch" or self.speed > 1:
            info.append(f"speed x{self.speed}")
        info.append("F1 help")
        self._center_text("   |   ".join(info), self.font_small, 47, DIM)

    def _draw_banner(self):
        m = self.match
        if m.winner is None:
            txt = "DRAW"
        elif self.mode == "play":
            txt = "ROUND WON!" if m.winner == 0 else "ROUND LOST"
        else:
            txt = f"{self.names[m.winner]} WINS THE ROUND"
        self._center_text(txt, self.font_big, HUD_H + C.ARENA_H // 2,
                          self.colors[m.winner] if m.winner is not None else TEXT)

    def _draw_help(self):
        pg = self.pg
        lines = ["CONTROLS",
                 "W / S  or arrows - forward / reverse",
                 "A / D  or arrows - rotate the hull",
                 "Mouse - aim the turret,  left click / SPACE - fire",
                 "1-6 - enemy aiming: static, const. velocity, const. turn, MLP, LSTM, GRU",
                 "V - show what the enemy predicts about you",
                 "T - aim assist: RNN prediction of the enemy",
                 "+/- simulation speed,  P pause,  R restart round,  F1 help,  ESC quit"]
        if self.mode == "watch":
            lines = ["WATCH MODE - two AI tanks, each aiming with a different predictor",
                     "V - show predictions,  +/- speed,  P pause,  R restart round,  ESC quit"]
        h = 30 + 26 * len(lines)
        box = pg.Rect(WIN_W // 2 - 330, HUD_H + 120, 660, h)
        surf = pg.Surface(box.size, pg.SRCALPHA)
        surf.fill((10, 12, 16, 215))
        self.screen.blit(surf, box.topleft)
        pg.draw.rect(self.screen, (90, 96, 108), box, 2, border_radius=6)
        for k, line in enumerate(lines):
            self._text(line, (box.left + 20, box.top + 15 + 26 * k),
                       color=TEXT if k else (255, 220, 120))

    # ------------------------------------------------------------------ main loop
    def run(self, max_frames: int | None = None, screenshot: str | None = None, fps: int = C.SIM_HZ) -> dict:
        pg = self.pg
        frames = 0
        running = True
        try:
            while running:
                running = self.handle_events()
                frame_dt = self.clock.tick(fps if max_frames is None else 0) / 1000.0
                self.update(frame_dt)
                self.draw()
                pg.display.flip()
                frames += 1
                if max_frames is not None and frames >= max_frames:
                    break
            if screenshot:
                Path(screenshot).parent.mkdir(parents=True, exist_ok=True)
                pg.image.save(self.screen, screenshot)
        finally:
            saved = self._save_recording()
            pg.quit()
        m = self.match
        return {"frames": frames, "score": list(m.score), "rounds": m.rounds, "shots": list(m.shots),
                "hits": list(m.hits), "recording": saved}

    def _save_recording(self):
        tr = self.match.trackers[0]
        if not self.recording or tr.recording is None:
            return None
        path = Path("data/recorded") / f"player_{time.strftime('%Y%m%d_%H%M%S')}.npz"
        n = save_recording(path, tr.recording)
        if n:
            print(f"recorded {n} samples of your driving -> {path}")
            return str(path)
        return None


def play(**kw) -> dict:
    max_frames = kw.pop("max_frames", None)
    screenshot = kw.pop("screenshot", None)
    return GameApp(**kw).run(max_frames=max_frames, screenshot=screenshot)
