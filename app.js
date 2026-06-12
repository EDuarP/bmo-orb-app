// ── BMO Orb — breathing particle sphere ──────────────────────────────────────
// Black background. Particles hold their position on the sphere (no rotation)
// and oscillate radially — drifting toward and away from the center — so the
// orb "breathes" without losing its spherical shape.
// Colors: idle = white, listening = light green, speaking = light blue.
const canvas = document.getElementById('orb');
const ctx = canvas.getContext('2d');

let DPR = 1, w = 0, h = 0;
function resize() {
  DPR = Math.min(window.devicePixelRatio || 1, 2);
  w = canvas.width = Math.floor(window.innerWidth * DPR);
  h = canvas.height = Math.floor(window.innerHeight * DPR);
}
resize();
window.addEventListener('resize', resize);

// ── reactive state (driven by the backend over WebSocket) ────────────────────
let appState = 'idle';                 // idle | recording | thinking | speaking
let audioLevel = 0.05;
let targetLevel = 0.05;

// Per-state mood: color, brightness, oscillation speed/amplitude, halo glow.
const MOOD = {
  idle:      { color: [255, 255, 255], bright: 0.50, speed: 0.55, osc: 0.07, glow: 0.10 },
  recording: { color: [140, 255, 185], bright: 1.00, speed: 1.00, osc: 0.13, glow: 0.24 },
  thinking:  { color: [235, 235, 245], bright: 0.75, speed: 1.25, osc: 0.09, glow: 0.16 },
  speaking:  { color: [140, 200, 255], bright: 0.95, speed: 1.10, osc: 0.12, glow: 0.22 },
};
const mood = { color: [255, 255, 255], bright: 0.5, speed: 0.55, osc: 0.07, glow: 0.10 };
function lerp(a, b, t) { return a + (b - a) * t; }

// ── tintable soft sprite (regenerated when the mood color shifts) ────────────
const SPRITE_SIZE = 64;
const sprite = document.createElement('canvas');
sprite.width = sprite.height = SPRITE_SIZE;
const spriteCtx = sprite.getContext('2d');
let spriteColor = [-1, -1, -1];
function renderSprite([r, g, b]) {
  const half = SPRITE_SIZE / 2;
  spriteCtx.clearRect(0, 0, SPRITE_SIZE, SPRITE_SIZE);
  const grad = spriteCtx.createRadialGradient(half, half, 0, half, half, half);
  grad.addColorStop(0.0, `rgba(${r | 0},${g | 0},${b | 0},1)`);
  grad.addColorStop(0.25, `rgba(${r | 0},${g | 0},${b | 0},0.55)`);
  grad.addColorStop(0.6, `rgba(${r | 0},${g | 0},${b | 0},0.12)`);
  grad.addColorStop(1.0, `rgba(${r | 0},${g | 0},${b | 0},0)`);
  spriteCtx.fillStyle = grad;
  spriteCtx.fillRect(0, 0, SPRITE_SIZE, SPRITE_SIZE);
  spriteColor = [r | 0, g | 0, b | 0];
}
renderSprite(mood.color);

// ── particles: Fibonacci lattice, each with its own oscillation rhythm ───────
const N = 1500;
const GOLDEN = Math.PI * (3 - Math.sqrt(5));
const P = new Array(N);
for (let i = 0; i < N; i++) {
  const y = 1 - (i / (N - 1)) * 2;
  const r = Math.sqrt(Math.max(0, 1 - y * y));
  const th = i * GOLDEN;
  P[i] = {
    x: Math.cos(th) * r, y, z: Math.sin(th) * r,
    phase: (i * 0.6180339887 * Math.PI * 2) % (Math.PI * 2), // golden-ratio scatter
    freq: 0.7 + ((i * 7919) % 1000) / 1000 * 0.8,            // 0.7 … 1.5 ×
  };
}

// Directional light (normalized): up-left, toward the camera.
const LX = -0.37, LY = -0.56, LZ = 0.74;

let oscT = 0, pulseT = 0, lastTs = 0;

// ── render loop ──────────────────────────────────────────────────────────────
function draw(ts) {
  const dt = Math.min(0.05, (ts - lastTs) / 1000 || 0.016);
  lastTs = ts;

  audioLevel += (targetLevel - audioLevel) * 0.10;

  const target = MOOD[appState] || MOOD.idle;
  for (let i = 0; i < 3; i++) mood.color[i] = lerp(mood.color[i], target.color[i], 0.05);
  mood.bright = lerp(mood.bright, target.bright, 0.04);
  mood.speed = lerp(mood.speed, target.speed, 0.04);
  mood.osc = lerp(mood.osc, target.osc, 0.04);
  mood.glow = lerp(mood.glow, target.glow, 0.04);

  // Re-tint the sprite only when the crossfading color actually moved.
  const [cr, cg, cb] = mood.color;
  if (Math.abs(cr - spriteColor[0]) + Math.abs(cg - spriteColor[1]) + Math.abs(cb - spriteColor[2]) > 2) {
    renderSprite(mood.color);
  }
  const tint = `${cr | 0},${cg | 0},${cb | 0}`;

  // Solid black clear.
  ctx.globalCompositeOperation = 'source-over';
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, w, h);

  const cx = w / 2, cy = h / 2;
  // Thinking pulse: barely-there slow breathing of the whole orb.
  const pulse = appState === 'thinking' ? 0.012 * Math.sin(pulseT * 1.8) : 0;
  const R = Math.min(w, h) * (0.27 + audioLevel * 0.05 + pulse);
  const persp = 2.5;

  oscT += dt * (1.2 * mood.speed + audioLevel * 1.6);
  pulseT += dt * mood.speed * 2.0;

  // Radial oscillation amplitude: mood + a touch of audio reactivity.
  const amp = mood.osc + audioLevel * 0.10;

  // ── ambient halo behind the orb ──
  const halo = ctx.createRadialGradient(cx, cy, R * 0.2, cx, cy, R * 2.6);
  halo.addColorStop(0, `rgba(${tint},${(mood.glow + audioLevel * 0.16).toFixed(3)})`);
  halo.addColorStop(0.5, `rgba(${tint},0.03)`);
  halo.addColorStop(1, 'rgba(0,0,0,0)');
  ctx.fillStyle = halo;
  ctx.fillRect(0, 0, w, h);

  // ── project (no rotation — each particle breathes radially in place) ──
  const proj = [];
  for (let i = 0; i < N; i++) {
    const p = P[i];
    // in/out drift around its own spot on the sphere
    const rr = 1 + amp * Math.sin(p.phase + oscT * p.freq);
    const x = p.x * rr, y0 = p.y * rr, z0 = p.z * rr;
    const lambert = Math.max(0, p.x * LX + p.y * LY + p.z * LZ);
    proj.push([x, y0, z0, lambert, p.phase]);
  }
  proj.sort((a, b) => a[2] - b[2]); // far → near

  // ── draw particles (additive glow) ──
  ctx.globalCompositeOperation = 'lighter';
  for (let i = 0; i < proj.length; i++) {
    const [x1, y1, z2, lambert, phase] = proj[i];
    const scale = persp / (persp - z2);
    const sx = cx + x1 * R * scale;
    const sy = cy + y1 * R * scale;
    const depth = (z2 + 1.2) / 2.4;                  // 0 back … 1 front
    const twinkle = 0.88 + 0.12 * Math.sin(phase + oscT * 2.2);
    const size = (1.1 + depth * 3.0 + lambert * 1.5) * DPR
               * (1 + audioLevel * 0.55) * twinkle;
    const alpha = mood.bright * twinkle
                * (0.05 + depth * 0.30 + lambert * 0.45)
                * (0.75 + audioLevel * 0.5);
    if (alpha < 0.01) continue;
    ctx.globalAlpha = Math.min(1, alpha);
    ctx.drawImage(sprite, sx - size, sy - size, size * 2, size * 2);
  }
  ctx.globalAlpha = 1;

  // ── soft inner core ──
  const core = ctx.createRadialGradient(cx, cy, 0, cx, cy, R * (0.45 + audioLevel * 0.25));
  core.addColorStop(0, `rgba(${tint},${(0.14 * mood.bright + audioLevel * 0.2).toFixed(3)})`);
  core.addColorStop(0.6, `rgba(${tint},0.03)`);
  core.addColorStop(1, 'rgba(0,0,0,0)');
  ctx.fillStyle = core;
  ctx.beginPath();
  ctx.arc(cx, cy, R * (0.55 + audioLevel * 0.25), 0, Math.PI * 2);
  ctx.fill();

  ctx.globalCompositeOperation = 'source-over';
}

function loop(ts) { draw(ts); requestAnimationFrame(loop); }

// ── push-to-talk: Alt+\ toggles listening (also bound globally in Hyprland) ──
window.addEventListener('keydown', (e) => {
  if (e.altKey && (e.code === 'Backslash' || e.key === '\\')) {
    e.preventDefault();
    fetch('/trigger', { method: 'POST' }).catch(() => {});
  }
});

// ── WebSocket: state + audio-level pulse (no text rendered) ──────────────────
function connectBackend() {
  const ws = new WebSocket(`ws://${window.location.host}/ws`);
  ws.onmessage = ({ data }) => {
    let payload;
    try { payload = JSON.parse(data); } catch { return; }
    if (payload.type === 'state') {
      appState = payload.state;
    } else if (payload.type === 'audio_level') {
      targetLevel = Math.max(0.04, Math.min(1, payload.level * 12));
    }
    // user_message / bot_message intentionally ignored — voice-only UI.
  };
  ws.onclose = () => setTimeout(connectBackend, 2000);
}

connectBackend();
requestAnimationFrame(loop);
