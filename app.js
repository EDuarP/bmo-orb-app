const canvas      = document.getElementById('orb');
const ctx         = canvas.getContext('2d');
const stateBadge  = document.getElementById('state-badge');
const levelStatus = document.getElementById('level-status');
const scoreStatus = document.getElementById('score-status');
const bubblesEl   = document.getElementById('bubbles');
const transcriptEl = document.getElementById('transcript');

let w = canvas.width  = window.innerWidth;
let h = canvas.height = window.innerHeight;
let t = 0;
let audioLevel  = 0.08;
let targetLevel = 0.08;
let appState = 'listening';

window.addEventListener('resize', () => {
  w = canvas.width  = window.innerWidth;
  h = canvas.height = window.innerHeight;
});

// ── orb palette per state ──────────────────────────────────────────────────────
const PALETTE = {
  listening: {
    glow1: [120, 225, 255],
    glow2: [68,  122, 255],
    p1:    [175, 245, 255],
    p2:    [116, 193, 255],
    p3:    [80,  130, 255],
    ring:  [180, 235, 255],
  },
  recording: {
    glow1: [80,  255, 180],
    glow2: [20,  200, 120],
    p1:    [100, 255, 190],
    p2:    [40,  220, 140],
    p3:    [20,  160, 100],
    ring:  [80,  255, 180],
  },
  thinking: {
    glow1: [190, 110, 255],
    glow2: [110,  50, 220],
    p1:    [210, 140, 255],
    p2:    [160,  90, 240],
    p3:    [100,  50, 200],
    ring:  [190, 110, 255],
  },
};

function palette() {
  return PALETTE[appState] || PALETTE.listening;
}

function rgba([r, g, b], a) {
  return `rgba(${r},${g},${b},${a})`;
}

// ── bubble management ──────────────────────────────────────────────────────────
let botBubbleEl  = null;
let userBubbleEl = null;

function removeBubble(el, cb) {
  if (!el) { cb && cb(); return; }
  el.classList.add('removing');
  el.addEventListener('animationend', () => { el.remove(); cb && cb(); }, { once: true });
}

function showBubble(type, text) {
  const existing = type === 'bot' ? botBubbleEl : userBubbleEl;

  const create = () => {
    const el = document.createElement('div');
    el.className = `bubble ${type}`;
    el.textContent = text;
    bubblesEl.appendChild(el);
    if (type === 'bot')  botBubbleEl  = el;
    else                 userBubbleEl = el;
  };

  if (existing) {
    removeBubble(existing, create);
  } else {
    create();
  }
}

// ── WebSocket ──────────────────────────────────────────────────────────────────
function connectBackend() {
  const ws = new WebSocket(`ws://${window.location.host}/ws`);

  ws.onmessage = ({ data }) => {
    const payload = JSON.parse(data);

    switch (payload.type) {
      case 'state': {
        appState = payload.state;
        stateBadge.dataset.state = payload.state;
        const labels = {
          listening: 'Listening',
          recording: 'Recording…',
          thinking:  'Thinking…',
        };
        stateBadge.textContent = labels[payload.state] ?? payload.state;
        break;
      }
      case 'audio_level': {
        targetLevel = Math.max(0.06, Math.min(1, payload.level * 12));
        levelStatus.textContent = `Level: ${payload.level.toFixed(3)}`;
        scoreStatus.textContent = `Score: ${payload.score.toFixed(3)}`;
        break;
      }
      case 'user_message': {
        transcriptEl.textContent = `"${payload.text}"`;
        showBubble('user', payload.text);
        break;
      }
      case 'bot_message': {
        // Bot message persists until the next bot message replaces it.
        showBubble('bot', payload.text);
        break;
      }
    }
  };

  ws.onclose = () => {
    stateBadge.textContent = 'Reconnecting…';
    stateBadge.dataset.state = 'listening';
    setTimeout(connectBackend, 2000);
  };
}

// ── orb drawing ────────────────────────────────────────────────────────────────
function drawOrb() {
  ctx.clearRect(0, 0, w, h);

  const cx = w / 2;
  const cy = h / 2;
  const baseRadius = Math.min(w, h) * 0.11;
  audioLevel += (targetLevel - audioLevel) * 0.08;

  const pal = palette();

  // glow
  const glow = ctx.createRadialGradient(cx, cy, baseRadius * 0.1, cx, cy, baseRadius * 2.2);
  glow.addColorStop(0,   rgba(pal.glow1, 0.52));
  glow.addColorStop(0.4, rgba(pal.glow2, 0.18));
  glow.addColorStop(1,   'rgba(0,0,0,0)');
  ctx.fillStyle = glow;
  ctx.beginPath();
  ctx.arc(cx, cy, baseRadius * (1.8 + audioLevel * 0.5), 0, Math.PI * 2);
  ctx.fill();

  // particle cloud
  const particles = 220;
  for (let i = 0; i < particles; i++) {
    const angle      = (i / particles) * Math.PI * 2 + t * 0.0025;
    const band       = 0.55 + 0.45 * Math.sin(i * 12.9898);
    const innerNoise = Math.sin(angle * 4 + t * 0.02 + band) * (2.5 + audioLevel * 6);
    const radius     = baseRadius * band + innerNoise;
    const x = cx + Math.cos(angle) * radius;
    const y = cy + Math.sin(angle) * radius;
    const size  = 1 + (1 - band) * 1.8 + audioLevel * 0.8;
    const alpha = 0.14 + (1 - band) * 0.28 + audioLevel * 0.08;

    ctx.fillStyle = band < 0.35
      ? rgba(pal.p1, alpha)
      : band < 0.7
      ? rgba(pal.p2, alpha)
      : rgba(pal.p3, alpha * 0.8);

    ctx.beginPath();
    ctx.arc(x, y, size, 0, Math.PI * 2);
    ctx.fill();
  }

  // ring wave
  ctx.strokeStyle = rgba(pal.ring, 0.12 + audioLevel * 0.12);
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i <= 120; i++) {
    const angle  = (i / 120) * Math.PI * 2;
    const wave   = Math.sin(angle * 6 + t * 0.03) * (2 + audioLevel * 7);
    const radius = baseRadius * 1.01 + wave;
    const x = cx + Math.cos(angle) * radius;
    const y = cy + Math.sin(angle) * radius;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.closePath();
  ctx.stroke();
}

function animate() {
  t++;
  drawOrb();
  setTimeout(() => requestAnimationFrame(animate), 33);
}

connectBackend();
animate();
