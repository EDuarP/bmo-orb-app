const canvas = document.getElementById('orb');
const ctx = canvas.getContext('2d');
const micStatus = document.getElementById('mic-status');
const wakeStatus = document.getElementById('wake-status');
const heardStatus = document.getElementById('heard-status');
const levelStatus = document.getElementById('level-status');
const transcriptLog = document.getElementById('transcript-log');

let w = canvas.width = window.innerWidth;
let h = canvas.height = window.innerHeight;
let t = 0;
let audioLevel = 0.08;
let targetLevel = 0.08;

window.addEventListener('resize', () => {
  w = canvas.width = window.innerWidth;
  h = canvas.height = window.innerHeight;
});

function connectBackend() {
  const ws = new WebSocket(`ws://${window.location.host}/ws`);
  ws.onopen = () => {
    transcriptLog.textContent = 'Backend connected. Waiting for microphone data...';
  };
  ws.onmessage = (event) => {
    const payload = JSON.parse(event.data);
    if (payload.type === 'status') {
      micStatus.textContent = `Mic: ${payload.mic}`;
      wakeStatus.textContent = `Wake word: ${payload.wake}`;
    }
    if (payload.type === 'audio_level') {
      targetLevel = Math.max(0.06, Math.min(1, payload.level * 12));
      micStatus.textContent = `Mic: ${payload.device}`;
      heardStatus.textContent = `Heard: ${payload.heard}`;
      levelStatus.textContent = `Level: ${payload.level.toFixed(3)}`;
      transcriptLog.textContent = `Audio stream active from ${payload.device}. Wake word backend pending.`;
    }
  };
  ws.onclose = () => {
    micStatus.textContent = 'Mic: reconnecting';
    wakeStatus.textContent = 'Wake word: backend reconnecting';
    setTimeout(connectBackend, 2000);
  };
}

function drawOrb() {
  ctx.clearRect(0, 0, w, h);
  const cx = w / 2;
  const cy = h / 2;
  const baseRadius = Math.min(w, h) * 0.11;
  audioLevel += (targetLevel - audioLevel) * 0.08;

  const glow = ctx.createRadialGradient(cx, cy, baseRadius * 0.1, cx, cy, baseRadius * 2.2);
  glow.addColorStop(0, 'rgba(120, 225, 255, 0.52)');
  glow.addColorStop(0.4, 'rgba(68, 122, 255, 0.18)');
  glow.addColorStop(1, 'rgba(0, 0, 0, 0)');
  ctx.fillStyle = glow;
  ctx.beginPath();
  ctx.arc(cx, cy, baseRadius * (1.8 + audioLevel * 0.5), 0, Math.PI * 2);
  ctx.fill();

  const particles = 220;
  for (let i = 0; i < particles; i++) {
    const angle = (i / particles) * Math.PI * 2 + t * 0.0025;
    const band = 0.55 + 0.45 * Math.sin(i * 12.9898);
    const innerNoise = Math.sin(angle * 4 + t * 0.02 + band) * (2.5 + audioLevel * 6);
    const radius = baseRadius * band + innerNoise;
    const x = cx + Math.cos(angle) * radius;
    const y = cy + Math.sin(angle) * radius;
    const size = 1 + (1 - band) * 1.8 + audioLevel * 0.8;
    const alpha = 0.14 + (1 - band) * 0.28 + audioLevel * 0.08;
    ctx.fillStyle = band < 0.35
      ? `rgba(175, 245, 255, ${alpha})`
      : band < 0.7
      ? `rgba(116, 193, 255, ${alpha})`
      : `rgba(80, 130, 255, ${alpha * 0.8})`;
    ctx.beginPath();
    ctx.arc(x, y, size, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.strokeStyle = `rgba(180, 235, 255, ${0.12 + audioLevel * 0.12})`;
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i <= 120; i++) {
    const angle = (i / 120) * Math.PI * 2;
    const wave = Math.sin(angle * 6 + t * 0.03) * (2 + audioLevel * 7);
    const radius = baseRadius * 1.01 + wave;
    const x = cx + Math.cos(angle) * radius;
    const y = cy + Math.sin(angle) * radius;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
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