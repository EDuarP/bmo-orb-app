const canvas = document.getElementById('orb');
const ctx = canvas.getContext('2d');
const micStatus = document.getElementById('mic-status');

let w = canvas.width = window.innerWidth;
let h = canvas.height = window.innerHeight;
let t = 0;
let audioLevel = 0.12;
let targetLevel = 0.12;
let analyser;
let dataArray;

window.addEventListener('resize', () => {
  w = canvas.width = window.innerWidth;
  h = canvas.height = window.innerHeight;
});

async function setupMic() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 256;
    dataArray = new Uint8Array(analyser.frequencyBinCount);
    const source = audioContext.createMediaStreamSource(stream);
    source.connect(analyser);
    micStatus.textContent = 'Mic: live';
  } catch (error) {
    micStatus.textContent = 'Mic: permission needed';
    console.error(error);
  }
}

function sampleAudio() {
  if (!analyser) {
    targetLevel = 0.14 + Math.sin(t * 0.01) * 0.02;
    return;
  }
  analyser.getByteFrequencyData(dataArray);
  let sum = 0;
  for (let i = 0; i < dataArray.length; i++) sum += dataArray[i];
  const avg = sum / dataArray.length / 255;
  targetLevel = Math.max(0.08, Math.min(1, avg * 2.8));
}

function drawOrb() {
  ctx.clearRect(0, 0, w, h);
  const cx = w / 2;
  const cy = h / 2;
  const baseRadius = Math.min(w, h) * 0.17;
  audioLevel += (targetLevel - audioLevel) * 0.08;

  const glow = ctx.createRadialGradient(cx, cy, baseRadius * 0.1, cx, cy, baseRadius * 1.8);
  glow.addColorStop(0, 'rgba(120, 225, 255, 0.45)');
  glow.addColorStop(0.35, 'rgba(68, 122, 255, 0.20)');
  glow.addColorStop(1, 'rgba(0, 0, 0, 0)');
  ctx.fillStyle = glow;
  ctx.beginPath();
  ctx.arc(cx, cy, baseRadius * (1.5 + audioLevel * 0.6), 0, Math.PI * 2);
  ctx.fill();

  for (let ring = 0; ring < 3; ring++) {
    const particles = 220 + ring * 60;
    const ringRadius = baseRadius + ring * 28 + audioLevel * 24;
    for (let i = 0; i < particles; i++) {
      const angle = (i / particles) * Math.PI * 2 + t * 0.003 * (ring % 2 === 0 ? 1 : -1);
      const noise = Math.sin(angle * 6 + t * 0.02 + ring) * (8 + audioLevel * 26);
      const radius = ringRadius + noise;
      const x = cx + Math.cos(angle) * radius;
      const y = cy + Math.sin(angle) * radius;
      const size = 1.2 + ring * 0.4 + audioLevel * 2.2;
      ctx.fillStyle = ring === 0
        ? `rgba(136, 240, 255, ${0.55 + audioLevel * 0.25})`
        : ring === 1
        ? `rgba(84, 135, 255, ${0.34 + audioLevel * 0.22})`
        : `rgba(255,255,255, ${0.10 + audioLevel * 0.12})`;
      ctx.beginPath();
      ctx.arc(x, y, size, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  ctx.strokeStyle = `rgba(160, 230, 255, ${0.2 + audioLevel * 0.2})`;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let i = 0; i <= 240; i++) {
    const angle = (i / 240) * Math.PI * 2;
    const wave = Math.sin(angle * 10 + t * 0.04) * (6 + audioLevel * 20);
    const radius = baseRadius * 0.84 + wave;
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
  sampleAudio();
  drawOrb();
  requestAnimationFrame(animate);
}

setupMic();
animate();