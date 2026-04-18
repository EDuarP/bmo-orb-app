const canvas = document.getElementById('orb');
const ctx = canvas.getContext('2d');
const micStatus = document.getElementById('mic-status');
const wakeStatus = document.getElementById('wake-status');
const heardStatus = document.getElementById('heard-status');

let w = canvas.width = window.innerWidth;
let h = canvas.height = window.innerHeight;
let t = 0;
let audioLevel = 0.12;
let targetLevel = 0.12;
let analyser;
let dataArray;
let wakeCooldown;

window.addEventListener('resize', () => {
  w = canvas.width = window.innerWidth;
  h = canvas.height = window.innerHeight;
});

function setupWakeWord() {
  const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!Recognition) {
    wakeStatus.textContent = 'Wake word: browser STT unavailable';
    return;
  }
  const recognition = new Recognition();
  recognition.lang = 'es-CO';
  recognition.continuous = true;
  recognition.interimResults = true;

  recognition.onstart = () => {
    wakeStatus.textContent = 'Wake word: listening';
  };

  recognition.onresult = (event) => {
    let transcript = '';
    for (let i = event.resultIndex; i < event.results.length; i++) {
      transcript += event.results[i][0].transcript + ' ';
    }
    transcript = transcript.trim();
    if (transcript) heardStatus.textContent = `Heard: ${transcript}`;
    if (/ey\s*bmo/i.test(transcript)) {
      wakeStatus.textContent = 'Wake word: detected';
      targetLevel = 0.95;
      clearTimeout(wakeCooldown);
      wakeCooldown = setTimeout(() => {
        wakeStatus.textContent = 'Wake word: listening';
      }, 5000);
    }
  };

  recognition.onerror = () => {
    wakeStatus.textContent = 'Wake word: restarting';
  };

  recognition.onend = () => {
    setTimeout(() => recognition.start(), 1200);
  };

  recognition.start();
}

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

  const particles = 520;
  for (let i = 0; i < particles; i++) {
    const angle = (i / particles) * Math.PI * 2 + t * 0.0025;
    const band = 0.55 + 0.45 * Math.sin(i * 12.9898);
    const innerNoise = Math.sin(angle * 5 + t * 0.03 + band) * (4 + audioLevel * 10);
    const radius = baseRadius * band + innerNoise;
    const x = cx + Math.cos(angle) * radius;
    const y = cy + Math.sin(angle) * radius;
    const size = 1.4 + (1 - band) * 2.8 + audioLevel * 1.4;
    const alpha = 0.18 + (1 - band) * 0.42 + audioLevel * 0.12;
    ctx.fillStyle = band < 0.35
      ? `rgba(175, 245, 255, ${alpha})`
      : band < 0.7
      ? `rgba(116, 193, 255, ${alpha})`
      : `rgba(80, 130, 255, ${alpha * 0.8})`;
    ctx.beginPath();
    ctx.arc(x, y, size, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.strokeStyle = `rgba(180, 235, 255, ${0.18 + audioLevel * 0.2})`;
  ctx.lineWidth = 1.25;
  ctx.beginPath();
  for (let i = 0; i <= 180; i++) {
    const angle = (i / 180) * Math.PI * 2;
    const wave = Math.sin(angle * 8 + t * 0.04) * (3 + audioLevel * 12);
    const radius = baseRadius * 1.02 + wave;
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
setupWakeWord();
animate();