const { frameCount } = window.APP_CONFIG;
const form = document.querySelector('#upload-form');
const fileInput = document.querySelector('#video-file');
const fileName = document.querySelector('#file-name');
const cameraPreview = document.querySelector('#camera-preview');
const startCameraButton = document.querySelector('#start-camera');
const captureCameraButton = document.querySelector('#capture-camera');
const stopCameraButton = document.querySelector('#stop-camera');
const cameraStatus = document.querySelector('#camera-status');
const result = document.querySelector('#result');
const resultSummary = document.querySelector('#result-summary');
const predictions = document.querySelector('#predictions');
const errorBox = document.querySelector('#error');

let cameraStream = null;
let isCapturing = false;

function setError(message = '') {
  errorBox.textContent = message;
  errorBox.classList.toggle('hidden', !message);
}

function setBusy(button, busy, label) {
  button.disabled = busy;
  if (busy) {
    button.dataset.label = button.textContent;
    button.textContent = label;
  } else if (button.dataset.label) {
    button.textContent = button.dataset.label;
  }
}

function showPredictions(items) {
  const best = items[0];
  resultSummary.textContent = `Najbardziej prawdopodobna czynność: ${best.class_name} (${(best.probability * 100).toFixed(1)}%).`;
  predictions.replaceChildren(...items.map((item) => {
    const element = document.createElement('li');
    element.textContent = `${item.class_name} — ${(item.probability * 100).toFixed(1)}%`;
    return element;
  }));
  result.classList.remove('hidden');
}

async function readResponse(response) {
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'Nie udało się wykonać klasyfikacji.');
  return data;
}

fileInput.addEventListener('change', () => {
  fileName.textContent = fileInput.files[0] ? `Wybrano: ${fileInput.files[0].name}` : 'Obsługiwane formaty: AVI, MP4, MOV, MKV, WebM. Maksymalnie 250 MB.';
});

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  if (!fileInput.files[0]) return;
  setError();
  const submitButton = form.querySelector('button');
  setBusy(submitButton, true, 'Analizuję film…');
  try {
    const formData = new FormData();
    formData.append('video', fileInput.files[0]);
    const data = await readResponse(await fetch('/api/predict-video', { method: 'POST', body: formData }));
    showPredictions(data.predictions);
  } catch (error) {
    setError(error.message);
  } finally {
    setBusy(submitButton, false);
  }
});

startCameraButton.addEventListener('click', async () => {
  setError();
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    cameraPreview.srcObject = cameraStream;
    captureCameraButton.disabled = false;
    stopCameraButton.disabled = false;
    startCameraButton.disabled = true;
    cameraStatus.textContent = 'Kamera jest gotowa.';
  } catch (error) {
    setError(`Nie można uruchomić kamery: ${error.message}`);
  }
});

stopCameraButton.addEventListener('click', () => {
  cameraStream?.getTracks().forEach((track) => track.stop());
  cameraStream = null;
  cameraPreview.srcObject = null;
  captureCameraButton.disabled = true;
  stopCameraButton.disabled = true;
  startCameraButton.disabled = false;
  cameraStatus.textContent = 'Kamera została wyłączona.';
});

captureCameraButton.addEventListener('click', async () => {
  if (!cameraStream || isCapturing) return;
  setError();
  isCapturing = true;
  captureCameraButton.disabled = true;
  const canvas = document.createElement('canvas');
  canvas.width = cameraPreview.videoWidth || 320;
  canvas.height = cameraPreview.videoHeight || 240;
  const context = canvas.getContext('2d');
  const capturedFrames = [];

  try {
    for (let index = 0; index < frameCount; index += 1) {
      context.drawImage(cameraPreview, 0, 0, canvas.width, canvas.height);
      capturedFrames.push(canvas.toDataURL('image/jpeg', 0.8));
      cameraStatus.textContent = `Nagrywanie: ${index + 1} z ${frameCount} klatek…`;
      await new Promise((resolve) => setTimeout(resolve, 250));
    }
    cameraStatus.textContent = 'Analizuję materiał z kamery…';
    const data = await readResponse(await fetch('/api/predict-camera', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ frames: capturedFrames }),
    }));
    showPredictions(data.predictions);
    cameraStatus.textContent = 'Gotowe. Możesz nagrać kolejną próbkę.';
  } catch (error) {
    setError(error.message);
    cameraStatus.textContent = 'Nagranie nie zostało przetworzone.';
  } finally {
    isCapturing = false;
    captureCameraButton.disabled = !cameraStream;
  }
});
