"use strict";

const canvas = document.querySelector("#drawing-canvas");
const context = canvas.getContext("2d", { willReadFrequently: true });
const placeholder = document.querySelector("#canvas-placeholder");
const clearButton = document.querySelector("#clear-button");
const recognizeButton = document.querySelector("#recognize-button");
const resultEmpty = document.querySelector("#result-empty");
const resultReady = document.querySelector("#result-ready");
const predictionDigit = document.querySelector("#prediction-digit");
const predictionCaption = document.querySelector("#prediction-caption");
const matchValue = document.querySelector("#match-value");
const matchBar = document.querySelector("#match-bar");
const alternativesList = document.querySelector("#alternatives-list");

const CANVAS_SIZE = 560;
const MODEL_SIZE = 28;
const CLASSIFIER_SIZE = 8;
const INK_THRESHOLD = 218;
let isDrawing = false;
let hasInk = false;
let activePointerId = null;

context.lineCap = "round";
context.lineJoin = "round";
context.strokeStyle = "#15171b";
context.lineWidth = 27;

function resetResult() {
  resultEmpty.hidden = false;
  resultReady.hidden = true;
  predictionDigit.textContent = "—";
  predictionCaption.textContent = "Draw a digit to begin";
  matchValue.textContent = "—";
  matchBar.style.width = "0";
  alternativesList.replaceChildren();
}

function clearCanvas() {
  context.clearRect(0, 0, canvas.width, canvas.height);
  hasInk = false;
  recognizeButton.disabled = true;
  placeholder.classList.remove("is-hidden");
  resetResult();
}

function getPoint(event) {
  const bounds = canvas.getBoundingClientRect();
  return {
    x: ((event.clientX - bounds.left) / bounds.width) * canvas.width,
    y: ((event.clientY - bounds.top) / bounds.height) * canvas.height,
  };
}

function startDrawing(event) {
  if (!event.isPrimary || (event.button !== undefined && event.button !== 0)) return;
  event.preventDefault();
  isDrawing = true;
  activePointerId = event.pointerId;
  const point = getPoint(event);
  context.beginPath();
  context.moveTo(point.x, point.y);
  context.lineTo(point.x + 0.1, point.y + 0.1);
  context.stroke();
  hasInk = true;
  recognizeButton.disabled = false;
  placeholder.classList.add("is-hidden");
  resetResult();
}

function draw(event) {
  if (!isDrawing || event.pointerId !== activePointerId) return;
  event.preventDefault();
  const point = getPoint(event);
  context.lineTo(point.x, point.y);
  context.stroke();
}

function stopDrawing(event) {
  if (!isDrawing || event.pointerId !== activePointerId) return;
  isDrawing = false;
  activePointerId = null;
  context.closePath();
}

function normalizeDrawing() {
  const source = context.getImageData(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  let minX = CANVAS_SIZE;
  let minY = CANVAS_SIZE;
  let maxX = -1;
  let maxY = -1;
  let massX = 0;
  let massY = 0;
  let mass = 0;

  for (let y = 0; y < CANVAS_SIZE; y += 1) {
    for (let x = 0; x < CANVAS_SIZE; x += 1) {
      const offset = (y * CANVAS_SIZE + x) * 4;
      const darkness = 255 - source.data[offset];
      if (source.data[offset + 3] < 20 || darkness < (255 - INK_THRESHOLD)) continue;
      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      maxX = Math.max(maxX, x);
      maxY = Math.max(maxY, y);
      massX += x * darkness;
      massY += y * darkness;
      mass += darkness;
    }
  }

  if (maxX < minX || mass === 0) return null;

  const boxSize = Math.max(maxX - minX + 1, maxY - minY + 1);
  const pad = 3;
  const innerSize = MODEL_SIZE - pad * 2;
  const scale = innerSize / boxSize;
  const targetCanvas = document.createElement("canvas");
  targetCanvas.width = MODEL_SIZE;
  targetCanvas.height = MODEL_SIZE;
  const targetContext = targetCanvas.getContext("2d", { willReadFrequently: true });
  const sourceSide = boxSize;
  const sourceLeft = (minX + maxX + 1 - sourceSide) / 2;
  const sourceTop = (minY + maxY + 1 - sourceSide) / 2;
  const scaledWidth = sourceSide * scale;
  const scaledHeight = sourceSide * scale;
  const targetCenter = (MODEL_SIZE - 1) / 2;
  const centerX = ((massX / mass - sourceLeft) * scale) + pad;
  const centerY = ((massY / mass - sourceTop) * scale) + pad;
  const left = targetCenter - centerX;
  const top = targetCenter - centerY;

  targetContext.fillStyle = "#fff";
  targetContext.fillRect(0, 0, MODEL_SIZE, MODEL_SIZE);
  targetContext.drawImage(
    canvas,
    sourceLeft,
    sourceTop,
    sourceSide,
    sourceSide,
    left,
    top,
    scaledWidth,
    scaledHeight,
  );

  const classifierCanvas = document.createElement("canvas");
  classifierCanvas.width = CLASSIFIER_SIZE;
  classifierCanvas.height = CLASSIFIER_SIZE;
  const classifierContext = classifierCanvas.getContext("2d", { willReadFrequently: true });
  classifierContext.drawImage(targetCanvas, 0, 0, CLASSIFIER_SIZE, CLASSIFIER_SIZE);

  const pixels = classifierContext.getImageData(0, 0, CLASSIFIER_SIZE, CLASSIFIER_SIZE);
  return Array.from({ length: CLASSIFIER_SIZE * CLASSIFIER_SIZE }, (_, index) => {
    const offset = index * 4;
    return (255 - pixels.data[offset]) / 255;
  });
}

function classifyDigit(input) {
  let activations = input;

  for (let layer = 0; layer < window.DIGIT_MODEL.weights.length; layer += 1) {
    const weights = window.DIGIT_MODEL.weights[layer];
    const biases = window.DIGIT_MODEL.biases[layer];
    const output = Array(biases.length);

    for (let neuron = 0; neuron < biases.length; neuron += 1) {
      let total = biases[neuron];
      for (let index = 0; index < activations.length; index += 1) {
        total += activations[index] * weights[index][neuron];
      }
      output[neuron] = layer === 0 ? Math.max(0, total) : total;
    }
    activations = output;
  }

  const maxLogit = Math.max(...activations);
  const probabilities = activations.map((value) => Math.exp(value - maxLogit));
  const probabilityTotal = probabilities.reduce((total, value) => total + value, 0);

  return probabilities
    .map((probability, digit) => ({ digit, probability: probability / probabilityTotal }))
    .sort((first, second) => second.probability - first.probability);
}

function showResult(ranked) {
  const winner = ranked[0];
  const confidence = Math.round(winner.probability * 100);
  const separation = winner.probability - ranked[1].probability;
  resultEmpty.hidden = true;
  resultReady.hidden = false;
  predictionDigit.textContent = String(winner.digit);
  predictionCaption.textContent =
    confidence < 45
      ? "Not sure — try drawing it more clearly"
      : separation < 0.1
        ? "A close call — try drawing it a little larger"
        : "That’s my best guess";
  matchValue.textContent = `${confidence}%`;
  requestAnimationFrame(() => {
    matchBar.style.width = `${confidence}%`;
  });
  alternativesList.replaceChildren();

  for (const candidate of ranked.slice(1, 4)) {
    const row = document.createElement("div");
    row.className = "alternative-row";

    const digit = document.createElement("span");
    digit.className = "alternative-digit";
    digit.textContent = String(candidate.digit);

    const meter = document.createElement("span");
    meter.className = "alternative-meter";
    meter.setAttribute("aria-hidden", "true");
    const meterFill = document.createElement("span");
    const candidateScore = Math.round(candidate.probability * 100);
    meterFill.style.width = `${candidateScore}%`;
    meter.append(meterFill);

    const score = document.createElement("span");
    score.className = "alternative-score";
    score.textContent = `${candidateScore}%`;

    row.append(digit, meter, score);
    alternativesList.append(row);
  }
}

function recognizeDigit() {
  if (!hasInk) return;
  const input = normalizeDrawing();
  if (!input) {
    clearCanvas();
    return;
  }
  showResult(classifyDigit(input));
}

canvas.addEventListener("pointerdown", startDrawing);
window.addEventListener("pointermove", draw);
window.addEventListener("pointerup", stopDrawing);
window.addEventListener("pointercancel", stopDrawing);
clearButton.addEventListener("click", clearCanvas);
recognizeButton.addEventListener("click", recognizeDigit);
