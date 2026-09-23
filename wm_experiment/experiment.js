const TEST_PARAMS = new URLSearchParams(location.search);
const SMOKE_TEST = TEST_PARAMS.has("smoke");
const PREVIEW_TEST = TEST_PARAMS.has("preview");
const CONFIG = Object.freeze({
  trialCount: 18,
  preTrialMs: SMOKE_TEST ? 10 : 900,
  delaysMs: SMOKE_TEST ? [20, 30, 40] : [1000, 3000, 5000],
  objectCount: 6,
});

const COLORS = [
  { name: "coral", hex: "#ff665c" },
  { name: "cyan", hex: "#38d7eb" },
  { name: "lime", hex: "#c4f04e" },
  { name: "violet", hex: "#a985ff" },
  { name: "amber", hex: "#ffb83e" },
  { name: "pink", hex: "#ff72bd" },
];
const SHAPES = ["circle", "square", "triangle", "diamond", "hexagon", "capsule"];
const TYPE_LABELS = {
  size_compare: "Relative size",
  closest_pair: "Closest pair",
  farthest_pair: "Farthest pair",
  highest: "Highest position",
  leftmost: "Leftmost position",
  nearest_fixation: "Nearest fixation",
};

const $ = (selector) => document.querySelector(selector);
const els = {
  screens: [...document.querySelectorAll(".screen")], welcome: $("#welcome"), trial: $("#trial"), results: $("#results"),
  form: $("#setup-form"), participant: $("#participant"), canvas: $("#scene-canvas"), stage: $("#stage"),
  fixation: $("#fixation"), phaseMessage: $("#phase-message"), questionPanel: $("#question-panel"),
  questionText: $("#question-text"), answers: $("#answer-options"), trialLabel: $("#trial-label"),
  progress: $("#progress-fill"), phaseLabel: $("#phase-label"), quit: $("#quit-button"), dialog: $("#end-dialog"),
};

let session = null;
let rng = Math.random;
let timerIds = new Set();
let currentTrial = null;
let questionShownAt = 0;
let acceptingAnswer = false;
let phase = "welcome";

function makeSeed() {
  if (globalThis.crypto?.getRandomValues) return crypto.getRandomValues(new Uint32Array(1))[0];
  return (Date.now() ^ Math.floor(Math.random() * 0xffffffff)) >>> 0;
}

function mulberry32(seed) {
  return function random() {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function shuffle(items) {
  const copy = [...items];
  for (let i = copy.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy;
}

function later(fn, ms) {
  const id = window.setTimeout(() => { timerIds.delete(id); fn(); }, ms);
  timerIds.add(id);
  return id;
}

function clearTimers() {
  timerIds.forEach(clearTimeout);
  timerIds.clear();
}

function showScreen(target) {
  els.screens.forEach((screen) => screen.classList.toggle("active", screen === target));
}

function createConditions() {
  const types = Object.keys(TYPE_LABELS);
  return shuffle(CONFIG.delaysMs.flatMap((delayMs) => shuffle(types).map((questionType) => ({ delayMs, questionType }))));
}

function createCandidateObjects() {
  const colors = shuffle(COLORS);
  const shapes = shuffle(SHAPES);
  // A broad, fixed range prevents visually ambiguous size judgments.
  const sizes = shuffle([0.027, 0.033, 0.039, 0.046, 0.053, 0.061]);
  const objects = [];
  for (let i = 0; i < CONFIG.objectCount; i++) {
    let x, y, attempts = 0;
    do {
      x = 0.10 + rng() * 0.80;
      y = 0.12 + rng() * 0.76;
      attempts++;
    } while (attempts < 100 && (
      (Math.abs(x - .5) < .24 && Math.abs(y - .5) < .16)
      || objects.some((o) => Math.hypot(x - o.x, y - o.y) < .17)
    ));
    objects.push({
      id: colors[i].name,
      color: colors[i].hex,
      shape: shapes[i],
      x, y,
      size: sizes[i],
      angle: rng() * Math.PI * 2,
    });
  }
  return objects;
}

function evaluateSalience(type, objects) {
  if (type === "size_compare") {
    const sorted = [...objects].sort((a, b) => a.size - b.size);
    return {
      correctId: sorted.at(-1).id,
      comparisonIds: [sorted[0].id, sorted.at(-1).id],
      margin: sorted.at(-1).size - sorted[0].size,
      threshold: 0.025,
    };
  }
  if (type === "closest_pair" || type === "farthest_pair") {
    const pairs = pairDistances(objects);
    const farthest = type === "farthest_pair";
    const target = farthest ? pairs.at(-1) : pairs[0];
    const runnerUp = farthest ? pairs.at(-2) : pairs[1];
    const margin = farthest ? target.value - runnerUp.value : runnerUp.value - target.value;
    return {
      correctId: target.ids.slice().sort().join("+"),
      correctPair: target.ids,
      margin,
      threshold: 0.085,
    };
  }
  const score = type === "highest"
    ? (object) => object.y
    : type === "leftmost"
      ? (object) => object.x
      : (object) => Math.hypot(object.x - .5, object.y - .5);
  const sorted = [...objects].sort((a, b) => score(a) - score(b));
  return {
    correctId: sorted[0].id,
    margin: score(sorted[1]) - score(sorted[0]),
    threshold: type === "nearest_fixation" ? 0.11 : 0.13,
  };
}

function createScene(questionType) {
  // Rejection sampling keeps scenes varied while enforcing an objective
  // separation between the correct answer and its nearest competitor.
  for (let attempt = 1; attempt <= 10000; attempt++) {
    const objects = createCandidateObjects();
    const salience = evaluateSalience(questionType, objects);
    if (salience.margin >= salience.threshold) {
      return { objects, salience: { ...salience, attempts: attempt } };
    }
  }
  throw new Error(`Unable to generate a sufficiently distinct ${questionType} scene.`);
}

function pairDistances(objects) {
  const pairs = [];
  for (let i = 0; i < objects.length; i++) for (let j = i + 1; j < objects.length; j++) {
    pairs.push({ ids: [objects[i].id, objects[j].id], value: Math.hypot(objects[i].x - objects[j].x, objects[i].y - objects[j].y) });
  }
  return pairs.sort((a, b) => a.value - b.value);
}

function objectOption(object) {
  return { id: object.id, label: `${capitalize(object.id)} ${object.shape}`, colors: [object.color] };
}

function pairOption(pair, objects) {
  const [a, b] = pair.ids.map((id) => objects.find((o) => o.id === id));
  return { id: pair.ids.slice().sort().join("+"), label: `${capitalize(a.id)} & ${capitalize(b.id)}`, colors: [a.color, b.color] };
}

function buildQuestion(type, objects, salience) {
  let prompt, correct, options;
  if (type === "size_compare") {
    const [a, b] = salience.comparisonIds.map((id) => objects.find((object) => object.id === id));
    prompt = `Which was bigger: the ${a.id} shape or the ${b.id} shape?`;
    correct = objects.find((object) => object.id === salience.correctId);
    options = shuffle([objectOption(a), objectOption(b)]);
  } else if (type === "closest_pair" || type === "farthest_pair") {
    const pairs = pairDistances(objects);
    const farthest = type === "farthest_pair";
    correct = pairs.find((pair) => pair.ids.slice().sort().join("+") === salience.correctId);
    prompt = `Which two shapes were ${farthest ? "furthest apart" : "closest together"}?`;
    const distractorPool = farthest ? pairs.slice(0, 9) : pairs.slice(-9);
    options = shuffle([pairOption(correct, objects), ...shuffle(distractorPool).slice(0, 3).map((pair) => pairOption(pair, objects))]);
    return { prompt, correctId: salience.correctId, options };
  } else {
    correct = objects.find((object) => object.id === salience.correctId);
    prompt = type === "highest"
      ? "Which shape was highest in the scene?"
      : type === "leftmost"
        ? "Which shape was furthest to the left?"
        : "Which shape was nearest to the center fixation ×?";
    options = shuffle([correct, ...shuffle(objects.filter((object) => object.id !== correct.id)).slice(0, 3)].map(objectOption));
  }
  return { prompt, correctId: correct.id, options };
}

function capitalize(text) { return text.charAt(0).toUpperCase() + text.slice(1); }

function beginSession(event) {
  event.preventDefault();
  const seed = makeSeed();
  rng = mulberry32(seed);
  session = {
    schemaVersion: 2,
    experiment: "SceneTrace gaze-centered relational search",
    participant: els.participant.value.trim() || `P-${String(seed).slice(-6)}`,
    seed,
    startedAt: new Date().toISOString(),
    config: CONFIG,
    userAgent: navigator.userAgent,
    viewport: { width: innerWidth, height: innerHeight, devicePixelRatio },
    conditions: createConditions(),
    trials: [],
    focusInterruptions: [],
    endedEarly: false,
  };
  showScreen(els.trial);
  runTrial(0);
}

function runTrial(index) {
  clearTimers();
  acceptingAnswer = false;
  if (index >= session.conditions.length) return finishSession();
  phase = "ready";
  const condition = session.conditions[index];
  const { objects, salience } = createScene(condition.questionType);
  const question = buildQuestion(condition.questionType, objects, salience);
  currentTrial = {
    index: index + 1,
    ...condition,
    questionOnsetMs: condition.delayMs,
    objects,
    salience,
    question,
    actualQuestionOnsetMs: null,
    response: null,
  };
  els.trialLabel.textContent = `TRIAL ${String(index + 1).padStart(2, "0")} / ${session.conditions.length}`;
  els.progress.style.width = `${(index / session.conditions.length) * 100}%`;
  els.questionPanel.classList.remove("visible");
  els.fixation.classList.add("visible");
  els.phaseMessage.textContent = "READY";
  els.phaseLabel.textContent = "GET READY";
  clearCanvas();
  later(() => exposeScene(), CONFIG.preTrialMs);
}

function exposeScene() {
  phase = "exposure";
  els.phaseMessage.textContent = "";
  els.phaseLabel.textContent = "OBSERVE";
  currentTrial.sceneShownAt = performance.now();
  drawScene(currentTrial.objects);
  els.fixation.classList.add("visible");
  later(showQuestion, currentTrial.delayMs);
}

function showQuestion() {
  phase = "question";
  currentTrial.actualQuestionOnsetMs = Math.round(performance.now() - currentTrial.sceneShownAt);
  els.fixation.classList.remove("visible");
  els.phaseLabel.textContent = "RESPOND";
  $("#focus-note").textContent = "Question and answers remain near center";
  els.questionText.textContent = currentTrial.question.prompt;
  els.answers.replaceChildren(...currentTrial.question.options.map((option, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "answer";
    button.dataset.answerId = option.id;
    const swatch = option.colors.length === 1
      ? `<span class="swatch" style="--swatch:${option.colors[0]}"></span>`
      : `<span class="swatch pair" style="--a:${option.colors[0]};--b:${option.colors[1]}"></span>`;
    button.innerHTML = `<span class="key">${index + 1}</span>${swatch}<span class="answer-label">${option.label}</span>`;
    button.addEventListener("click", () => answer(option.id));
    return button;
  }));
  els.questionPanel.classList.add("visible");
  questionShownAt = performance.now();
  acceptingAnswer = true;
  els.answers.querySelector("button")?.focus({ preventScroll: true });
  if (SMOKE_TEST) later(() => answer(currentTrial.question.correctId), 5);
}

function answer(answerId) {
  if (!acceptingAnswer || phase !== "question") return;
  acceptingAnswer = false;
  currentTrial.response = {
    answerId,
    correct: answerId === currentTrial.question.correctId,
    responseTimeMs: Math.round(performance.now() - questionShownAt),
    answeredAt: new Date().toISOString(),
  };
  session.trials.push(currentTrial);
  els.progress.style.width = `${(currentTrial.index / session.conditions.length) * 100}%`;
  els.questionPanel.classList.remove("visible");
  els.phaseMessage.textContent = "RESPONSE RECORDED";
  els.phaseLabel.textContent = "RECORDED";
  $("#focus-note").textContent = "Keep your eyes on the center ×";
  later(() => runTrial(currentTrial.index), SMOKE_TEST ? 5 : 650);
}

function resizeCanvas() {
  const rect = els.canvas.getBoundingClientRect();
  const ratio = Math.min(devicePixelRatio || 1, 2);
  els.canvas.width = Math.max(1, Math.round(rect.width * ratio));
  els.canvas.height = Math.max(1, Math.round(rect.height * ratio));
}

function clearCanvas() {
  resizeCanvas();
  const ctx = els.canvas.getContext("2d");
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, els.canvas.width, els.canvas.height);
}

function drawScene(objects) {
  resizeCanvas();
  const ctx = els.canvas.getContext("2d");
  const w = els.canvas.width, h = els.canvas.height;
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, w, h);
  objects.forEach((o) => drawShape(ctx, o, w, h));
}

function drawShape(ctx, object, width, height) {
  const scale = Math.min(width, height);
  const radius = object.size * scale;
  ctx.save();
  ctx.translate(object.x * width, object.y * height);
  ctx.rotate(object.angle);
  ctx.beginPath();
  if (object.shape === "circle") ctx.arc(0, 0, radius, 0, Math.PI * 2);
  else if (object.shape === "square") ctx.rect(-radius, -radius, radius * 2, radius * 2);
  else if (object.shape === "diamond") { ctx.moveTo(0, -radius * 1.25); ctx.lineTo(radius, 0); ctx.lineTo(0, radius * 1.25); ctx.lineTo(-radius, 0); ctx.closePath(); }
  else if (object.shape === "triangle") polygon(ctx, 3, radius * 1.25, -Math.PI / 2);
  else if (object.shape === "hexagon") polygon(ctx, 6, radius * 1.1, 0);
  else {
    const r = radius * .65, length = radius * 1.7;
    ctx.roundRect(-length / 2, -r, length, r * 2, r);
  }
  ctx.fillStyle = object.color;
  ctx.shadowColor = object.color;
  ctx.shadowBlur = radius * .35;
  ctx.fill();
  ctx.restore();
}

function polygon(ctx, sides, radius, offset) {
  for (let i = 0; i < sides; i++) {
    const angle = offset + (i * Math.PI * 2) / sides;
    const fn = i === 0 ? "moveTo" : "lineTo";
    ctx[fn](Math.cos(angle) * radius, Math.sin(angle) * radius);
  }
  ctx.closePath();
}

function finishSession(endedEarly = false) {
  clearTimers();
  phase = "results";
  session.endedEarly = endedEarly;
  session.completedAt = new Date().toISOString();
  session.durationMs = new Date(session.completedAt) - new Date(session.startedAt);
  showScreen(els.results);
  renderResults();
}

function renderResults() {
  const trials = session.trials;
  const correct = trials.filter((t) => t.response?.correct).length;
  const accuracy = trials.length ? correct / trials.length : 0;
  $("#score-value").textContent = `${Math.round(accuracy * 100)}%`;
  $("#session-meta").textContent = `${session.participant} · ${trials.length} trial${trials.length === 1 ? "" : "s"}${session.endedEarly ? " · ended early" : ""}`;
  $("#correct-metric").textContent = `${correct} / ${trials.length}`;
  const rts = trials.map((t) => t.response?.responseTimeMs).filter(Number.isFinite).sort((a, b) => a - b);
  const median = rts.length ? rts[Math.floor(rts.length / 2)] : null;
  $("#rt-metric").textContent = median === null ? "—" : `${(median / 1000).toFixed(2)} s`;
  $("#focus-metric").textContent = String(session.focusInterruptions.length);

  $("#delay-chart").replaceChildren(...CONFIG.delaysMs.map((delay) => {
    const subset = trials.filter((t) => t.delayMs === delay);
    const score = subset.length ? subset.filter((t) => t.response?.correct).length / subset.length : 0;
    const group = document.createElement("div");
    group.className = "bar-group";
    group.innerHTML = `<span class="bar-value">${subset.length ? Math.round(score * 100) + "%" : "—"}</span><div class="bar" style="height:${Math.max(1, score * 145)}px"></div><span class="bar-label">${delay / 1000} SEC</span>`;
    return group;
  }));

  $("#type-table").replaceChildren(...Object.entries(TYPE_LABELS).map(([type, label]) => {
    const subset = trials.filter((t) => t.questionType === type);
    const value = subset.length ? `${subset.filter((t) => t.response?.correct).length} / ${subset.length}` : "—";
    const cell = document.createElement("div");
    cell.className = "type-cell";
    cell.innerHTML = `<span>${label}</span><strong>${value}</strong>`;
    return cell;
  }));
}

function download(name, content, mime) {
  const link = document.createElement("a");
  link.href = URL.createObjectURL(new Blob([content], { type: mime }));
  link.download = name;
  link.click();
  later(() => URL.revokeObjectURL(link.href), 1000);
}

function csvEscape(value) {
  const text = value == null ? "" : String(value);
  return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function downloadCsv() {
  const columns = ["participant", "seed", "trial", "question_onset_ms", "actual_question_onset_ms", "question_type", "salience_margin", "salience_threshold", "generation_attempts", "question", "correct_answer", "response", "correct", "response_time_ms"];
  const rows = session.trials.map((t) => [session.participant, session.seed, t.index, t.questionOnsetMs, t.actualQuestionOnsetMs, t.questionType, t.salience.margin, t.salience.threshold, t.salience.attempts, t.question.prompt, t.question.correctId, t.response?.answerId, t.response?.correct, t.response?.responseTimeMs]);
  download(`scenetrace_${session.participant}_${session.seed}.csv`, [columns, ...rows].map((row) => row.map(csvEscape).join(",")).join("\n"), "text/csv");
}

els.participant.value = `P-${String(makeSeed()).slice(-6)}`;
els.form.addEventListener("submit", beginSession);
els.quit.addEventListener("click", () => els.dialog.showModal());
$("#continue-session").addEventListener("click", () => els.dialog.close());
$("#confirm-end").addEventListener("click", () => { els.dialog.close(); finishSession(true); });
$("#download-json").addEventListener("click", () => download(`scenetrace_${session.participant}_${session.seed}.json`, JSON.stringify(session, null, 2), "application/json"));
$("#download-csv").addEventListener("click", downloadCsv);
$("#restart").addEventListener("click", () => location.reload());
window.addEventListener("resize", () => {
  if (currentTrial && (phase === "exposure" || phase === "question")) drawScene(currentTrial.objects);
  else clearCanvas();
});
window.addEventListener("keydown", (event) => {
  if (phase === "question" && /^[1-4]$/.test(event.key)) {
    const button = els.answers.children[Number(event.key) - 1];
    if (button) button.click();
  }
});
document.addEventListener("visibilitychange", () => {
  if (session && document.hidden && !["welcome", "results"].includes(phase)) {
    session.focusInterruptions.push({ trial: currentTrial?.index ?? null, phase, at: new Date().toISOString() });
  }
});

if (SMOKE_TEST || PREVIEW_TEST) {
  $("#acknowledge").checked = true;
  els.form.requestSubmit();
}
