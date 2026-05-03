const state = {
  threads: [],
  activeThreadId: null,
  view: "home",
  graphRenderer: null,
  graphLoaded: false,
};

const els = {
  landing: document.getElementById("landing"),
  appShell: document.getElementById("app"),
  enterApp: document.getElementById("enter-app"),
  heroEnterApp: document.getElementById("hero-enter-app"),
  appBrand: document.getElementById("app-brand"),
  threadList: document.getElementById("thread-list"),
  countThreads: document.getElementById("count-threads"),
  countEvents: document.getElementById("count-events"),
  countEntities: document.getElementById("count-entities"),
  newThreadBtn: document.getElementById("new-thread-btn"),
  searchForm: document.getElementById("search-form"),
  searchInput: document.getElementById("search-input"),
  memoryGraphBtn: document.getElementById("memory-graph-btn"),
  homeView: document.getElementById("home-view"),
  threadView: document.getElementById("thread-view"),
  graphView: document.getElementById("graph-view"),
  recentGrid: document.getElementById("recent-grid"),
  promptForm: document.getElementById("prompt-form"),
  promptInput: document.getElementById("prompt-input"),
  promptSend: document.getElementById("prompt-send"),
  answerCard: document.getElementById("answer-card"),
  answerBody: document.getElementById("answer-body"),
  answerContext: document.getElementById("answer-context"),
  answerDismiss: document.getElementById("answer-dismiss"),
  backBtn: document.getElementById("back-btn"),
  threadTitle: document.getElementById("thread-title"),
  threadSummary: document.getElementById("thread-summary"),
  threadMeta: document.getElementById("thread-meta"),
  threadEntities: document.getElementById("thread-entities"),
  threadEvents: document.getElementById("thread-events"),
  graphBackBtn: document.getElementById("graph-back-btn"),
  graphCanvas: document.getElementById("graph-canvas"),
  eventModal: document.getElementById("event-modal"),
  eventClose: document.getElementById("event-close"),
  eventContent: document.getElementById("event-content"),
};

async function requestJson(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return response.json();
}

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function formatRelative(timestamp) {
  if (!timestamp) return "";
  const then = new Date(timestamp).getTime();
  if (Number.isNaN(then)) return "";
  const diffSec = Math.max(0, (Date.now() - then) / 1000);
  if (diffSec < 60) return "just now";
  if (diffSec < 3600) return `${Math.floor(diffSec / 60)}m ago`;
  if (diffSec < 86400) return `${Math.floor(diffSec / 3600)}h ago`;
  if (diffSec < 86400 * 7) return `${Math.floor(diffSec / 86400)}d ago`;
  return new Date(timestamp).toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
  });
}

function formatAbsolute(timestamp) {
  if (!timestamp) return "";
  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function sourceLabel(source) {
  const labels = {
    chatgpt: "chatgpt",
    github: "github",
    youtube: "youtube",
    notion: "notion",
    claude: "claude",
    gemini: "gemini",
  };
  return labels[source] || source;
}

function shortenText(text, limit = 140) {
  const value = String(text ?? "");
  if (value.length <= limit) return value;
  return `${value.slice(0, limit - 1).trimEnd()}…`;
}

function setView(view) {
  state.view = view;
  els.homeView.classList.toggle("hidden", view !== "home");
  els.threadView.classList.toggle("hidden", view !== "thread");
  els.graphView.classList.toggle("hidden", view !== "graph");
}

function showHome() {
  state.activeThreadId = null;
  setView("home");
  highlightActiveThread();
  if (els.promptInput) {
    els.promptInput.focus();
  }
}

function highlightActiveThread() {
  for (const item of els.threadList.querySelectorAll("[data-thread-id]")) {
    item.classList.toggle(
      "active",
      Number(item.dataset.threadId) === state.activeThreadId
    );
  }
}

function renderSidebar(threads) {
  if (!threads.length) {
    els.threadList.innerHTML = `<div class="thread-empty">No threads yet.</div>`;
    return;
  }
  els.threadList.innerHTML = threads
    .map(
      (thread) => `
        <button class="thread-item" type="button" data-thread-id="${thread.id}">
          <span class="thread-item-title">${escapeHtml(thread.title)}</span>
          <span class="thread-item-time">${escapeHtml(formatRelative(thread.updated_at))}</span>
        </button>
      `
    )
    .join("");

  for (const item of els.threadList.querySelectorAll("[data-thread-id]")) {
    item.addEventListener("click", () => {
      openThread(Number(item.dataset.threadId));
    });
  }
  highlightActiveThread();
}

function renderRecent(threads) {
  const top = [...threads]
    .sort((a, b) => new Date(b.updated_at) - new Date(a.updated_at))
    .slice(0, 3);

  if (!top.length) {
    els.recentGrid.innerHTML = `<div class="recent-empty">Recent threads will appear here once events are ingested.</div>`;
    return;
  }

  els.recentGrid.innerHTML = top
    .map(
      (thread) => `
        <button class="recent-card" type="button" data-recent-id="${thread.id}">
          <h3 class="recent-card-title">${escapeHtml(thread.title)}</h3>
          <p class="recent-card-summary">${escapeHtml(shortenText(thread.summary, 160))}</p>
          <div class="recent-card-foot">
            <span><span class="recent-card-dot"></span>thread</span>
            <span>${escapeHtml(formatRelative(thread.updated_at))}</span>
          </div>
        </button>
      `
    )
    .join("");

  for (const item of els.recentGrid.querySelectorAll("[data-recent-id]")) {
    item.addEventListener("click", () => openThread(Number(item.dataset.recentId)));
  }
}

function renderCounts(counts) {
  if (!counts) return;
  els.countThreads.textContent = counts.threads ?? 0;
  els.countEvents.textContent = counts.events ?? 0;
  els.countEntities.textContent = counts.entities ?? 0;
}

async function loadOverview() {
  try {
    const overview = await requestJson("/api/overview");
    state.threads = overview.threads || [];
    renderSidebar(state.threads);
    renderRecent(state.threads);
    renderCounts(overview.counts);
  } catch (error) {
    console.error("loadOverview failed", error);
    els.threadList.innerHTML = `<div class="thread-empty">Failed to load.</div>`;
  }
}

async function searchThreads(query) {
  const trimmed = query.trim();
  if (!trimmed) {
    await loadOverview();
    return;
  }
  try {
    const threads = await requestJson(`/threads?q=${encodeURIComponent(trimmed)}`);
    state.threads = threads;
    renderSidebar(threads);
    renderRecent(threads);
  } catch (error) {
    console.error("searchThreads failed", error);
  }
}

function renderThread(payload) {
  els.threadTitle.textContent = payload.thread.title;
  els.threadSummary.textContent = payload.thread.summary;
  els.threadMeta.textContent = `Updated ${formatAbsolute(payload.thread.updated_at)} · ${payload.events.length} events · ${payload.entities.length} entities`;

  els.threadEntities.innerHTML = payload.entities.length
    ? payload.entities
        .map((entity) => `<span class="entity-chip">${escapeHtml(entity.name)}</span>`)
        .join("")
    : `<span class="entity-chip" style="opacity:0.6">no entities yet</span>`;

  els.threadEvents.innerHTML = payload.events
    .map(
      (event) => `
        <button class="event-row" type="button" data-event-id="${event.id}">
          <div class="event-row-top">
            <h3>${escapeHtml(event.content)}</h3>
            <span class="source-chip">${escapeHtml(sourceLabel(event.source))}</span>
          </div>
          <span class="event-row-time">${escapeHtml(formatAbsolute(event.timestamp))}</span>
        </button>
      `
    )
    .join("");

  for (const item of els.threadEvents.querySelectorAll("[data-event-id]")) {
    const ev = payload.events.find((entry) => entry.id === Number(item.dataset.eventId));
    if (ev) {
      item.addEventListener("click", () => openEventModal(ev));
    }
  }
}

async function openThread(threadId) {
  try {
    const payload = await requestJson(`/threads/${threadId}`);
    state.activeThreadId = threadId;
    renderThread(payload);
    setView("thread");
    highlightActiveThread();
    document.querySelector(".main").scrollTop = 0;
  } catch (error) {
    console.error("openThread failed", error);
  }
}

function formatMetaLabel(key) {
  if (key === "url") return "URL";
  if (key === "id") return "ID";
  return key.charAt(0).toUpperCase() + key.slice(1).replace(/[_-]/g, " ");
}

function renderMetaValue(key, value) {
  const str = String(value ?? "");
  if ((key === "url" || key.endsWith("_url")) && /^https?:\/\//.test(str)) {
    return `<a class="meta-link" href="${escapeHtml(str)}" target="_blank" rel="noopener noreferrer">${escapeHtml(str)}</a>`;
  }
  return escapeHtml(str);
}

function renderMetaList(metadata) {
  const entries = Object.entries(metadata || {}).filter(([, value]) => value != null && value !== "");
  if (!entries.length) return "";
  return `
    <dl class="event-meta-list">
      ${entries
        .map(
          ([key, value]) => `
            <div class="event-meta-item">
              <dt class="meta-label">${escapeHtml(formatMetaLabel(key))}</dt>
              <dd class="meta-value">${renderMetaValue(key, value)}</dd>
            </div>
          `
        )
        .join("")}
    </dl>
  `;
}

function openEventModal(event) {
  els.eventContent.innerHTML = `
    <div class="event-modal-head">
      <span class="event-eyebrow-row">
        <span class="event-source-pill">${escapeHtml(sourceLabel(event.source))}</span>
        <span class="event-time">${escapeHtml(formatAbsolute(event.timestamp))}</span>
      </span>
      <p class="event-eyebrow">Captured event</p>
    </div>

    ${renderMetaList(event.metadata)}

    <div class="event-content-block">
      <p class="event-content-label">Content</p>
      <h2>${escapeHtml(event.content)}</h2>
    </div>

    <p class="event-foot">
      Context shouldn't live inside the tools you use — it should live outside,
      where every agent can read it. This is one of the events Meniscus captured
      to build a shared, structured memory of what you're working on.
    </p>
  `;
  if (typeof els.eventModal.showModal === "function") {
    els.eventModal.showModal();
  }
}

function showAnswerThinking() {
  els.answerCard.classList.remove("hidden");
  els.answerBody.classList.add("thinking");
  els.answerBody.textContent = "Retrieving subgraph…";
  els.answerContext.innerHTML = "";
}

function renderAnswer(payload) {
  els.answerCard.classList.remove("hidden");
  els.answerBody.classList.remove("thinking");
  els.answerBody.textContent = payload.answer || "No answer available.";

  // Update the eyebrow label to reflect what mode the response came from.
  const labelEl = els.answerCard.querySelector(".answer-label");
  if (labelEl) {
    if (payload.mode === "general") labelEl.textContent = "General reply";
    else if (payload.mode === "overview") labelEl.textContent = "Activity overview";
    else labelEl.textContent = "Retrieved answer";
  }

  const threads = payload.threads || [];
  if (!threads.length) {
    // General mode — no retrieval happened, hide the citation row entirely.
    els.answerContext.innerHTML = "";
    els.answerContext.style.display = "none";
    return;
  }
  els.answerContext.style.display = "";

  els.answerContext.innerHTML = threads
    .map(
      (thread) => `
        <button class="context-chip" type="button" data-answer-thread="${thread.id}">
          <span>${escapeHtml(thread.title)}</span>
          <span class="context-chip-meta">${escapeHtml(formatRelative(thread.updated_at))}</span>
        </button>
      `
    )
    .join("");

  for (const item of els.answerContext.querySelectorAll("[data-answer-thread]")) {
    item.addEventListener("click", () => openThread(Number(item.dataset.answerThread)));
  }
}

async function ask(question) {
  if (!question) return;
  showAnswerThinking();
  els.promptSend.disabled = true;
  try {
    const payload = await requestJson("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    renderAnswer(payload);
  } catch (error) {
    console.error("ask failed", error);
    els.answerBody.classList.remove("thinking");
    els.answerBody.textContent = "Meniscus could not answer that. Try rephrasing.";
    els.answerContext.innerHTML = "";
  } finally {
    els.promptSend.disabled = false;
  }
}

function autosizePrompt() {
  els.promptInput.style.height = "auto";
  els.promptInput.style.height = `${Math.min(els.promptInput.scrollHeight, 220)}px`;
}

/* ---------- Graph rendering ----------
   Calm alpha-cooled force simulation.
   Nodes start scattered, settle within ~3 seconds, never explode. */

function getNodeColor(kind) {
  if (kind === "thread") return "#6f6bd9";
  if (kind === "entity") return "#b3b1c8";
  return "#8c8c95";
}

function getNodeStroke(kind) {
  if (kind === "thread") return "rgba(91, 88, 196, 0.35)";
  if (kind === "entity") return "rgba(120, 120, 140, 0.20)";
  return "rgba(110, 110, 122, 0.20)";
}

function createGraphRenderer(container) {
  const canvas = document.createElement("canvas");
  container.replaceChildren(canvas);
  const context = canvas.getContext("2d");
  const renderer = {
    frame: null,
    data: { nodes: [], links: [], nodeIndex: new Map() },
    dpr: 1,
    alpha: 1,
    alphaDecay: 0.018,
    alphaMin: 0.005,
  };

  function resize() {
    const bounds = container.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    renderer.dpr = dpr;
    canvas.width = Math.max(320, Math.floor(bounds.width * dpr));
    canvas.height = Math.max(360, Math.floor(bounds.height * dpr));
    canvas.style.width = `${bounds.width}px`;
    canvas.style.height = `${bounds.height}px`;
  }

  function buildNodes(nodes) {
    const cx = canvas.width / 2;
    const cy = canvas.height / 2;
    const spread = Math.min(canvas.width, canvas.height) * 0.32;
    // Scatter inside an ellipse around center — uniform area distribution.
    return nodes.map((node) => {
      const angle = Math.random() * Math.PI * 2;
      const r = Math.sqrt(Math.random()) * spread;
      return {
        ...node,
        x: cx + Math.cos(angle) * r,
        y: cy + Math.sin(angle) * r,
        vx: 0,
        vy: 0,
      };
    });
  }

  function indexNodes(nodes) {
    const map = new Map();
    for (const node of nodes) map.set(node.id, node);
    return map;
  }

  function simulate() {
    if (renderer.alpha < renderer.alphaMin) return;

    const dpr = renderer.dpr;
    const a = renderer.alpha;
    const nodes = renderer.data.nodes;
    const cx = canvas.width / 2;
    const cy = canvas.height / 2;

    // Pairwise repulsion (O(n²), fine for ~60 nodes).
    for (let i = 0; i < nodes.length; i += 1) {
      for (let j = i + 1; j < nodes.length; j += 1) {
        const ni = nodes[i];
        const nj = nodes[j];
        const dx = nj.x - ni.x;
        const dy = nj.y - ni.y;
        const distSq = dx * dx + dy * dy + 0.5;
        const dist = Math.sqrt(distSq);
        // Tuned strength: enough to spread, soft enough not to launch.
        const strength = (900 * dpr * dpr * a) / distSq;
        const fx = (dx / dist) * strength;
        const fy = (dy / dist) * strength;
        ni.vx -= fx;
        ni.vy -= fy;
        nj.vx += fx;
        nj.vy += fy;
      }
    }

    // Spring forces along edges.
    for (const link of renderer.data.links) {
      const source = renderer.data.nodeIndex.get(link.source);
      const target = renderer.data.nodeIndex.get(link.target);
      if (!source || !target) continue;
      const dx = target.x - source.x;
      const dy = target.y - source.y;
      const dist = Math.sqrt(dx * dx + dy * dy) + 0.01;
      const rest = link.kind === "event_thread" ? 180 * dpr : 110 * dpr;
      const strength = ((dist - rest) * 0.04) * a;
      const fx = (dx / dist) * strength;
      const fy = (dy / dist) * strength;
      source.vx += fx;
      source.vy += fy;
      target.vx -= fx;
      target.vy -= fy;
    }

    // Subtle center gravity so nothing drifts off-screen.
    const gravity = 0.012 * a;
    for (const node of nodes) {
      node.vx += (cx - node.x) * gravity;
      node.vy += (cy - node.y) * gravity;
    }

    // Apply velocities — clamp + damp + bounds.
    const maxV = 18 * dpr;
    const inset = 40 * dpr;
    for (const node of nodes) {
      if (node.vx > maxV) node.vx = maxV;
      else if (node.vx < -maxV) node.vx = -maxV;
      if (node.vy > maxV) node.vy = maxV;
      else if (node.vy < -maxV) node.vy = -maxV;

      node.vx *= 0.62;
      node.vy *= 0.62;

      node.x = Math.min(canvas.width - inset, Math.max(inset, node.x + node.vx));
      node.y = Math.min(canvas.height - inset, Math.max(inset, node.y + node.vy));
    }

    renderer.alpha += (renderer.alphaMin - renderer.alpha) * renderer.alphaDecay;
  }

  function draw() {
    const dpr = renderer.dpr;
    context.clearRect(0, 0, canvas.width, canvas.height);

    // Edges
    for (const link of renderer.data.links) {
      const source = renderer.data.nodeIndex.get(link.source);
      const target = renderer.data.nodeIndex.get(link.target);
      if (!source || !target) continue;
      const grad = context.createLinearGradient(source.x, source.y, target.x, target.y);
      grad.addColorStop(0, "rgba(111, 107, 217, 0.28)");
      grad.addColorStop(1, "rgba(120, 120, 140, 0.12)");
      context.beginPath();
      context.strokeStyle = grad;
      context.lineWidth = 1 * dpr;
      context.moveTo(source.x, source.y);
      context.lineTo(target.x, target.y);
      context.stroke();
    }

    // Nodes
    for (const node of renderer.data.nodes) {
      const baseRadius = (node.size || 12) * 0.55 * dpr;
      const color = getNodeColor(node.kind);

      // Soft halo behind threads.
      if (node.kind === "thread") {
        context.beginPath();
        context.fillStyle = "rgba(111, 107, 217, 0.14)";
        context.arc(node.x, node.y, baseRadius * 2.1, 0, Math.PI * 2);
        context.fill();
      }

      context.beginPath();
      context.fillStyle = color;
      context.arc(node.x, node.y, baseRadius, 0, Math.PI * 2);
      context.fill();
      context.lineWidth = 1 * dpr;
      context.strokeStyle = getNodeStroke(node.kind);
      context.stroke();

      const label = (node.label || "").length > 28 ? `${node.label.slice(0, 27)}…` : node.label || "";
      if (!label) continue;

      const fontSize = node.kind === "thread" ? 12 * dpr : node.kind === "entity" ? 11 * dpr : 10 * dpr;
      context.font = `${node.kind === "thread" ? 600 : 400} ${fontSize}px "JetBrains Mono", Menlo, monospace`;
      context.fillStyle = node.kind === "thread" ? "rgba(21, 21, 26, 0.92)" : "rgba(60, 60, 70, 0.74)";
      context.fillText(label, node.x + baseRadius + 8 * dpr, node.y + 4 * dpr);
    }
  }

  function tick() {
    simulate();
    draw();
    renderer.frame = window.requestAnimationFrame(tick);
  }

  function setData(data) {
    if (renderer.frame) {
      window.cancelAnimationFrame(renderer.frame);
      renderer.frame = null;
    }
    resize();
    const nodes = buildNodes(data.nodes);
    renderer.data = {
      nodes,
      links: data.links,
      nodeIndex: indexNodes(nodes),
    };
    renderer.alpha = 1;
    tick();
  }

  function reheat() {
    renderer.alpha = Math.max(renderer.alpha, 0.4);
  }

  function stop() {
    if (renderer.frame) {
      window.cancelAnimationFrame(renderer.frame);
      renderer.frame = null;
    }
  }

  window.addEventListener("resize", () => {
    if (state.view !== "graph") return;
    resize();
    reheat();
  });

  return { setData, stop };
}

async function showGraph() {
  setView("graph");
  if (!state.graphRenderer) {
    state.graphRenderer = createGraphRenderer(els.graphCanvas);
  }
  try {
    const payload = await requestJson("/api/graph");
    state.graphRenderer.setData(payload);
  } catch (error) {
    console.error("showGraph failed", error);
  }
}

/* ---------- Wire up events ---------- */

els.newThreadBtn.addEventListener("click", () => {
  showHome();
});

els.searchForm.addEventListener("submit", (event) => {
  event.preventDefault();
  searchThreads(els.searchInput.value);
});

let searchDebounce = null;
els.searchInput.addEventListener("input", () => {
  if (searchDebounce) {
    window.clearTimeout(searchDebounce);
  }
  searchDebounce = window.setTimeout(() => {
    searchThreads(els.searchInput.value);
  }, 220);
});

els.memoryGraphBtn.addEventListener("click", () => {
  showGraph();
});

els.backBtn.addEventListener("click", () => {
  showHome();
});

els.graphBackBtn.addEventListener("click", () => {
  showHome();
});

els.promptForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = els.promptInput.value.trim();
  if (!question) return;
  ask(question);
});

els.promptInput.addEventListener("input", autosizePrompt);

els.promptInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    els.promptForm.requestSubmit();
  }
});

els.answerDismiss.addEventListener("click", () => {
  els.answerCard.classList.add("hidden");
  els.answerBody.textContent = "";
  els.answerContext.innerHTML = "";
});

els.eventClose.addEventListener("click", () => {
  els.eventModal.close();
});

els.eventModal.addEventListener("click", (event) => {
  if (event.target === els.eventModal) {
    els.eventModal.close();
  }
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    if (state.view === "thread" || state.view === "graph") {
      showHome();
    }
  }
});

/* ---------- Landing ↔ Workspace ---------- */

function showWorkspace() {
  els.landing.classList.add("hidden");
  els.appShell.classList.remove("hidden");
  document.body.style.overflow = "hidden";
  if (window.location.hash !== "#app") {
    history.replaceState(null, "", "#app");
  }
  showHome();
}

function showLanding() {
  els.appShell.classList.add("hidden");
  els.landing.classList.remove("hidden");
  document.body.style.overflow = "";
  if (window.location.hash) {
    history.replaceState(null, "", window.location.pathname);
  }
  window.scrollTo({ top: 0, behavior: "auto" });
}

if (els.enterApp) els.enterApp.addEventListener("click", showWorkspace);
if (els.heroEnterApp) els.heroEnterApp.addEventListener("click", showWorkspace);
if (els.appBrand) els.appBrand.addEventListener("click", showLanding);

for (const node of document.querySelectorAll("[data-scroll]")) {
  node.addEventListener("click", () => {
    const target = document.getElementById(node.dataset.scroll);
    if (target) target.scrollIntoView({ behavior: "smooth", block: "start" });
  });
}

async function bootstrap() {
  await loadOverview();
  autosizePrompt();
  if (window.location.hash === "#app") {
    showWorkspace();
  } else {
    showLanding();
  }
}

bootstrap().catch((error) => {
  console.error("bootstrap failed", error);
});
