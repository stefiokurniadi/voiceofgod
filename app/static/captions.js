window.CaptionView = function (container, options) {
  const fieldOf = options.fieldOf;
  let followLive = true;
  const jump = options.jumpButton || null;

  function showGrowing(line) {
    if (!line || line.hidden) return false;
    return !finalsOnly() || Boolean(line.suppressed);
  }

  function finalsOnly() {
    if (typeof options.finalsOnly === "function") return Boolean(options.finalsOnly());
    return Boolean(options.finalsOnly);
  }

  function dictating(line) {
    if (!line || line.state !== "provisional" || line.suppressed) return false;
    if (typeof options.dictate === "function") return Boolean(options.dictate());
    return Boolean(options.dictate);
  }

  function textFor(line) {
    if (finalsOnly() && line.state !== "final" && !line.suppressed) return "";
    if (line.hidden && !options.keepHidden) return "";
    const value = line[fieldOf()] || "";
    return value;
  }

  function clock(seconds) {
    const whole = Math.max(0, Math.floor(Number(seconds) || 0));
    const hours = Math.floor(whole / 3600);
    const minutes = Math.floor((whole % 3600) / 60);
    const secs = whole % 60;
    const pad = (value) => String(value).padStart(2, "0");
    return hours ? `${hours}:${pad(minutes)}:${pad(secs)}` : `${minutes}:${pad(secs)}`;
  }

  function ensureBody(node) {
    let body = node.querySelector(":scope > .line-text");
    if (!body) {
      body = document.createElement("span");
      body.className = "line-text";
      node.append(body);
    }
    return body;
  }

  function ensureTime(node, line) {
    if (!options.showTime) return;
    let time = node.querySelector(":scope > .line-time");
    if (!time) {
      time = document.createElement("time");
      time.className = "line-time";
      node.prepend(time);
    }
    const label = clock(line.t0);
    time.textContent = label;
    time.dateTime = label;
  }

  function markLive() {
    const nodes = [...container.querySelectorAll(":scope > [data-id]")];
    nodes.forEach((node, index) => node.classList.toggle("live", index === nodes.length - 1));
  }

  function renderDictation(node, text, settledCount) {
    const words = text.trim().split(/\s+/).filter(Boolean);
    const settled = Math.min(Math.max(0, settledCount || 0), words.length);
    const previous = node.dataset.dictated ? node.dataset.dictated.split("\n") : [];
    node.replaceChildren();
    words.forEach((word, index) => {
      if (index) node.append(document.createTextNode(" "));
      const span = document.createElement("span");
      span.textContent = word;
      const firm = index < settled;
      span.className = firm ? "settled" : "pending";
      if (!firm && (previous[index] || "").toLowerCase() !== word.toLowerCase()) span.classList.add("arriving");
      node.append(span);
    });
    node.dataset.dictated = words.join("\n");
  }

  function upsert(line) {
    let node = container.querySelector(`[data-id="${CSS.escape(line.id)}"]`);
    if (line.hidden && !options.keepHidden) {
      if (node) node.remove();
      return;
    }
    const text = textFor(line);
    if (!text) {
      if (node && line.state !== "final") node.remove();
      if (!text && line.state === "provisional") {
        if (node) node.remove();
        return;
      }
    }
    if (!node) {
      node = document.createElement("p");
      node.dataset.id = line.id;
      container.appendChild(node);
    }
    const body = ensureBody(node);
    if (dictating(line)) renderDictation(body, text, Number(line.settled) || 0);
    else {
      body.textContent = text;
      delete body.dataset.dictated;
    }
    ensureTime(node, line);
    node.dataset.state = line.state;
    node.className = line.state === "final" ? "final" : "provisional";
    node.classList.toggle("hidden-line", Boolean(line.hidden));
    if (options.decorate) options.decorate(node, line);
    markLive();
  }

  function clearProvisional(exceptId) {
    for (const node of [...container.querySelectorAll(".provisional")]) {
      if (!exceptId || node.dataset.id !== exceptId) node.remove();
    }
  }

  function applyResume(lines, provisional) {
    for (const line of lines || []) upsert(line);
    if (showGrowing(provisional)) upsert(provisional);
    else clearProvisional();
    markLive();
    follow();
  }

  function applySnapshot(snapshot) {
    const seen = new Set();
    for (const line of snapshot.lines || []) {
      upsert(line);
      seen.add(line.id);
    }
    if (showGrowing(snapshot.provisional)) {
      upsert(snapshot.provisional);
      seen.add(snapshot.provisional.id);
    }
    for (const node of [...container.querySelectorAll("[data-id]")]) {
      if (!seen.has(node.dataset.id)) node.remove();
    }
    markLive();
    follow();
  }

  function applyEvent(event) {
    if (event.line) upsert(event.line);
    if (showGrowing(event.provisional)) upsert(event.provisional);
    else if (event.type === "state" || event.type === "line" || event.type === "snapshot" || event.type === "resume") {
      clearProvisional(event.line && event.line.id);
    }
    markLive();
    follow();
  }

  function refresh(snapshot) {
    applySnapshot(snapshot);
  }

  container.addEventListener("scroll", () => {
    const distance = container.scrollHeight - container.scrollTop - container.clientHeight;
    followLive = distance < 90;
    if (jump) jump.hidden = followLive;
  });

  function follow() {
    if (followLive) container.scrollTop = container.scrollHeight;
  }

  if (jump) {
    jump.addEventListener("click", () => {
      followLive = true;
      jump.hidden = true;
      follow();
    });
  }

  return { applySnapshot, applyEvent, applyResume, refresh, follow };
};
