window.CaptionView = function (container, options) {
  const fieldOf = options.fieldOf;
  let followLive = true;
  const jump = options.jumpButton || null;

  function textFor(line) {
    if (line.hidden) return "";
    const value = line[fieldOf()] || "";
    return value;
  }

  function upsert(line) {
    let node = container.querySelector(`[data-id="${CSS.escape(line.id)}"]`);
    if (line.hidden) {
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
    node.textContent = text;
    node.dataset.state = line.state;
    node.className = line.state === "final" ? "final" : "provisional";
    if (options.decorate) options.decorate(node, line);
  }

  function applyResume(lines, provisional) {
    for (const line of lines || []) upsert(line);
    if (provisional && !provisional.hidden) upsert(provisional);
    follow();
  }

  function applySnapshot(snapshot) {
    const seen = new Set();
    for (const line of snapshot.lines || []) {
      upsert(line);
      seen.add(line.id);
    }
    if (snapshot.provisional && !snapshot.provisional.hidden) {
      upsert(snapshot.provisional);
      seen.add(snapshot.provisional.id);
    }
    for (const node of [...container.querySelectorAll("[data-id]")]) {
      if (!seen.has(node.dataset.id)) node.remove();
    }
    follow();
  }

  function applyEvent(event) {
    if (event.line) upsert(event.line);
    if (event.provisional) upsert(event.provisional);
    else if (event.type === "state" || event.type === "line" || event.type === "snapshot") {
      for (const node of [...container.querySelectorAll(".provisional")]) {
        if (!event.line || node.dataset.id !== event.line.id) node.remove();
      }
    }
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
