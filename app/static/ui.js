(function () {
  const LANGUAGES = { en: "English", id: "Bahasa Indonesia" };
  const STATUS = { ready: "Ready", live: "Live", paused: "Paused", ended: "Ended" };

  function el(tag, attrs, ...children) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(attrs || {})) {
      if (value === null || value === undefined || value === false) continue;
      if (key === "class") node.className = value;
      else if (key === "text") node.textContent = value;
      else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
      else node.setAttribute(key, value === true ? "" : value);
    }
    for (const child of children.flat()) {
      if (child === null || child === undefined || child === false) continue;
      node.append(child instanceof Node ? child : document.createTextNode(String(child)));
    }
    return node;
  }

  async function api(path, options = {}) {
    const init = { method: options.method || "GET", headers: { ...(options.headers || {}) } };
    if (options.body !== undefined) {
      init.headers["content-type"] = "application/json";
      init.body = JSON.stringify(options.body);
    }
    let response;
    try {
      response = await fetch(path, init);
    } catch (err) {
      throw new Error("The caption service is not reachable. Check the connection and try again.");
    }
    let data = {};
    try { data = await response.json(); } catch (err) { data = {}; }
    if (response.status === 401 && options.redirect !== false) {
      location.href = "/login";
      throw new Error("Sign in to continue.");
    }
    if (!response.ok) {
      const error = new Error(data.detail || options.fallback || "Something went wrong. Try again.");
      error.status = response.status;
      throw error;
    }
    return data;
  }

  function toast(message, kind = "info") {
    let region = document.querySelector(".toasts");
    if (!region) {
      region = el("div", { class: "toasts", role: "status", "aria-live": "polite" });
      document.body.append(region);
    }
    const item = el("div", { class: `toast toast-${kind}` }, message);
    region.append(item);
    setTimeout(() => {
      item.classList.add("leaving");
      setTimeout(() => item.remove(), 250);
    }, kind === "error" ? 6000 : 3000);
  }

  function busy(button, on, label) {
    if (on) {
      button.dataset.label = button.innerHTML;
      button.disabled = true;
      button.setAttribute("aria-busy", "true");
      button.replaceChildren(el("span", { class: "spinner", "aria-hidden": "true" }), label || button.textContent);
    } else {
      button.disabled = false;
      button.removeAttribute("aria-busy");
      if (button.dataset.label !== undefined) button.innerHTML = button.dataset.label;
    }
  }

  async function session(options = {}) {
    const me = await api("/api/me");
    if (options.church !== false && !me.church_id) {
      location.href = "/onboarding";
      throw new Error("Set up the church first.");
    }
    return me;
  }

  function header(me, active) {
    const bar = document.querySelector("#topbar");
    if (!bar) return;
    const links = [["services", "/", "Services"], ["settings", "/settings", "Settings"]];
    if (me.role === "admin") links.push(["users", "/admin", "Users"]);
    const signOut = el("button", { class: "btn btn-ghost btn-sm", type: "button" }, "Sign out");
    signOut.addEventListener("click", async () => {
      busy(signOut, true, "Signing out");
      try { await fetch("/api/logout", { method: "POST" }); } catch (err) { /* leave anyway */ }
      location.href = "/login";
    });
    bar.replaceChildren(
      el("a", { class: "brand", href: "/" }, el("img", { src: "/static/icon.svg", alt: "" }), el("span", {}, "voiceofgod")),
      el("nav", { class: "nav", "aria-label": "Main" },
        links.map(([key, href, label]) => el("a", { href, "aria-current": key === active ? "page" : null }, label))
      ),
      el("div", { class: "account" },
        el("span", { class: "avatar", "aria-hidden": "true" }, (me.username || "?").slice(0, 1)),
        el("span", { class: "account-name" },
          el("strong", {}, me.username),
          el("small", {}, me.church_name || (me.role === "admin" ? "Admin" : "Operator"))
        ),
        signOut
      )
    );
  }

  function dialog({ title, description, body, actions }) {
    const node = el("dialog", { class: "dialog" },
      el("form", { method: "dialog" },
        el("div", { class: "dialog-head" }, el("h2", {}, title), description ? el("p", {}, description) : null),
        body ? el("div", { class: "dialog-body" }, body) : null,
        el("div", { class: "dialog-foot" }, actions)
      )
    );
    node.addEventListener("close", () => setTimeout(() => node.remove(), 0));
    document.body.append(node);
    node.showModal();
    return node;
  }

  function confirmAction({ title, message, confirm = "Confirm", danger = false }) {
    return new Promise((resolve) => {
      const node = dialog({
        title,
        description: message,
        actions: [
          el("button", { class: "btn", value: "cancel" }, "Cancel"),
          el("button", { class: danger ? "btn btn-danger" : "btn btn-primary", value: "ok" }, confirm)
        ]
      });
      node.addEventListener("close", () => resolve(node.returnValue === "ok"));
    });
  }

  function statusBadge(status) {
    const known = STATUS[status] ? status : "ready";
    return el("span", { class: `badge badge-${known}` }, STATUS[known]);
  }

  function setBadge(node, status) {
    const known = STATUS[status] ? status : "ready";
    node.className = `badge badge-${known}`;
    node.textContent = STATUS[known];
  }

  function parseDay(value) {
    if (!value) return null;
    const date = /^\d{4}-\d{2}-\d{2}$/.test(value) ? new Date(`${value}T00:00:00`) : new Date(value);
    return Number.isNaN(date.getTime()) ? null : date;
  }

  function formatDate(value, options) {
    const date = parseDay(value);
    if (!date) return "";
    return date.toLocaleDateString(undefined, options || { weekday: "short", day: "numeric", month: "short", year: "numeric" });
  }

  function today() {
    const now = new Date();
    const pad = (n) => String(n).padStart(2, "0");
    return `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}`;
  }

  function languagePair(speaker) {
    const target = speaker === "en" ? "id" : "en";
    return `${LANGUAGES[speaker] || speaker} → ${LANGUAGES[target]}`;
  }

  function lines(value) {
    return value.split(/\n/).map((line) => line.trim()).filter(Boolean);
  }

  async function copy(text, label) {
    try {
      await navigator.clipboard.writeText(text);
      toast(`${label || "Link"} copied`, "success");
    } catch (err) {
      toast("Copy was blocked by the browser. Select the text and copy it.", "error");
    }
  }

  function password(length = 14) {
    const alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz23456789";
    const values = crypto.getRandomValues(new Uint32Array(length));
    return Array.from(values, (value) => alphabet[value % alphabet.length]).join("");
  }

  window.UI = {
    LANGUAGES, el, api, toast, busy, session, header, dialog, confirmAction,
    statusBadge, setBadge, parseDay, formatDate, today, languagePair, lines, copy, password
  };
})();
