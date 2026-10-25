// Page behaviors shared by every page: menus, toasts, the fact drawer, the timeline's year
// rail and keyboard moves, and the Ask panel. No inline scripts (strict CSP). Case text from
// the user is inserted with textContent; HTML only ever comes from our own server responses.
(function () {
  "use strict";

  var $ = function (sel, root) { return (root || document).querySelector(sel); };
  var $$ = function (sel, root) { return Array.prototype.slice.call((root || document).querySelectorAll(sel)); };
  var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  var narrow = window.matchMedia("(max-width: 860px)");
  var scrollBehavior = function () { return reduceMotion.matches ? "auto" : "smooth"; };
  var store = {
    get: function (k) { try { return window.sessionStorage.getItem(k); } catch (e) { return null; } },
    set: function (k, v) { try { window.sessionStorage.setItem(k, v); } catch (e) { /* private mode */ } },
    del: function (k) { try { window.sessionStorage.removeItem(k); } catch (e) { /* private mode */ } },
  };

  /* ------------------------------------------------------------ toast */

  var toastTimer = null;
  function toast(message) {
    var el = document.getElementById("toast");
    if (!el) return;
    el.textContent = message;
    el.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(function () { el.hidden = true; }, 6000);
  }
  window.ctToast = toast;

  // Errors from HTMX requests (422 validation, 403, ...) arrive as plain text.
  document.addEventListener("htmx:responseError", function (evt) {
    var xhr = evt.detail && evt.detail.xhr;
    var text = xhr && xhr.responseText ? xhr.responseText : "Request failed";
    if (text.length > 400) text = text.slice(0, 400) + "…";
    toast(text);
  });
  document.addEventListener("htmx:sendError", function () {
    toast("Could not reach the server.");
  });

  /* ------------------------------------------------------------ forms */

  // Auto-submitting selects (with a plain button as the no-JS fallback).
  document.addEventListener("change", function (evt) {
    var el = evt.target;
    if (el instanceof HTMLSelectElement && el.hasAttribute("data-autosubmit") && el.form) {
      if (el.form.requestSubmit) el.form.requestSubmit(); else el.form.submit();
    }
  });

  document.addEventListener("submit", function (evt) {
    var form = evt.target;
    if (!(form instanceof HTMLFormElement)) return;
    var question = form.getAttribute("data-confirm");
    if (question && !window.confirm(question)) evt.preventDefault();
  }, true);

  // Upload drop zone: dropping files fills the form's file input.
  $$("[data-dropzone]").forEach(function (zone) {
    var input = $('input[type="file"]', zone);
    ["dragenter", "dragover"].forEach(function (name) {
      zone.addEventListener(name, function (evt) { evt.preventDefault(); zone.classList.add("is-over"); });
    });
    ["dragleave", "drop"].forEach(function (name) {
      zone.addEventListener(name, function () { zone.classList.remove("is-over"); });
    });
    zone.addEventListener("drop", function (evt) {
      evt.preventDefault();
      if (input && evt.dataTransfer && evt.dataTransfer.files.length) input.files = evt.dataTransfer.files;
    });
  });

  /* ------------------------------------------------------------ menus */

  function closeMenus(except) {
    $$("details.menu[open]").forEach(function (d) { if (!except || !d.contains(except)) d.removeAttribute("open"); });
  }
  document.addEventListener("click", function (evt) { closeMenus(evt.target); });
  // Opening a menu with a text field focuses it.
  document.addEventListener("toggle", function (evt) {
    var d = evt.target;
    if (!(d instanceof HTMLDetailsElement) || !d.open || !d.classList.contains("menu")) return;
    var field = $(".panel input:not([type=hidden]):not([type=checkbox]), .panel textarea", d);
    if (field && d.querySelector("form.form-panel, .whoform")) field.focus();
  }, true);

  /* ------------------------------------------------------------ fact drawer */

  var drawer = document.getElementById("drawer");
  var drawerBody = document.getElementById("drawer-body");
  var drawerOpen = false;
  var drawerFact = null;

  function setCurrent(id) {
    $$("tr.fact.is-current").forEach(function (r) { r.classList.remove("is-current"); });
    var row = id ? document.getElementById(id) : null;
    if (row && row.classList.contains("fact")) row.classList.add("is-current");
    currentId = id;
  }

  function focusTarget(opts) {
    var target = null;
    if (opts.ev) target = drawerBody.querySelector('[id="' + CSS.escape(opts.ev) + '"]');
    else if (opts.sec === "conflicts") target = $("#d-conflicts", drawerBody);
    if (!target) return;
    target.classList.add("is-target");
    target.scrollIntoView({ block: "nearest" });
  }

  function openFact(id, opts) {
    if (!drawer || !window.htmx) return false;
    opts = opts || {};
    setCurrent(id);
    var url = "/facts/" + encodeURIComponent(id);
    $("#drawer-id").textContent = id;
    $("#drawer-full").href = url;
    drawer.classList.add("is-open");
    drawer.setAttribute("aria-hidden", "false");
    document.body.classList.add("drawer-open");
    drawerOpen = true;
    if (drawerFact === id && !opts.reload) {
      focusTarget(opts);
      return true;
    }
    drawerFact = id;
    drawerBody.textContent = "";
    var skeleton = document.createElement("div");
    skeleton.className = "drawer-loading";
    for (var i = 0; i < 5; i++) skeleton.appendChild(document.createElement("span"));
    drawerBody.appendChild(skeleton);
    window.htmx.ajax("GET", url + "?partial=1", { target: "#drawer-body", swap: "innerHTML" }).then(function () {
      drawerBody.scrollTop = 0;
      if (!opts.keepFocus) drawerBody.focus({ preventScroll: true });
      focusTarget(opts);
    });
    return true;
  }
  window.ctOpenFact = openFact;

  function closeDrawer() {
    if (!drawer) return;
    drawer.classList.remove("is-open");
    drawer.setAttribute("aria-hidden", "true");
    document.body.classList.remove("drawer-open");
    drawerOpen = false;
    var row = currentId && document.getElementById(currentId);
    var link = row && $(".ftitle", row);
    if (link) link.focus({ preventScroll: true });
  }

  document.addEventListener("click", function (evt) {
    var t = evt.target instanceof Element ? evt.target.closest("[data-open], [data-close-drawer]") : null;
    if (!t) return;
    if (t.hasAttribute("data-close-drawer")) { closeDrawer(); return; }
    if (evt.metaKey || evt.ctrlKey || evt.shiftKey || evt.altKey || evt.button !== 0) return;
    if (openFact(t.getAttribute("data-open"), {
      ev: t.getAttribute("data-ev"),
      sec: t.getAttribute("data-sec"),
      keepFocus: !!t.closest("#drawer"),
    })) evt.preventDefault();
  });

  /* ------------------------------------------------------------ timeline board */

  var table = document.getElementById("facts");
  var rail = document.getElementById("rail");
  var currentId = null;
  var collapsed = {};
  var COLLAPSE_KEY = "ct-collapsed:" + location.pathname;

  function stickyOffset() {
    var bar = $(".appbar");
    var top = bar ? bar.getBoundingClientRect().bottom : 0;
    if (narrow.matches && rail) return top + rail.offsetHeight + 6;
    return top + (table && table.tHead ? table.tHead.offsetHeight : 0) + 6;
  }

  function updateSpy() {
    if (!table || !rail) return;
    var bodies = Array.prototype.slice.call(table.tBodies);
    if (!bodies.length) return;
    var line = stickyOffset();
    var tb = bodies.filter(function (b) { return b.getBoundingClientRect().top <= line + 12; }).pop() || bodies[0];
    var year = tb.dataset.year;
    $$(".rail-group", rail).forEach(function (g) { g.classList.toggle("is-active", g.dataset.year === year); });
    $$(".rail-year", rail).forEach(function (b) {
      var on = b.dataset.jumpYear === year;
      b.classList.toggle("is-active", on);
      if (on) b.setAttribute("aria-current", "true"); else b.removeAttribute("aria-current");
    });
    $$(".tick.is-here", rail).forEach(function (t) { t.classList.remove("is-here"); });
    if (!tb.classList.contains("is-collapsed")) {
      var top = $$("tr.fact", tb).filter(function (r) { return r.getBoundingClientRect().bottom > line; })[0];
      if (top) {
        var tick = rail.querySelector('.tick[data-jump="' + top.id + '"]');
        if (tick) tick.classList.add("is-here");
      }
    }
    var active = $(".rail-year.is-active", rail);
    if (!active) return;
    if (narrow.matches) {
      var left = active.offsetLeft - 16;
      if (left < rail.scrollLeft || active.offsetLeft + active.offsetWidth > rail.scrollLeft + rail.clientWidth) rail.scrollLeft = left;
    } else if (rail.scrollHeight > rail.clientHeight) {
      var t = active.offsetTop;
      if (t < rail.scrollTop || t > rail.scrollTop + rail.clientHeight - 60) rail.scrollTop = t - 8;
    }
  }

  function setCollapsed(year, on) {
    var tb = document.getElementById("y-" + year);
    if (!tb) return;
    tb.classList.toggle("is-collapsed", on);
    var btn = $(".group-toggle", tb);
    if (btn) btn.setAttribute("aria-expanded", String(!on));
    if (on) collapsed[year] = true; else delete collapsed[year];
  }

  function syncCollapseAll() {
    var btn = $("[data-collapse-all]");
    if (!btn || !table) return;
    var years = $$("tbody[data-year]", table).map(function (b) { return b.dataset.year; });
    var all = years.length && years.every(function (y) { return collapsed[y]; });
    btn.textContent = all ? "Expand all" : "Collapse all";
    store.set(COLLAPSE_KEY, JSON.stringify(Object.keys(collapsed)));
  }

  function flash(row) {
    if (reduceMotion.matches) return;
    row.classList.remove("is-flash");
    void row.offsetWidth;
    row.classList.add("is-flash");
  }

  function jumpTo(id, focus) {
    var row = document.getElementById(id);
    if (!row) return;
    var tb = row.closest("tbody");
    if (tb && tb.classList.contains("is-collapsed")) { setCollapsed(tb.dataset.year, false); syncCollapseAll(); }
    setCurrent(id);
    var link = $(".ftitle", row);
    if (focus !== false && link) link.focus({ preventScroll: true });
    row.scrollIntoView({ block: "start", behavior: scrollBehavior() });
    flash(row);
  }

  function step(dir) {
    if (!table) return;
    var rows = $$("tbody:not(.is-collapsed) tr.fact", table);
    if (!rows.length) return;
    var i = rows.map(function (r) { return r.id; }).indexOf(currentId);
    var next;
    if (i === -1) {
      var line = stickyOffset();
      next = rows.filter(function (r) { return r.getBoundingClientRect().bottom > line; })[0] || rows[0];
    } else {
      next = rows[Math.min(rows.length - 1, Math.max(0, i + dir))];
    }
    setCurrent(next.id);
    var link = $(".ftitle", next);
    if (link) link.focus({ preventScroll: true });
    next.scrollIntoView({ block: "nearest" });
    if (drawerOpen) openFact(next.id, { keepFocus: true });
  }

  if (table) {
    var saved = store.get(COLLAPSE_KEY);
    if (saved) {
      try { JSON.parse(saved).forEach(function (y) { setCollapsed(y, true); }); } catch (e) { /* ignore */ }
      syncCollapseAll();
    }
    document.addEventListener("click", function (evt) {
      var t = evt.target instanceof Element ? evt.target.closest("button") : null;
      if (!t) return;
      if (t.classList.contains("group-toggle")) {
        setCollapsed(t.dataset.year, !collapsed[t.dataset.year]);
        syncCollapseAll();
        updateSpy();
      } else if (t.hasAttribute("data-collapse-all")) {
        var years = $$("tbody[data-year]", table).map(function (b) { return b.dataset.year; });
        var all = years.every(function (y) { return collapsed[y]; });
        years.forEach(function (y) { setCollapsed(y, !all); });
        syncCollapseAll();
        updateSpy();
      } else if (t.dataset.jumpYear) {
        var tb = document.getElementById("y-" + t.dataset.jumpYear);
        if (!tb) return;
        tb.scrollIntoView({ block: "start", behavior: scrollBehavior() });
        var first = $("tr.fact", tb);
        if (first) setCurrent(first.id);
      } else if (t.dataset.jump) {
        jumpTo(t.dataset.jump);
      }
    });
    var spyQueued = false;
    window.addEventListener("scroll", function () {
      if (spyQueued) return;
      spyQueued = true;
      requestAnimationFrame(function () { spyQueued = false; updateSpy(); });
    }, { passive: true });
    window.addEventListener("resize", updateSpy);
    updateSpy();
    // Arriving from a fact page (#F-0012): highlight that row.
    if (/^#F-\d+$/.test(location.hash)) jumpTo(location.hash.slice(1), false);
  }

  /* ------------------------------------------------------------ Ask panel */

  var ask = document.getElementById("ask");
  var askLog = document.getElementById("ask-log");
  var askForm = document.getElementById("ask-form");
  var askInput = document.getElementById("ask-q");
  var askFab = document.getElementById("ask-open");
  var LOG_KEY = "ct-ask-log:" + location.host;
  var OPEN_KEY = "ct-ask-open";
  var intro = document.getElementById("ask-intro");
  var lastQuestion = "";

  function setAsk(open) {
    if (!ask) return;
    document.body.classList.toggle("ask-open", open);
    ask.setAttribute("aria-hidden", String(!open));
    if (askFab) askFab.setAttribute("aria-expanded", String(open));
    if (open) store.set(OPEN_KEY, "1"); else store.del(OPEN_KEY);
    if (open && askInput) askInput.focus({ preventScroll: true });
    if (!open && askFab) askFab.focus({ preventScroll: true });
  }

  function scrollLog() { if (askLog) askLog.scrollTop = askLog.scrollHeight; }

  function saveLog() {
    if (!askLog) return;
    var copy = askLog.cloneNode(true);
    $$(".pending, #ask-intro", copy).forEach(function (n) { n.remove(); });
    if (copy.children.length) store.set(LOG_KEY, copy.innerHTML); else store.del(LOG_KEY);
  }

  function autosize() {
    if (!askInput) return;
    askInput.style.height = "auto";
    askInput.style.height = Math.min(askInput.scrollHeight + 2, 160) + "px";
  }

  if (ask && askLog && askForm && askInput) {
    var restored = store.get(LOG_KEY);
    if (restored) {
      askLog.innerHTML = restored;  // our own server-rendered answers, saved below
      if (window.htmx) window.htmx.process(askLog);
    }
    if (store.get(OPEN_KEY)) {
      document.body.classList.add("ask-open");
      ask.setAttribute("aria-hidden", "false");
      if (askFab) askFab.setAttribute("aria-expanded", "true");
    }
    scrollLog();

    if (askFab) askFab.addEventListener("click", function () { setAsk(true); });
    ask.addEventListener("click", function (evt) {
      var t = evt.target instanceof Element ? evt.target.closest("button") : null;
      if (!t) return;
      if (t.hasAttribute("data-ask-close")) setAsk(false);
      else if (t.hasAttribute("data-ask-clear")) {
        askLog.textContent = "";
        if (intro) askLog.appendChild(intro);
        store.del(LOG_KEY);
        askInput.focus();
      } else if (t.hasAttribute("data-ask-example")) {
        askInput.value = t.textContent;
        if (askForm.requestSubmit) askForm.requestSubmit();
      }
    });

    askInput.addEventListener("input", autosize);
    askInput.addEventListener("keydown", function (evt) {
      if (evt.key === "Enter" && !evt.shiftKey && !evt.isComposing) {
        evt.preventDefault();
        if (askInput.value.trim() && askForm.requestSubmit) askForm.requestSubmit();
      }
    });

    askForm.addEventListener("htmx:beforeRequest", function () {
      lastQuestion = askInput.value;
      var i = document.getElementById("ask-intro");
      if (i) i.remove();
      var msg = document.createElement("div");
      msg.className = "msg me pending";
      var bubble = document.createElement("div");
      bubble.className = "bubble";
      bubble.textContent = lastQuestion.trim();
      msg.appendChild(bubble);
      askLog.appendChild(msg);
      askInput.value = "";
      autosize();
      scrollLog();
    });
    askForm.addEventListener("htmx:afterRequest", function (evt) {
      $$(".pending", askLog).forEach(function (n) { n.remove(); });
      if (!evt.detail.successful) {
        askInput.value = lastQuestion;
        autosize();
        if (!askLog.children.length && intro) askLog.appendChild(intro);
      }
    });
    askLog.addEventListener("htmx:afterSwap", function () {
      scrollLog();
      saveLog();
    });
  }

  /* ------------------------------------------------------------ keyboard */

  document.addEventListener("keydown", function (evt) {
    if (evt.key === "Escape") {
      if ($$("details.menu[open]").length) { closeMenus(); return; }
      if (document.body.classList.contains("ask-open") && ask && ask.contains(document.activeElement)) { setAsk(false); return; }
      if (drawerOpen) { closeDrawer(); return; }
      if (document.body.classList.contains("ask-open")) setAsk(false);
      return;
    }
    var t = evt.target;
    if (t instanceof Element && (t.matches("input, textarea, select") || t.isContentEditable)) return;
    if (evt.metaKey || evt.ctrlKey || evt.altKey) return;
    var search = document.getElementById("q");
    if (evt.key === "/" && search) { evt.preventDefault(); search.focus(); search.select(); }
    else if ((evt.key === "j" || evt.key === "k") && table) { evt.preventDefault(); step(evt.key === "j" ? 1 : -1); }
  });

  document.addEventListener("click", function (evt) {
    var btn = evt.target instanceof Element ? evt.target.closest("[data-scroll-to]") : null;
    if (!btn) return;
    var target = document.getElementById(btn.getAttribute("data-scroll-to"));
    if (target) target.scrollIntoView({ block: "center", behavior: scrollBehavior() });
  });
})();
