// Visual view: renders the JSON from /cases/{id}/facts?format=json with vis-timeline.
// Points for single dates, bars for ranges, key facts highlighted, undated facts in their own
// lane. Clicking a fact opens it in the fact drawer. Text is set with textContent only.
(function () {
  "use strict";
  var el = document.getElementById("timeline");
  var lane = document.getElementById("undated-lane");
  if (!el) return;

  function openFact(item) {
    if (!(window.ctOpenFact && window.ctOpenFact(item.id))) window.location.href = item.url;
  }

  function label(item) {
    var span = document.createElement("span");
    span.textContent = item.title;
    return span;
  }

  function tooltip(item) {
    var box = document.createElement("div");
    var head = document.createElement("strong");
    head.textContent = item.id + " · " + (item.date_label || "Undated");
    box.appendChild(head);
    var title = document.createElement("div");
    title.textContent = item.title;
    box.appendChild(title);
    var flags = [];
    if (item.review === "draft") flags.push("AI draft");
    if (item.key_fact) flags.push("Key fact");
    if (item.statuses.indexOf("disputed") >= 0) flags.push("Disputed");
    if (item.conflict) flags.push("Possible conflict");
    if (item.alleged) flags.push("Alleged");
    if (item.date_text) flags.push("As written: " + item.date_text);
    if (flags.length) {
      var meta = document.createElement("div");
      meta.className = "muted";
      meta.textContent = flags.join(" · ");
      box.appendChild(meta);
    }
    return box;
  }

  function renderLane(undated) {
    if (!lane) return;
    lane.textContent = "";
    if (!undated.length) {
      var none = document.createElement("span");
      none.className = "muted small";
      none.textContent = "No undated facts.";
      lane.appendChild(none);
      return;
    }
    undated.forEach(function (item) {
      var btn = document.createElement("button");
      btn.type = "button";
      btn.className = "lane-item " + item.class_name + (item.key_fact ? " key" : "");
      btn.textContent = item.title;
      btn.title = item.id;
      btn.addEventListener("click", function () { openFact(item); });
      lane.appendChild(btn);
    });
  }

  var DAY = 24 * 3600 * 1000;
  var SPANS = { day: 21 * DAY, month: 240 * DAY, year: 6 * 365 * DAY };

  fetch(el.dataset.src, { headers: { Accept: "application/json" }, credentials: "same-origin" })
    .then(function (r) {
      if (!r.ok) throw new Error("HTTP " + r.status);
      return r.json();
    })
    .then(function (data) {
      if (data.error) {
        el.textContent = data.error;
        renderLane([]);
        return;
      }
      var dated = data.items.filter(function (i) { return !i.undated; });
      var byId = {};
      data.items.forEach(function (i) { byId[i.id] = i; });
      renderLane(data.items.filter(function (i) { return i.undated; }));
      if (!dated.length) {
        el.textContent = "No dated facts match.";
        return;
      }
      var items = new window.vis.DataSet(
        dated.map(function (i) {
          return {
            id: i.id,
            start: i.start,
            end: i.end || undefined,
            type: i.end ? "range" : "point",
            content: label(i),
            title: tooltip(i),
            className: i.class_name + (i.key_fact ? " key" : ""),
          };
        })
      );
      var timeline = new window.vis.Timeline(el, items, {
        stack: true,
        zoomKey: "ctrlKey",
        zoomMin: DAY * 3,
        zoomMax: 60 * 365 * DAY,
        maxHeight: "60vh",
        orientation: { axis: "top" },
        tooltip: { followMouse: true, overflowMethod: "cap" },
        selectable: true,
      });
      timeline.on("select", function (props) {
        if (props.items && props.items.length && byId[props.items[0]]) {
          openFact(byId[props.items[0]]);
        }
      });
      document.querySelectorAll("[data-zoom]").forEach(function (btn) {
        btn.addEventListener("click", function () {
          var kind = btn.getAttribute("data-zoom");
          if (kind === "fit") { timeline.fit(); return; }
          var w = timeline.getWindow();
          var mid = (w.start.valueOf() + w.end.valueOf()) / 2;
          var half = SPANS[kind] / 2;
          timeline.setWindow(new Date(mid - half), new Date(mid + half));
        });
      });
    })
    .catch(function (err) {
      el.textContent = "Could not load the timeline: " + err.message;
    });
})();
