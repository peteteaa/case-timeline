// Renders /api/events with vis-timeline. All event text is inserted as text, never as HTML.
(function () {
  "use strict";
  var el = document.getElementById("timeline");
  if (!el || !window.vis) return;

  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  function label(item) {
    var span = document.createElement("span");
    span.textContent = item.label;
    return span;
  }

  function tooltip(item) {
    var parts = [
      "<strong>" + esc(item.date_text) + "</strong> (" + esc(item.certainty) + ", " + esc(item.status) + ")",
      esc(item.label),
    ];
    if (item.actors && item.actors.length) parts.push("Actors: " + esc(item.actors.join(", ")));
    if (item.quote) parts.push("<em>“" + esc(item.quote) + "”</em>");
    if (item.deadline_relevant) parts.push("<strong>Needs attorney review</strong>");
    if (item.contradiction) parts.push("<strong>Involved in a contradiction</strong>");
    return parts.join("<br>");
  }

  fetch(el.dataset.src || "/api/events", { headers: { Accept: "application/json" } })
    .then(function (r) {
      if (!r.ok) throw new Error("HTTP " + r.status);
      return r.json();
    })
    .then(function (data) {
      var dated = data.items.filter(function (i) { return i.start; });
      if (!dated.length) {
        el.textContent = "No dated events to plot yet.";
        return;
      }
      var items = new vis.DataSet(
        dated.map(function (i) {
          return {
            id: i.id,
            start: i.start,
            end: i.end || undefined,
            type: i.end ? "range" : "box",
            content: label(i),
            title: tooltip(i),
            className: i.class_name,
            url: i.source_url,
          };
        })
      );
      var timeline = new vis.Timeline(el, items, {
        stack: true,
        zoomKey: "ctrlKey",
        maxHeight: "65vh",
        orientation: { axis: "top" },
        tooltip: { followMouse: true, overflowMethod: "cap" },
        selectable: true,
      });
      timeline.on("click", function (props) {
        if (!props.item) return;
        var it = items.get(props.item);
        if (it && it.url && it.url.indexOf("/source/") === 0) window.location.href = it.url;
      });
    })
    .catch(function (err) {
      el.textContent = "Could not load timeline: " + err.message;
    });
})();
