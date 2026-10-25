// Source viewer: scroll the cited quote into view, and turn a text selection into stored-text
// character offsets for "Add fact" / "Add as evidence".
//
// The server renders each page as one <span class="seg" data-start=N data-end=M> whose text
// content is exactly text[N:M], so an offset inside a segment is data-start plus the length of
// the text before the selection point. The server re-derives the quote from its stored text;
// the quote we send (single-segment selections only) is a cross-check.
(function () {
  "use strict";

  var hl = document.getElementById("hl");
  if (hl) hl.scrollIntoView({ block: "center" });

  var body = document.getElementById("doc-body");
  if (!body) return;
  var hint = document.getElementById("sel-hint");
  var preview = document.getElementById("sel-preview");

  function segOf(node) {
    var el = node && node.nodeType === 3 ? node.parentNode : node;
    return el && el.closest ? el.closest(".seg") : null;
  }

  function offsetIn(seg, node, offset) {
    var r = document.createRange();
    r.setStart(seg, 0);
    r.setEnd(node, offset);
    return parseInt(seg.dataset.start, 10) + r.toString().length;
  }

  function boundarySegs(range) {
    var segs = Array.prototype.filter.call(body.querySelectorAll(".seg"), function (s) {
      return range.intersectsNode(s);
    });
    return { first: segs[0] || null, last: segs[segs.length - 1] || null };
  }

  function currentSpan() {
    var sel = window.getSelection();
    if (!sel || sel.rangeCount === 0 || sel.isCollapsed) return null;
    var range = sel.getRangeAt(0);
    if (!body.contains(range.commonAncestorContainer)) return null;
    var startSeg = segOf(range.startContainer);
    var endSeg = segOf(range.endContainer);
    var edges = null;
    var start, end;
    if (startSeg) {
      start = offsetIn(startSeg, range.startContainer, range.startOffset);
    } else {
      edges = edges || boundarySegs(range);
      if (!edges.first) return null;
      start = parseInt(edges.first.dataset.start, 10);
    }
    if (endSeg) {
      end = offsetIn(endSeg, range.endContainer, range.endOffset);
    } else {
      edges = edges || boundarySegs(range);
      if (!edges.last) return null;
      end = parseInt(edges.last.dataset.end, 10);
    }
    if (!(end > start)) return null;
    var text = range.toString();
    return {
      start: start,
      end: end,
      quote: startSeg && startSeg === endSeg ? text : "",
      preview: text,
    };
  }

  function apply(span) {
    document.querySelectorAll("[data-sel]").forEach(function (input) {
      input.value = span ? String(span[input.getAttribute("data-sel")]) : "";
    });
    document.querySelectorAll("[data-needs-span]").forEach(function (btn) {
      btn.disabled = !span;
    });
    if (preview) {
      preview.hidden = !span;
      var shown = span ? span.preview : "";
      if (shown.length > 600) shown = shown.slice(0, 600) + "…";
      preview.textContent = shown;
    }
    if (hint) {
      hint.textContent = span
        ? "Selected characters " + span.start + "–" + span.end + "."
        : "Select text in the document to quote it as evidence.";
    }
  }

  var pending = null;
  document.addEventListener("selectionchange", function () {
    clearTimeout(pending);
    pending = setTimeout(function () {
      var span = currentSpan();
      if (span) apply(span); // keep the last selection while the user types a title
    }, 120);
  });

  var evidenceForm = document.getElementById("add-evidence-form");
  if (evidenceForm) {
    evidenceForm.addEventListener("submit", function (evt) {
      var input = evidenceForm.querySelector("[data-fact-target]");
      var id = input ? input.value.trim().toUpperCase() : "";
      if (!/^F-\d+$/.test(id)) {
        evt.preventDefault();
        if (window.ctToast) window.ctToast("Enter a fact ID like F-0012.");
        return;
      }
      evidenceForm.action = "/facts/" + encodeURIComponent(id) + "/evidence";
    });
  }
})();
