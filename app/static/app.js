// Small page behaviors shared by every page. No inline scripts (strict CSP); all text is
// inserted with textContent, never as HTML.
(function () {
  "use strict";

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

  // "Select all" checkboxes: data-check-all="<form id>" toggles that form's fact checkboxes.
  document.addEventListener("change", function (evt) {
    var box = evt.target;
    if (!(box instanceof HTMLInputElement) || !box.dataset.checkAll) return;
    var sel = 'input[type="checkbox"][name="fact_ids"][form="' + box.dataset.checkAll + '"]';
    document.querySelectorAll(sel).forEach(function (cb) { cb.checked = box.checked; });
  });

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
    var submitter = evt.submitter;
    // Bulk buttons need at least one selected draft.
    if (submitter && submitter.hasAttribute("data-needs-selection") && form.id) {
      var picked = document.querySelectorAll(
        'input[name="fact_ids"][form="' + form.id + '"]:checked'
      );
      if (!picked.length) {
        evt.preventDefault();
        toast("Select at least one draft first.");
        return;
      }
    }
    var question = form.getAttribute("data-confirm");
    if (question && !window.confirm(question)) {
      evt.preventDefault();
    }
  }, true);

  document.addEventListener("click", function (evt) {
    var btn = evt.target instanceof Element ? evt.target.closest("[data-scroll-to]") : null;
    if (!btn) return;
    var target = document.getElementById(btn.getAttribute("data-scroll-to"));
    if (target) target.scrollIntoView({ block: "center", behavior: "smooth" });
  });
})();
