// Scroll the highlighted quote into view on load.
(function () {
  "use strict";
  var hl = document.getElementById("hl");
  if (hl) hl.scrollIntoView({ block: "center" });
})();
