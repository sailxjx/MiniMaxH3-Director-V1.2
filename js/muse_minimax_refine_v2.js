const { app } = window.comfyAPI.app;

// Muse Minimax Refine V2 — candidate selector.
//
// Same re-skin as the plain Refine / V1.3 / V14 nodes (see js/muse_minimax_refine.js):
// the node itself takes a plain "candidate" INT widget (0-4) that decides which of the
// four wired candidate_N_latent slots actually gets refined. This file re-skins that
// INT widget as four clickable number buttons, and hides the widgets V2 exposes as
// backend inputs but never needs a human to touch (seed, steps and
// two_stage_first_pass_steps all travel embedded on the candidate latent itself;
// timeline_data isn't used by this node at all). raw_latent_carry_test is genuinely
// used by this node (see muse_minimax_refine_v2.py) and is intentionally left
// visible/toggleable — it was wrongly hidden here on 2026-09-04 and is fixed as of
// the same date.
//
// [2026-09-06] "steps" newly hidden here: this widget's own value used to silently
// win over the candidate's real total step count whenever it didn't happen to match
// what the candidate was actually generated with — confirmed to produce audible
// garbled audio right at the Stage-2 continuation seam. Now that the Python side
// always restores and uses the candidate's real total (_muse_steps_used) instead of
// this widget, leaving it visible/editable would only invite the exact same mismatch
// again for no benefit — hidden the same way seed and two_stage_first_pass_steps
// already are.

function hideWidget(w) {
  if (!w) return;
  w.hidden = true;
  if (!w.options) w.options = {};
  w.options.hidden = true;
  if (!window.LiteGraph || !window.LiteGraph.vueNodesMode) {
    w.computeSize = () => [0, -4];
    w.draw = () => {};
  }
  if (w.element) w.element.style.display = "none";
}

// The seed widget must reuse the candidate's exact original seed, but ComfyUI's
// frontend still auto-attaches a control_after_generate dropdown to any widget
// named "seed" regardless of the node's own intent. Left visible, that's a real
// trap — locking it to "fixed" and hiding it removes the failure mode instead of
// relying on people noticing.
function lockControlAfterGenerate(node) {
  const w = (node.widgets || []).find((w) => w.name === "control_after_generate");
  if (!w) return;
  w.value = "fixed";
  hideWidget(w);
}

function hideInheritedWidgets(node, nodeName) {
  if (nodeName !== "MuseMinimaxRefineV2") return;
  const inheritedOrLegacy = new Set([
    "seed", "steps", "two_stage_first_pass_steps", "timeline_data",
  ]);
  for (const widget of node.widgets || []) {
    if (inheritedOrLegacy.has(widget.name)) hideWidget(widget);
  }
}

function buildCandidateSelector(node, widget) {
  const wrap = document.createElement("div");
  wrap.style.display = "flex";
  wrap.style.gap = "6px";
  wrap.style.padding = "4px 0";
  wrap.style.justifyContent = "center";

  const label = document.createElement("div");
  label.textContent = "Candidate";
  label.style.cssText = "color:#bbb;font-size:11px;display:flex;align-items:center;margin-right:4px;";
  wrap.appendChild(label);

  const buttons = [];
  const refresh = () => {
    buttons.forEach((b, i) => {
      const active = Number(widget.value) === i + 1;
      b.style.background = active ? "#4F8EF7" : "#2a2a2a";
      b.style.color = active ? "#fff" : "#ccc";
      b.style.borderColor = active ? "#4F8EF7" : "rgba(255,255,255,0.22)";
    });
  };

  for (let i = 1; i <= 4; i++) {
    const btn = document.createElement("button");
    btn.textContent = String(i);
    btn.type = "button";
    btn.style.cssText =
      "width:28px;height:28px;border-radius:6px;border:1px solid rgba(255,255,255,0.22);" +
      "cursor:pointer;font-size:13px;font-weight:600;";
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      // Clicking the already-active candidate deselects it (back to 0/none) instead
      // of being a one-way switch — otherwise there was no way to reset the node to
      // its "not ready" state short of typing 0 into the hidden underlying widget.
      widget.value = Number(widget.value) === i ? 0 : i;
      if (widget.callback) widget.callback(widget.value);
      node.setDirtyCanvas(true, true);
      refresh();
    });
    buttons.push(btn);
    wrap.appendChild(btn);
  }

  refresh();
  return { container: wrap, refresh };
}

const REFINE_NODE_NAMES = ["MuseMinimaxRefineV2"];

app.registerExtension({
  name: "Muse.MinimaxRefineV2",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!REFINE_NODE_NAMES.includes(nodeData.name)) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

      const candidateWidget = (this.widgets || []).find((w) => w.name === "candidate");
      if (candidateWidget) {
        hideWidget(candidateWidget);
        const { container, refresh } = buildCandidateSelector(this, candidateWidget);
        this._museRefineRefresh = refresh;
        this.addDOMWidget("mmr_candidate_ui", "mmr_candidate_ui", container, {
          serialize: false,
          hideOnZoom: false,
        });
      }
      lockControlAfterGenerate(this);
      hideInheritedWidgets(this, nodeData.name);

      return r;
    };

    // onNodeCreated fires before ComfyUI applies a loaded workflow's saved
    // widgets_values onto the widgets, so the button highlight built there only
    // ever reflects the INPUT_TYPES default (0), never a real saved candidate value.
    // onConfigure fires after the real widgets_values is applied, so re-running
    // refresh() there is what makes the display match what will actually execute.
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
      if (this._museRefineRefresh) this._museRefineRefresh();
      lockControlAfterGenerate(this);
      hideInheritedWidgets(this, nodeData.name);
      return r;
    };
  },
});
