from .muse_minimax_hybrid_patch import install_hybrid_payload_patch

# Preserve FL2VA keyframes and Ref2VA media in one packed H3 payload.
install_hybrid_payload_patch()

from .muse_minimax_director import (
    NODE_CLASS_MAPPINGS as _DIRECTOR_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as _DIRECTOR_NODE_DISPLAY_NAME_MAPPINGS,
)
from .muse_minimax_refine_v14 import (
    NODE_CLASS_MAPPINGS as _REFINE_V14_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as _REFINE_V14_NODE_DISPLAY_NAME_MAPPINGS,
)

# 2026-09-04: retired the old plain Refine, Refine V1.3, Refine V2, and
# Muse Model Route bundle — each of those was built as a companion to a
# specific earlier Director variant (plain Refine for the original V1.2,
# V1.3 for its own First/Last Frame + Hybrid work, V2 for the separate
# TwoStage-Beta package's own scouting/continuity mechanism) that this
# release replaces outright, not extends. Refine V14 is V1.4's own matched
# companion — same "additive, not replacement" relationship the old bundle
# had internally, just with the whole old family retired together rather
# than left orphaned alongside a Director version nobody's shipping anymore.
NODE_CLASS_MAPPINGS = {
    **_DIRECTOR_NODE_CLASS_MAPPINGS,
    **_REFINE_V14_NODE_CLASS_MAPPINGS,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    **_DIRECTOR_NODE_DISPLAY_NAME_MAPPINGS,
    **_REFINE_V14_NODE_DISPLAY_NAME_MAPPINGS,
}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
