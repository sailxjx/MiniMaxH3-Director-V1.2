from .muse_minimax_hybrid_patch import install_hybrid_payload_patch

# Preserve FL2VA keyframes and Ref2VA media in one packed H3 payload.
install_hybrid_payload_patch()

from .muse_minimax_director import (
    NODE_CLASS_MAPPINGS as _DIRECTOR_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as _DIRECTOR_NODE_DISPLAY_NAME_MAPPINGS,
)
from .muse_minimax_refine_v2 import (
    NODE_CLASS_MAPPINGS as _REFINE_V2_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as _REFINE_V2_NODE_DISPLAY_NAME_MAPPINGS,
)

# [2026-09-05] Refine V2 (Beta-matched) is the one supported Refine node in
# this repo going forward. Plain Refine, Refine V14 and Muse Model Route have
# been removed from this package — V2 is the node this Director's own
# two-stage/Seed Hunt scouting path actually matches, and keeping only one
# Refine node avoids anyone picking the wrong one for their scouted candidate.
NODE_CLASS_MAPPINGS = {
    **_DIRECTOR_NODE_CLASS_MAPPINGS,
    **_REFINE_V2_NODE_CLASS_MAPPINGS,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    **_DIRECTOR_NODE_DISPLAY_NAME_MAPPINGS,
    **_REFINE_V2_NODE_DISPLAY_NAME_MAPPINGS,
}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
