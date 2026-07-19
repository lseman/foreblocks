"""foreblocks.models.transformer.

Re-exports the ModernTransformerTuner for auto-hyperparameter selection.

"""

from foreblocks.models.transformer.tuner import (
    ModernTransformerTuner,
    TransformerTuner,
)
from foreblocks.models.transformer.runtime.outputs import (
    TransformerDecoderOutput,
    TransformerEncoderOutput,
    TransformerGenerationOutput,
)
from foreblocks.models.transformer.runtime.residual_state import (
    append_attention_residual_update,
    attention_residual_input,
    attention_residual_values,
    init_attention_residual_state,
)
from foreblocks.models.transformer.runtime.routing import (
    gateskip_active_mask_from_padding,
    gather_padding_mask,
    gather_query_mask,
    gather_sequence_tokens,
    gather_square_mask,
    patchify_gateskip_active_mask,
    scatter_mixture_of_depths_output,
)

__all__ = [
    "ModernTransformerTuner",
    "TransformerDecoderOutput",
    "TransformerEncoderOutput",
    "TransformerGenerationOutput",
    "TransformerTuner",
    "append_attention_residual_update",
    "attention_residual_input",
    "attention_residual_values",
    "gateskip_active_mask_from_padding",
    "gather_padding_mask",
    "gather_query_mask",
    "gather_sequence_tokens",
    "gather_square_mask",
    "init_attention_residual_state",
    "patchify_gateskip_active_mask",
    "scatter_mixture_of_depths_output",
]
