# timesnet_head_custom.py

"""TimesNet forecasting head imported into the transformer popular registry."""

from __future__ import annotations

from foreblocks.blocks.popular.timesnet import TimesNetHeadCustom as _TimesNetHeadCustom


class TimesNetHeadCustom(_TimesNetHeadCustom):
    """TimesNet head exposed through the transformer.popular package."""

    pass
