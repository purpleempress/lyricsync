"""LyricSync — forced-align song audio to known lyrics.

Pipeline: parse unsynced TTML (ttml_in) -> forced alignment (align) ->
internal datamodel (model) -> word-level TTML (ttml_out) / spicy-lyrics JSON
(spicy_json).
"""

__version__ = "0.1.0"
