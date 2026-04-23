from . import corrset, tabular

RENDERER_MAP = {
    "markdown": tabular.MarkdownTableRenderer,
    "csv": tabular.CSVTableRenderer,
    "json": tabular.JSONTableRenderer,
    "language-tabular": tabular.LanguageTableRenderer,
    "language-corrset": corrset.LanguageCorrSetRenderer,
    "corrset": corrset.PureCorrSetRenderer,
}
