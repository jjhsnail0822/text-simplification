from src.level_assessment import LevelAssessor

class UnknownFinder(LevelAssessor):
    def get_unknown_tokens(self, output, level, lang):
        level = self.LEVEL_CONVERT[lang][level]
        doc = self._get_docs_cached([output], [lang])[0]
        counts, unknown_counts = self._counts_from_doc(doc, lang)
        return list(unknown_counts.keys())