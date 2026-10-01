from deep_translator import GoogleTranslator


def translate_text(text, source_language, target_language):

    if source_language == "auto":
        source_language = "auto"

    translator = GoogleTranslator(
        source=source_language,
        target=target_language
    )

    result = translator.translate(text)

    return result