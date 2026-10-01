import os
import re

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, request


load_dotenv()


app = Flask(__name__)


DEEPL_API_KEY = os.getenv("DEEPL_API_KEY")

DEEPL_URL = "https://api-free.deepl.com/v2/translate"


# Maximum characters sent in one translation chunk.
# Keeping this comfortably below large-request limits
# makes long text requests more reliable.
MAX_CHUNK_CHARS = 4500


@app.route("/")
def home():

    return jsonify({
        "project": "LinguaAI",
        "message": "Python backend is running!"
    })


@app.route("/health")
def health():

    return jsonify({
        "status": "ok",
        "service": "LinguaAI Python API"
    })


# =========================================================
# TEXT CHUNKING
# =========================================================

def hard_split_text(text, max_chars):
    """
    Split very long text at whitespace when possible.

    If a single word is longer than max_chars,
    it will be split directly.
    """

    chunks = []

    current = ""

    # Keep whitespace while splitting.
    parts = re.split(r"(\s+)", text)

    for part in parts:

        if not part:
            continue


        # Normal case:
        # The new part still fits in the current chunk.
        if len(current) + len(part) <= max_chars:

            current += part

            continue


        # Save current chunk first.
        if current.strip():

            chunks.append(
                current.strip()
            )


        # If the individual part itself is too large,
        # split it directly.
        if len(part) > max_chars:

            start = 0

            while start < len(part):

                end = start + max_chars

                piece = part[start:end]

                if piece.strip():

                    chunks.append(
                        piece.strip()
                    )

                start = end


            current = ""

        else:

            current = part


    if current.strip():

        chunks.append(
            current.strip()
        )


    return chunks


def split_text_into_chunks(text, max_chars=MAX_CHUNK_CHARS):
    """
    Split large text while trying to preserve sentence boundaries.

    Order of preference:
    1. Keep the complete text when it is small.
    2. Split by paragraphs.
    3. Split large paragraphs by sentences.
    4. Split extremely long sentences by words.
    """

    if len(text) <= max_chars:

        return [text]


    chunks = []


    # Normalize Windows line endings.
    normalized_text = text.replace(
        "\r\n",
        "\n"
    )


    normalized_text = normalized_text.replace(
        "\r",
        "\n"
    )


    # Split into paragraphs while keeping the
    # paragraph separator available.
    paragraph_parts = re.split(
        r"(\n\s*\n)",
        normalized_text
    )


    current_chunk = ""


    for part in paragraph_parts:

        if not part:
            continue


        # Paragraph separator
        if part.strip() == "":

            if (
                current_chunk
                and
                len(current_chunk) + len(part)
                <= max_chars
            ):

                current_chunk += part

            continue


        # Small paragraph
        if len(part) <= max_chars:

            if (
                current_chunk
                and
                len(current_chunk) + len(part)
                <= max_chars
            ):

                current_chunk += "\n\n" + part

            else:

                if current_chunk.strip():

                    chunks.append(
                        current_chunk.strip()
                    )

                current_chunk = part

            continue


        # The paragraph is too large.
        # Split it into sentences.
        sentences = re.split(
            r"(?<=[.!?。！？])\s+",
            part
        )


        for sentence in sentences:

            sentence = sentence.strip()


            if not sentence:
                continue


            # Extremely long sentence.
            if len(sentence) > max_chars:

                if current_chunk.strip():

                    chunks.append(
                        current_chunk.strip()
                    )

                    current_chunk = ""


                long_chunks = hard_split_text(
                    sentence,
                    max_chars
                )


                chunks.extend(
                    long_chunks
                )

                continue


            # Sentence fits current chunk.
            if (
                current_chunk
                and
                len(current_chunk)
                + 1
                + len(sentence)
                <= max_chars
            ):

                current_chunk += " " + sentence


            else:

                if current_chunk.strip():

                    chunks.append(
                        current_chunk.strip()
                    )


                current_chunk = sentence


    if current_chunk.strip():

        chunks.append(
            current_chunk.strip()
        )


    return chunks


# =========================================================
# TRANSLATION
# =========================================================

@app.route("/translate", methods=["POST"])
def translate():

    data = request.get_json(
        silent=True
    ) or {}


    text = data.get(
        "text",
        ""
    ).strip()


    source_language = data.get(
        "source_language",
        "auto"
    )


    target_language = data.get(
        "target_language",
        "en"
    )


    # -----------------------------------------------------
    # Validation
    # -----------------------------------------------------

    if not text:

        return jsonify({
            "error": "Text cannot be empty."
        }), 400


    if not DEEPL_API_KEY:

        return jsonify({
            "error":
                "DeepL API key is not configured."
        }), 500


    # DeepL uses uppercase language codes.
    target_language =target_language.upper()


    # -----------------------------------------------------
    # Create chunks
    # -----------------------------------------------------

    chunks = split_text_into_chunks(
        text
    )


    # -----------------------------------------------------
    # DeepL headers
    # -----------------------------------------------------

    headers = {

        "Authorization":
            f"DeepL-Auth-Key {DEEPL_API_KEY}",

        "Content-Type":
            "application/json"

    }


    translated_chunks = []

    detected_language = None


    try:

        # -------------------------------------------------
        # Translate every chunk
        # -------------------------------------------------

        for index, chunk in enumerate(
            chunks,
            start=1
        ):

            payload = {

                "text": [
                    chunk
                ],

                "target_lang":
                    target_language

            }


            # Add source language only when
            # it is explicitly selected.
            if (
                source_language
                and
                source_language.lower()
                != "auto"
            ):

                payload["source_lang"] =source_language.upper()


            api_response = requests.post(

                DEEPL_URL,

                headers=headers,

                json=payload,

                timeout=60

            )


            # -------------------------------------------------
            # DeepL error
            # -------------------------------------------------

            if not api_response.ok:

                return jsonify({

                    "error":
                        "DeepL translation failed.",

                    "chunk":
                        index,

                    "total_chunks":
                        len(chunks),

                    "details":
                        api_response.text

                }), api_response.status_code


            result = api_response.json()


            translations =result.get(
                    "translations",
                    []
                )


            if not translations:

                return jsonify({

                    "error":
                        "DeepL returned no translation.",

                    "chunk":
                        index,

                    "total_chunks":
                        len(chunks)

                }), 502


            translated_text =translations[0].get(
                    "text",
                    ""
                )


            if not translated_text:

                return jsonify({

                    "error":
                        "DeepL returned an empty translation.",

                    "chunk":
                        index,

                    "total_chunks":
                        len(chunks)

                }), 502


            translated_chunks.append(
                translated_text
            )


            # Save detected source language
            # from the first available chunk.
            if not detected_language:

                detected_language =translations[0].get(
                        "detected_source_language"
                    )


        # -------------------------------------------------
        # Join all translations
        # -------------------------------------------------

        final_translation = "\n\n".join(
            translated_chunks
        )


        return jsonify({

            "original_text":
                text,

            "source_language":
                detected_language
                or source_language,

            "target_language":
                target_language,

            "translation":
                final_translation,

            "provider":
                "DeepL",

            "chunks":
                len(chunks),

            "chunked":
                len(chunks) > 1

        })


    except requests.RequestException as error:

        return jsonify({

            "error":
                "Could not connect to DeepL.",

            "details":
                str(error)

        }), 502


# =========================================================
# LANGUAGES
# =========================================================

@app.route("/languages", methods=["GET"])
def languages():

    if not DEEPL_API_KEY:

        return jsonify({
            "error":
                "DeepL API key is not configured."
        }), 500


    headers = {

        "Authorization":
            f"DeepL-Auth-Key {DEEPL_API_KEY}"

    }


    try:

        deepl_response = requests.get(

            "https://api-free.deepl.com/v3/languages",

            params={
                "resource":
                    "translate_text"
            },

            headers=headers,

            timeout=30

        )


        if not deepl_response.ok:

            return jsonify({

                "error":
                    "Unable to retrieve DeepL languages.",

                "status_code":
                    deepl_response.status_code,

                "details":
                    deepl_response.text

            }), deepl_response.status_code


        deep_l_languages =deepl_response.json()


        languages = []


        for language in deep_l_languages:

            code =language.get("lang")


            name =language.get("name")


            if not code or not name:

                continue


            languages.append({

                "code":
                    code,

                "name":
                    name,

                "source":
                    language.get(
                        "usable_as_source",
                        False
                    ),

                "target":
                    language.get(
                        "usable_as_target",
                        False
                    )

            })


        languages.sort(
            key=lambda item:
                item["name"].lower()
        )


        return jsonify({

            "languages":
                languages

        })


    except requests.RequestException as error:

        return jsonify({

            "error":
                "Could not connect to DeepL.",

            "details":
                str(error)

        }), 502


# =========================================================
# START SERVER
# =========================================================

# if __name__ == "__main__":

#     app.run(

#         host="127.0.0.1",

#         port=5000,

#         debug=True

#     )

# =========================================================
# For deployment
# =========================================================
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )