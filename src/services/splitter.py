import re
import textwrap

from TTS.tts.layers.xtts.tokenizer import get_spacy_lang


def improved_split_sentence(text, lang, text_split_length=250):
    """
    Split text for TTS with nuanced chunk management.
    - Tries to keep chunks at or below text_split_length/2 (target)
    - Never appends NLP sentence or punctuation-split chunks past the target;
        start new chunk instead.
    - If a chunk must be forcibly split by textwrap (last resort), only allow
        artificial chunks to be appended together
        (past target, up to hard max) if the previous chunk was ALSO due to textwrap.

    By appending the sentences and punctuation splits only up to text_split_length/2
    leaves us slack to append textwrap chunks together and avoid unnatural split
    of sentences

    Args:
        text (str): The input text.
        lang (str): Language code for NLP processing.
        text_split_length (int): Max length of each split chunk.

    Returns:
        List[str]: The resulting split strings.
    """

    max_text_split_length = text_split_length
    target_text_split_length = int(max_text_split_length / 2)

    def split_on_punct(sentence, puncts=",;:"):
        parts = re.split(f"([{re.escape(puncts)}])", sentence)
        combined = []
        buf = ""
        for part in parts:
            buf += part
            if part in puncts:
                combined.append((buf.strip(), "punct"))
                buf = ""
        if buf.strip():
            combined.append((buf.strip(), "punct"))
        return combined

    # Maintain two parallel lists: one for text, one for chunk types
    text_splits = []
    split_types = []

    if len(text) >= target_text_split_length:
        nlp = get_spacy_lang(lang)
        nlp.add_pipe("sentencizer")
        doc = nlp(text)
        units = [str(sentence) for sentence in doc.sents]

        # Each element in pending_chunks is a tuple: (text, chunk_type)
        pending_chunks = []
        for unit in units:
            # If a sentence/unit is bigger than soft max, split further.
            if len(unit) > target_text_split_length:
                # Try punctuation split before hard wrapping
                punct_chunks = split_on_punct(unit)
                for chunk, _ in punct_chunks:
                    if len(chunk) > target_text_split_length:
                        # Really long, hard-wrap
                        for forced in textwrap.wrap(
                            chunk,
                            width=target_text_split_length,
                            drop_whitespace=True,
                            break_on_hyphens=False,
                            tabsize=1,
                        ):
                            pending_chunks.append((forced, "textwrap"))
                    else:
                        pending_chunks.append((chunk, "punct"))
            else:
                pending_chunks.append((unit, "nlp"))

        for chunk, chunk_type in pending_chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
            if not text_splits:  # First chunk
                text_splits.append(chunk)
                split_types.append(chunk_type)
                continue

            current = text_splits[-1]
            current_type = split_types[-1]

            if len(current) < target_text_split_length:
                if (len(current) + 1 + len(chunk) <= target_text_split_length) or (
                    chunk_type == "textwrap"
                    and current_type == "textwrap"
                    and len(current) + 1 + len(chunk) <= max_text_split_length
                ):
                    text_splits[-1] = (current + " " + chunk).strip()
                    split_types[-1] = (
                        "textwrap"
                        if current_type == "textwrap" and chunk_type == "textwrap"
                        else current_type
                    )
                else:
                    text_splits.append(chunk)
                    split_types.append(chunk_type)
            else:
                # Only allow artificial (textwrap) chunks to be appended beyond target, up to hard max,
                # if BOTH current and incoming are textwrap
                if (
                    chunk_type == "textwrap"
                    and current_type == "textwrap"
                    and len(current) + 1 + len(chunk) <= max_text_split_length
                ):
                    text_splits[-1] = (current + " " + chunk).strip()
                    split_types[-1] = "textwrap"
                else:
                    text_splits.append(chunk)
                    split_types.append(chunk_type)
    else:
        text_splits = [text.strip()]

    text_splits = [chunk.rstrip('.,;:-)"') for chunk in text_splits]
    for text_split in text_splits:
        print(text_split)

    return text_splits
