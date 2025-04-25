import pytest

from src.services.text_to_speech import improved_split_sentence


class TestCodeUnderTest:
    # Splitting text shorter than text_split_length returns the original text
    def test_short_text_returns_original(self):
        text = "This is a short text"
        lang = "en"
        text_split_length = 250

        result = improved_split_sentence(text, lang, text_split_length)
        assert result == [text]

    # Empty input text
    def test_empty_text_returns_empty_list_item(self):
        text = ""
        lang = "en"
        text_split_length = 250

        result = improved_split_sentence(text, lang, text_split_length)
        assert result == [""]

    # --- PARAMETERIZED TESTS ---

    @pytest.mark.parametrize(
        "text, lang, length, expected",
        [
            # Basic sentence splitting (just under limit)
            (
                "This is sentence one. This is sentence two.",
                "en",
                50,
                ["This is sentence one.", "This is sentence two."],
            ),
            # Sentence that fits exactly into length
            (
                "A" * 100,
                "en",
                200,
                ["A" * 100],
            ),
            # Two sentences, each short, but together just too long
            (
                "Sentence one. Sentence two.",
                "en",
                30,
                ["Sentence one.", "Sentence two."],
            ),
            # Single long sentence, needs punctuation splitting
            (
                "This is a sentence, but it has, several, commas, and still continues.",
                "en",
                40,
                [
                    "This is a sentence,",
                    "but it has, several,",
                    "commas,",
                    "and still continues.",
                ],
            ),
            # Single long sentence, no punctuation,
            (
                "A" * 80,
                "en",
                40,
                ["A" * 20, "A" * 20, "A" * 20, "A" * 20],
            ),
            # Sentence with various punctuation and very tight length
            (
                "Alpha, Beta; Gamma: Delta.",
                "en",
                12,
                ["Alpha,", "Beta;", "Gamma:", "Delta."],
            ),
            # Multi-lingual (should not break, provided lang is valid for spaCy)
            (
                "Das ist der erste Satz. Dies ist der zweite Satz.",
                "de",
                50,
                ["Das ist der erste Satz.", "Dies ist der zweite Satz."],
            ),
            # Sentence with newlines, excessive spaces
            (
                "Line one.\nLine two.   Line three.",
                "en",
                32,
                ["Line one.", "\nLine two.", "Line three."],
            ),
            # Sentence with tabs and special whitespace chars
            (
                "Hello, very cruel\tworld. How are\tyou?",
                "en",
                36,
                ["Hello,", "very cruel\tworld.", "How are\tyou?"],
            ),
            # Normal sentence
            (
                'The flickering glow of your trideo projector casts shifting shadows, very dark, across your cramped NeoNET-branded cube apartment in the Ork Underground. The latest episode of Ghost Cartel: Tacoma plays at low volume, the dramatic synth-music barely covering the constant drip-drip of leaking pipes in the hallway outside. Your cyberdeck sits on the coffee table, its status LEDs pulsing like a sleeping predator. Suddenly, your commlink buzzes with an encrypted call. The caller ID flashes your fixer\'s tag - "Golden Ticket" - along with a priority marker. When you answer, the familiar voice comes through with its usual mix of amusement and urgency:',
                "en",
                200,
                [
                    "The flickering glow of your trideo projector casts shifting shadows, very dark,",
                    "across your cramped NeoNET-branded cube apartment in the Ork Underground.",
                    "The latest episode of Ghost Cartel: Tacoma plays at low volume,",
                    "the dramatic synth-music barely covering the constant drip-drip of leaking pipes in the hallway outside.",
                    "Your cyberdeck sits on the coffee table, its status LEDs pulsing like a sleeping predator.",
                    "Suddenly, your commlink buzzes with an encrypted call.",
                    'The caller ID flashes your fixer\'s tag - "Golden Ticket" - along with a priority marker.',
                    "When you answer, the familiar voice comes through with its usual mix of amusement and urgency:",
                ],
            ),
        ],
    )
    def test_various_sentences_and_lengths(self, text, lang, length, expected):
        result = improved_split_sentence(text, lang, length)
        # Remove extra whitespace from expected, as implementation lstrip's a lot
        assert [x.strip() for x in result] == [x.strip() for x in expected]

    @pytest.mark.parametrize(
        "text, expected_lang",
        [
            ("Sentence.", "en"),
            ("Satz.", "de"),
            ("Phrase.", "fr"),
        ],
    )
    def test_language_compatibility(self, text, expected_lang):
        # Should not raise
        res = improved_split_sentence(text, expected_lang, 25)
        assert isinstance(res, list)

    def test_edge_case_one_char_chunks(self):
        # Forces splitting to 1-char chunks for pathological test
        text = "abc"
        lang = "en"
        length = 1
        result = improved_split_sentence(text, lang, length)
        assert result == ["a", "b", "c"]

    def test_chunk_exactly_matches_limit_after_punct_split(self):
        # A chunk followed by comma, chunk length equals split limit
        text = "hello,world,bye,"
        lang = "en"
        length = 6  # "hello," is 6 chars
        result = improved_split_sentence(text, lang, length)
        assert result == ["hello,", "world,", "bye,"]

    def test_list_output_nonempty_even_when_whitespace(self):
        text = "  "
        lang = "en"
        length = 10
        result = improved_split_sentence(text, lang, length)
        assert result == [""]

    def test_sentence_with_many_punctuations_and_tight_split(self):
        text = "one,two;three:four;five,six"
        lang = "en"
        length = 6
        # All splits should preserve punct at end
        assert improved_split_sentence(text, lang, length) == [
            "one,",
            "two;",
            "three:",
            "four;",
            "five,",
            "six",
        ]
