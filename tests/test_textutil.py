from vox.config import tomllib, DEFAULT_TOML
from vox.textutil import SentenceSplitter, clean_for_speech, guess_lang, is_noise, split_wake

CFG = tomllib.loads(DEFAULT_TOML)
PHRASES = [CFG["wake"]["phrase"], *CFG["wake"]["aliases"]]
TH = CFG["wake"]["sensitivity"]


def test_wake_variants_match():
    for heard in ["Hey Claude.", "Hey, Claude!", "hej Claude", "Hey Cloud", "Hej klaud,", "Hey Clod", "Okay, hey Claude",
                  "Hey Claud", "HeyClaude", "Hej, Klaud."]:
        assert split_wake(heard, PHRASES, TH)[0], heard


def test_wake_rejects_other_speech():
    for heard in ["Hey Clara", "What's the weather", "Klaudia, chodź tu", "Hello", "Hey", "Claude Monet was a painter",
                  "Dzień dobry", "Hej, co tam?", "Okay cool"]:
        assert not split_wake(heard, PHRASES, TH)[0], heard


def test_wake_remainder():
    assert split_wake("Hey Claude, what time is it?", PHRASES, TH) == (True, "what time is it?")
    assert split_wake("Hej Klaud, jaka jest pogoda jutro?", PHRASES, TH) == (True, "jaka jest pogoda jutro?")
    assert split_wake("Hey Claude.", PHRASES, TH) == (True, "")


def test_sentence_splitter_streaming():
    s = SentenceSplitter()
    out = []
    for chunk in ["Jutro w Poznaniu będzie", " słonecznie, ok. 18 stopni. Weź np. kurtkę", " na wieczór! A w", " niedzielę deszcz."]:
        out += s.feed(chunk)
    out += s.flush()
    assert out == ["Jutro w Poznaniu będzie słonecznie, ok. 18 stopni.", "Weź np. kurtkę na wieczór!",
                   "A w niedzielę deszcz."]


def test_sentence_splitter_numbers_and_short():
    s = SentenceSplitter()
    out = s.feed("Tak. To jest 3. wersja aplikacji. Dalej ") + s.flush()
    assert out == ["Tak. To jest 3. wersja aplikacji.", "Dalej"]


def test_clean_for_speech():
    assert clean_for_speech("**Bold** and `code` 🚀 see [docs](https://x.y)") == "Bold and code see docs"
    assert clean_for_speech("- one\n- two") == "one two"


def test_guess_lang():
    assert guess_lang("Jutro będzie ładnie.", "en") == "pl"
    assert guess_lang("It is going to be sunny and warm.", "pl") == "en"
    assert guess_lang("OK.", "pl") == "pl"


def test_noise():
    assert is_noise("Dziękuję.") and is_noise("Thank you.") and is_noise("  ")
    assert is_noise("Napisy wykonane przez społeczność Amara.org")
    assert not is_noise("Dziękuję za pomoc, a jaka będzie jutro pogoda?")
