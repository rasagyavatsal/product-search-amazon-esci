from esci.preprocess import analyze, clean, tokenize


def test_clean_strips_html_and_punctuation():
    assert clean("<b>Apple</b> iPhone&nbsp;13 &amp; Case!") == "apple iphone 13 case"


def test_clean_handles_empty():
    assert clean("") == ""
    assert clean(None) == ""


def test_tokenize_removes_stopwords():
    assert tokenize("a case for the iphone") == ["case", "iphone"]


def test_analyze_stems():
    assert analyze("Running shoes for women") == ["run", "shoe", "women"]
