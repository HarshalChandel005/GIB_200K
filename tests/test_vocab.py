from gib import vocab


def test_extract_handles_synonyms_plurals_and_articles():
    assert vocab.extract_answer("A puppy.", "animal") == "dog"
    assert vocab.extract_answer("two dogs", "animal") == "dog"
    assert vocab.extract_answer("It is a Sofa", "piece of furniture") == "couch"
    assert vocab.extract_answer("television", "electronic device") == "tv"
    assert vocab.extract_answer("Grey", "color") == "gray"
    assert vocab.extract_answer("on top of", "spatial") == "on"
    assert vocab.extract_answer("to the left", "leftright") == "left"


def test_longest_match_wins():
    assert vocab.extract_answer("a teddy bear", "household item") == "teddy bear"
    assert vocab.extract_answer("hot dog", "food") == "hot dog"
    assert vocab.extract_answer("in front of the car", "spatial") == "in front of"


def test_off_vocabulary_is_none():
    assert vocab.extract_answer("I am not sure", "animal") is None
    assert vocab.extract_answer("", "color") is None
    assert vocab.extract_answer("a person", "animal") is None


def test_first_mention_is_the_answer():
    assert vocab.extract_answer("cat, not a dog", "animal") == "cat"


def test_chair_mentions():
    assert vocab.chair_mentions("A man riding a bike next to two dogs and a hot dog stand.") == \
        ["bicycle", "dog", "hot dog", "person"]


def test_fill_fixes_articles_and_display_names():
    assert vocab.fill("This photo shows a {ANS}.", "elephant") == "This photo shows an elephant."
    assert vocab.fill("A {ANS} on the table.", "tv") == "A television on the table."
    assert vocab.fill("The {OBJ} is {ANS}.", "red", {"OBJ": "bus"}) == "The bus is red."


def test_caption_template_swaps_plural_mentions():
    t = vocab.mention_to_slot("Two dogs playing in a park.", "dog")
    assert t == "Two {ANS_PL} playing in a park."
    assert vocab.fill(t, "cat") == "Two cats playing in a park."
    assert vocab.mention_to_slot("A cat on a couch.", "dog") is None


def test_plural_nouns_get_agreement_and_no_article():
    assert vocab.is_plural("scissors") and vocab.is_plural("jeans") and vocab.is_plural("skis")
    assert not vocab.is_plural("bus") and not vocab.is_plural("glass") and not vocab.is_plural("stop sign")
    assert vocab.fill("This photo shows a {ANS}.", "skis") == "This photo shows skis."
    assert vocab.fill("A {ANS} can be seen in this photograph.", "scissors") == "Scissors can be seen in this photograph."
    assert vocab.fill("{ANS} {OBJ} can be seen in the image.", "blue", {"OBJ": "jeans"}) == "Blue jeans can be seen in the image."
