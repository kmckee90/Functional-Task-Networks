"""
tasks.py — Mutually-exclusive style-transformation tasks for the LoRA-Mux
LLM experiment.

Every task is a deterministic, learnable text-to-text transformation applied
to a SHARED corpus of source sentences.  Same input -> different output across
tasks => strict concept shift, the regime where regularization-based CL fails.

Why these tasks are gradient-discriminable:
- Each transform requires a distinct token-level pattern (case change, vowel
  removal, repetition, reversal, fixed-prefix wrapping, etc.), so the loss
  surface for a fresh batch from task k vs.\ task j has a clearly different
  gradient w.r.t. the LoRA-mask logits.  The mask configurer's gradient
  proposal exploits exactly this.
"""
from __future__ import annotations

import random
from typing import Callable, List, Tuple


# ---------------------------------------------------------------------------
# Style transformations
# ---------------------------------------------------------------------------


def t_upper(s: str) -> str:
    return s.upper()


def t_lower(s: str) -> str:
    return s.lower()


def t_reverse_words(s: str) -> str:
    return " ".join(s.split()[::-1])


def t_double_words(s: str) -> str:
    return " ".join(w + " " + w for w in s.split())


def t_starred_vowels(s: str) -> str:
    out = []
    for ch in s:
        out.append("*" if ch.lower() in "aeiou" else ch)
    return "".join(out)


def t_pirate(s: str) -> str:
    return f"Arr! {s} matey!"


def t_bracketed(s: str) -> str:
    return "[" + s.strip().rstrip(".!?") + "]"


def t_question(s: str) -> str:
    base = s.strip().rstrip(".!?")
    return base + ", really?"


# Headline list of mutually-exclusive style regimes.  Pick `n_tasks` from this
# list in order; default uses the first 6.
ALL_STYLES: List[Tuple[str, Callable[[str], str]]] = [
    ("upper",         t_upper),
    ("lower",         t_lower),
    ("reverse_words", t_reverse_words),
    ("double_words",  t_double_words),
    ("starred_vowels", t_starred_vowels),
    ("pirate",        t_pirate),
    ("bracketed",     t_bracketed),
    ("question",      t_question),
]


# ---------------------------------------------------------------------------
# Source corpus — short, mixed-case English sentences.  Embedded so the
# experiment runs without external data downloads.
# ---------------------------------------------------------------------------


SOURCE_CORPUS: List[str] = [
    "The quick brown fox jumps over the lazy dog.",
    "She sells seashells by the seashore.",
    "How vexingly quick daft zebras jump.",
    "Pack my box with five dozen liquor jugs.",
    "Sphinx of black quartz, judge my vow.",
    "The five boxing wizards jump quickly.",
    "Crazy Fredrick bought many very exquisite opal jewels.",
    "Bright vixens jump dozy fowl quack.",
    "A quick movement of the enemy will jeopardize six gunboats.",
    "Heavy boxes perform quick waltzes and jigs.",
    "The early bird catches the worm.",
    "An apple a day keeps the doctor away.",
    "Better late than never.",
    "Birds of a feather flock together.",
    "Curiosity killed the cat.",
    "Don't count your chickens before they hatch.",
    "Every cloud has a silver lining.",
    "Fortune favors the bold.",
    "Good things come to those who wait.",
    "Honesty is the best policy.",
    "If at first you don't succeed, try again.",
    "Knowledge is power.",
    "Look before you leap.",
    "Money does not grow on trees.",
    "Necessity is the mother of invention.",
    "Old habits die hard.",
    "Practice makes perfect.",
    "Rome was not built in a day.",
    "Slow and steady wins the race.",
    "The pen is mightier than the sword.",
    "Time flies when you are having fun.",
    "Two heads are better than one.",
    "Where there is smoke there is fire.",
    "You cannot judge a book by its cover.",
    "Actions speak louder than words.",
    "Beauty is in the eye of the beholder.",
    "Cleanliness is next to godliness.",
    "Don't put all your eggs in one basket.",
    "Even a broken clock is right twice a day.",
    "Familiarity breeds contempt.",
    "Great minds think alike.",
    "Hope springs eternal.",
    "Ignorance is bliss.",
    "Justice delayed is justice denied.",
    "Keep your friends close and your enemies closer.",
    "Laughter is the best medicine.",
    "Many hands make light work.",
    "No pain, no gain.",
    "Once bitten, twice shy.",
    "Patience is a virtue.",
    "Quality matters more than quantity.",
    "Rome was a city of seven hills.",
    "Strike while the iron is hot.",
    "The grass is always greener on the other side.",
    "United we stand, divided we fall.",
    "Variety is the spice of life.",
    "When in Rome, do as the Romans do.",
    "You reap what you sow.",
    "Zealous students study hard for exams.",
    "Albert Einstein developed the theory of relativity.",
    "Marie Curie won two Nobel Prizes in different sciences.",
    "Isaac Newton formulated the laws of motion.",
    "Charles Darwin proposed the theory of natural selection.",
    "Galileo Galilei improved the telescope and observed the heavens.",
    "Ada Lovelace wrote the first computer algorithm.",
    "Alan Turing helped break the Enigma code during the war.",
    "Grace Hopper invented an early high-level programming language.",
    "Tim Berners-Lee created the World Wide Web at CERN.",
    "Linus Torvalds released the first Linux kernel in 1991.",
    "Mountains rise slowly when continental plates collide.",
    "The Pacific Ocean covers about a third of the globe.",
    "Whales are mammals despite living entirely in the ocean.",
    "Bees pollinate roughly a third of the food we eat.",
    "Photosynthesis turns sunlight into chemical energy.",
    "Black holes warp spacetime so much that light cannot escape.",
    "The Andromeda galaxy is on a collision course with the Milky Way.",
    "Volcanoes have shaped the surface of the Earth for billions of years.",
    "Coral reefs host an extraordinary diversity of marine life.",
    "Glaciers store roughly seventy percent of the fresh water on Earth.",
    "Tokyo is the largest metropolitan area in the world.",
    "Paris is famous for the Eiffel Tower and its art museums.",
    "The Amazon rainforest spans nine countries in South America.",
    "Mount Everest is the highest peak above sea level.",
    "The Sahara is the largest hot desert on the planet.",
    "Antarctica is the coldest and driest continent.",
    "The Nile is among the longest rivers in the world.",
    "Iceland sits on the boundary of two tectonic plates.",
    "Venice is built on more than a hundred small islands.",
    "Cairo is home to the ancient pyramids of Giza.",
    "The library was quiet except for the soft turn of pages.",
    "She found an old letter tucked inside a forgotten book.",
    "He whistled a tune as he walked down the cobbled street.",
    "Lightning split the sky and thunder followed close behind.",
    "The aroma of fresh bread drifted through the open window.",
    "She painted the kitchen a cheerful shade of yellow.",
    "An owl called from the woods just after midnight.",
    "He dropped the camera but it landed safely on the rug.",
    "The garden bloomed with tulips, daffodils, and roses.",
    "They danced under the streetlights until the rain returned.",
    "The musician tuned her violin behind the heavy red curtain.",
    "A thin layer of frost covered the windshield every morning.",
    "The chef seasoned the soup and tasted it before serving.",
    "The detective questioned every guest at the country house.",
    "She solved the puzzle minutes before the timer expired.",
    "The astronaut floated weightlessly inside the spacecraft.",
    "He read the same paragraph three times before it made sense.",
    "Her laughter echoed across the empty stone courtyard.",
    "The cat curled up on the warm laptop keyboard again.",
    "They built a small wooden boat for the summer regatta.",
    "The committee approved the proposal after lengthy debate.",
    "A robin pecked at the breadcrumbs on the snowy windowsill.",
    "The old map showed a path that no longer exists.",
    "He carved his initials into the bark of an ancient oak.",
    "She watered the basil and parsley on the kitchen sill.",
    "The orchestra paused, then began the second movement softly.",
    "He kept all the postcards his grandmother had sent him.",
    "The river flooded its banks after the unusually heavy rain.",
    "She climbed to the top of the lighthouse for the view.",
    "Lessons in patience are often taught by long lines.",
    "The clock tower chimed nine times across the village square.",
]


def make_pairs(
    style_name: str,
    transform: Callable[[str], str],
    sentences: List[str],
) -> List[Tuple[str, str]]:
    """Apply transform to every source sentence; return (source, target) pairs."""
    pairs: List[Tuple[str, str]] = []
    for s in sentences:
        pairs.append((s, transform(s)))
    return pairs


# ---------------------------------------------------------------------------
# Task suite
# ---------------------------------------------------------------------------


class StyleTaskSuite:
    """
    n_tasks distinct style regimes.  Each task is (source_sentences -> styled).
    The source corpus is shared, so input distribution is identical across
    tasks and only the target text differs (concept shift).

    Args:
        n_tasks:    number of styles to use (1..len(ALL_STYLES))
        seed:       reproducibility for the train/test split
        train_frac: fraction of the corpus used for training (rest is held-out)
    """

    def __init__(
        self,
        n_tasks: int = 6,
        seed: int = 0,
        train_frac: float = 0.8,
    ) -> None:
        assert 1 <= n_tasks <= len(ALL_STYLES), f"need 1..{len(ALL_STYLES)} tasks"
        self.n_tasks = n_tasks
        self.styles = ALL_STYLES[:n_tasks]

        rng = random.Random(seed)
        idx = list(range(len(SOURCE_CORPUS)))
        rng.shuffle(idx)
        n_train = int(len(idx) * train_frac)
        train_ids = sorted(idx[:n_train])
        test_ids = sorted(idx[n_train:])
        self.train_sentences = [SOURCE_CORPUS[i] for i in train_ids]
        self.test_sentences = [SOURCE_CORPUS[i] for i in test_ids]

        # (style_name, transform) -> (train_pairs, test_pairs)
        self._train: List[List[Tuple[str, str]]] = [
            make_pairs(name, fn, self.train_sentences) for name, fn in self.styles
        ]
        self._test: List[List[Tuple[str, str]]] = [
            make_pairs(name, fn, self.test_sentences) for name, fn in self.styles
        ]

    def style_names(self) -> List[str]:
        return [name for name, _ in self.styles]

    def sample_pairs(
        self,
        task_idx: int,
        batch_size: int,
        rng: random.Random,
        split: str = "train",
    ) -> List[Tuple[str, str]]:
        pool = self._train[task_idx] if split == "train" else self._test[task_idx]
        return [rng.choice(pool) for _ in range(batch_size)]

    def all_pairs(self, task_idx: int, split: str = "test") -> List[Tuple[str, str]]:
        return self._test[task_idx] if split == "test" else self._train[task_idx]
