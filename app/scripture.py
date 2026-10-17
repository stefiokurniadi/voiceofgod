"""Verse reference detection.

A book name plus chapter:verse is a citation. A long tail after a citation is
treated as a scripture reading and is not shown to listeners. Uncertain text
is left alone.
"""

from __future__ import annotations

from dataclasses import dataclass
import re


BOOKS: list[tuple[tuple[str, ...], str, str, bool]] = [
    (("Genesis", "Kejadian"), "Genesis", "Kejadian", False),
    (("Exodus", "Keluaran"), "Exodus", "Keluaran", False),
    (("Leviticus", "Imamat"), "Leviticus", "Imamat", False),
    (("Numbers", "Bilangan"), "Numbers", "Bilangan", False),
    (("Deuteronomy", "Ulangan"), "Deuteronomy", "Ulangan", False),
    (("Joshua", "Yosua"), "Joshua", "Yosua", False),
    (("Judges", "Hakim-hakim"), "Judges", "Hakim-hakim", False),
    (("Ruth", "Rut"), "Ruth", "Rut", False),
    (("Samuel",), "Samuel", "Samuel", True),
    (("Kings", "Raja-raja"), "Kings", "Raja-raja", True),
    (("Chronicles", "Tawarikh"), "Chronicles", "Tawarikh", True),
    (("Ezra",), "Ezra", "Ezra", False),
    (("Nehemiah", "Nehemia"), "Nehemiah", "Nehemia", False),
    (("Esther", "Ester"), "Esther", "Ester", False),
    (("Job", "Ayub"), "Job", "Ayub", False),
    (("Psalms", "Psalm", "Mazmur"), "Psalms", "Mazmur", False),
    (("Proverbs", "Amsal"), "Proverbs", "Amsal", False),
    (("Ecclesiastes", "Pengkhotbah"), "Ecclesiastes", "Pengkhotbah", False),
    (("Song of Solomon", "Song of Songs", "Kidung Agung"), "Song of Songs", "Kidung Agung", False),
    (("Isaiah", "Yesaya"), "Isaiah", "Yesaya", False),
    (("Jeremiah", "Yeremia"), "Jeremiah", "Yeremia", False),
    (("Lamentations", "Ratapan"), "Lamentations", "Ratapan", False),
    (("Ezekiel", "Yehezkiel"), "Ezekiel", "Yehezkiel", False),
    (("Daniel",), "Daniel", "Daniel", False),
    (("Hosea",), "Hosea", "Hosea", False),
    (("Joel", "Yoel"), "Joel", "Yoel", False),
    (("Amos",), "Amos", "Amos", False),
    (("Obadiah", "Obaja"), "Obadiah", "Obaja", False),
    (("Jonah", "Yunus"), "Jonah", "Yunus", False),
    (("Micah", "Mikha"), "Micah", "Mikha", False),
    (("Nahum",), "Nahum", "Nahum", False),
    (("Habakkuk", "Habakuk"), "Habakkuk", "Habakuk", False),
    (("Zephaniah", "Zefanya"), "Zephaniah", "Zefanya", False),
    (("Haggai", "Hagai"), "Haggai", "Hagai", False),
    (("Zechariah", "Zakharia"), "Zechariah", "Zakharia", False),
    (("Malachi", "Maleakhi"), "Malachi", "Maleakhi", False),
    (("Matthew", "Matius"), "Matthew", "Matius", False),
    (("Mark", "Markus"), "Mark", "Markus", False),
    (("Luke", "Lukas"), "Luke", "Lukas", False),
    (("John", "Yohanes"), "John", "Yohanes", True),
    (("Acts", "Kisah Para Rasul"), "Acts", "Kisah Para Rasul", False),
    (("Romans", "Roma"), "Romans", "Roma", False),
    (("Corinthians", "Korintus"), "Corinthians", "Korintus", True),
    (("Galatians", "Galatia"), "Galatians", "Galatia", False),
    (("Ephesians", "Efesus"), "Ephesians", "Efesus", False),
    (("Philippians", "Filipi"), "Philippians", "Filipi", False),
    (("Colossians", "Kolose"), "Colossians", "Kolose", False),
    (("Thessalonians", "Tesalonika"), "Thessalonians", "Tesalonika", True),
    (("Timothy", "Timotius"), "Timothy", "Timotius", True),
    (("Titus",), "Titus", "Titus", False),
    (("Philemon", "Filemon"), "Philemon", "Filemon", False),
    (("Hebrews", "Ibrani"), "Hebrews", "Ibrani", False),
    (("James", "Yakobus"), "James", "Yakobus", False),
    (("Peter", "Petrus"), "Peter", "Petrus", True),
    (("Jude", "Yudas"), "Jude", "Yudas", False),
    (("Revelation", "Wahyu"), "Revelation", "Wahyu", False),
]

PREFIX_WORDS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "pertama": 1,
    "kedua": 2,
    "ketiga": 3,
}

COMMENTARY_RE = re.compile(
    r"\b(saudara(?:-saudara)?|jemaat|hari ini|mari kita|let us|i want to|today we|brothers and sisters)\b",
    re.IGNORECASE,
)

LONG_READING_WORDS = 20


def _aliases() -> list[tuple[str, str, str, bool]]:
    rows: list[tuple[str, str, str, bool]] = []
    for names, display_en, display_id, numbered in BOOKS:
        for name in names:
            rows.append((name, display_en, display_id, numbered))
    rows.sort(key=lambda row: len(row[0]), reverse=True)
    return rows


_ALIAS_ROWS = _aliases()
_BOOK_PATTERN = "|".join(re.escape(name) for name, *_rest in _ALIAS_ROWS)
_BOOK_INDEX = {name.casefold(): (display_en, display_id, numbered) for name, display_en, display_id, numbered in _ALIAS_ROWS}
_CITATION_RE = re.compile(
    rf"(?<![\w])(?:(?P<prefix>[1-3]|[Ii]{{1,3}}|{'|'.join(PREFIX_WORDS)})\s+)?(?P<book>{_BOOK_PATTERN})\s+(?P<chapter>\d{{1,3}})\s*[:：]\s*(?P<verse>\d{{1,3}})(?:\s*[-–—]\s*(?P<verse_end>\d{{1,3}}))?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Citation:
    display_en: str
    display_id: str
    start: int
    end: int

    def format(self, language: str) -> str:
        return self.display_en if language == "en" else self.display_id


def find_citations(text: str) -> list[Citation]:
    found: list[Citation] = []
    for match in _CITATION_RE.finditer(text):
        book_name = match.group("book")
        display_en, display_id, numbered = _BOOK_INDEX[book_name.casefold()]
        prefix = _prefix_number(match.group("prefix"))
        if prefix and not numbered:
            continue
        if prefix:
            display_en = f"{prefix} {display_en}"
            display_id = f"{prefix} {display_id}"
        verse_end = int(match.group("verse_end")) if match.group("verse_end") else None
        verse = int(match.group("verse"))
        verse_text = f"{verse}-{verse_end}" if verse_end and verse_end != verse else str(verse)
        chapter = int(match.group("chapter"))
        found.append(
            Citation(
                display_en=f"{display_en} {chapter}:{verse_text}",
                display_id=f"{display_id} {chapter}:{verse_text}",
                start=match.start(),
                end=match.end(),
            )
        )
    return found


def primary_citation(text: str) -> Citation | None:
    citations = find_citations(text)
    return citations[0] if citations else None


def is_long_reading(text: str, citation: Citation) -> bool:
    tail_words = re.findall(r"\w+", text[citation.end :], flags=re.UNICODE)
    return len(tail_words) >= LONG_READING_WORDS


def looks_like_commentary(text: str) -> bool:
    return COMMENTARY_RE.search(text) is not None


def reading_status(citation: Citation | None, speaker_language: str) -> tuple[str, str, str | None, str | None]:
    """Listener original, listener translation, reference_en, reference_id."""
    reference_en = citation.display_en if citation else None
    reference_id = citation.display_id if citation else None
    if speaker_language == "en":
        original = f"{reference_en} — Scripture is being read" if reference_en else "Scripture is being read"
        translation = f"{reference_id} — Firman sedang dibacakan" if reference_id else "Firman sedang dibacakan"
    else:
        original = f"{reference_id} — Firman sedang dibacakan" if reference_id else "Firman sedang dibacakan"
        translation = f"{reference_en} — Scripture is being read" if reference_en else "Scripture is being read"
    return original, translation, reference_en, reference_id


def _prefix_number(raw: str | None) -> int | None:
    if not raw:
        return None
    if raw.isdigit():
        return int(raw)
    roman = {"i": 1, "ii": 2, "iii": 3}
    if raw.casefold() in roman:
        return roman[raw.casefold()]
    return PREFIX_WORDS.get(raw.casefold())
