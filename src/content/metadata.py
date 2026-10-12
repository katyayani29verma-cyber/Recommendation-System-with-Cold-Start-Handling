"""Consistent movie text used during training and inference."""

FORMAT_VERSION = 1


def metadata_text(metadata: dict) -> str:
    title = metadata.get("title")

    if not isinstance(title, str) or not title.strip():
        raise ValueError("Movie metadata requires a nonempty title.")

    title = " ".join(title.split())
    genres = metadata.get("genres", "")

    if genres is None:
        values = []
    elif isinstance(genres, str):
        values = genres.split("|")
    elif isinstance(genres, (list, tuple)):
        values = list(genres)
    else:
        raise ValueError("Genres must be a string, list, or tuple.")

    cleaned = []

    for value in values:
        if not isinstance(value, str):
            raise ValueError("Every genre must be a string.")

        value = " ".join(value.split())

        if value and value != "(no genres listed)":
            cleaned.append(value)

    cleaned = sorted(set(cleaned), key=str.casefold)

    text = f"Title: {title}"

    if cleaned:
        text += "\nGenres: " + ", ".join(cleaned)

    return text