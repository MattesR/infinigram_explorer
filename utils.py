import html
import re

def clean_html(text):
    # Decode HTML entities (e.g., &lt;, &amp;, &nbsp;)
    text = html.unescape(text)

    # Collapse multiple spaces
    text = re.sub(r'</?(br|div|p|b|i|span|a)[^>]*>', ' ', text, flags=re.IGNORECASE)

    return text


def get_token(key: str, filename: str = "token_file.ini") -> str | None:
    try:
        with open(filename, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() == key:
                    return v.strip().strip("'\"")
    except FileNotFoundError:
        print(f"Token file '{filename}' not found.")
    return None
