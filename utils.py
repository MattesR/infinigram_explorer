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
