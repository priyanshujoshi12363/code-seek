import os

REPOS_DIR = "data/repos"
OUTPUT_FILE = "data/code_data/embedded_docs.txt"

DOC_EXTENSIONS = {
    ".md",
    ".txt",
    ".rst"
}

SKIP_FILES = {
    "license",
    "license.md",
    "license.txt",
    "copying",
    "copyright",
    "notice",
    "notice.txt",
    "authors",
    "contributors",
    "patents"
}

SKIP_DIRS = {
    ".git",
    ".github",
    "__pycache__",
    "build",
    "legal",
    "licenses",
    "license",
    "third_party",
    "thirdparty",
    "vendor",
    "vendors"
}

GOOD_KEYWORDS = [
    "esp32",
    "wifi",
    "http",
    "https",
    "mqtt",
    "gpio",
    "uart",
    "spi",
    "i2c",
    "adc",
    "pwm",
    "bluetooth",
    "ble",
    "sensor",
    "freertos",
    "task",
    "interrupt",
    "driver",
    "arduino",
    "network",
    "tcp",
    "udp",
    "websocket",
    "ota"
]

os.makedirs("data/code_data", exist_ok=True)

docs_saved = 0

with open(OUTPUT_FILE, "w", encoding="utf-8") as out:

    for root, dirs, files in os.walk(REPOS_DIR):

        dirs[:] = [
            d for d in dirs
            if d.lower() not in SKIP_DIRS
        ]

        for file in files:

            if file.lower() in SKIP_FILES:
                continue

            ext = os.path.splitext(file)[1].lower()

            if ext not in DOC_EXTENSIONS:
                continue

            path = os.path.join(root, file)

            try:
                with open(
                    path,
                    "r",
                    encoding="utf-8",
                    errors="ignore"
                ) as f:

                    text = f.read()

                if len(text) < 300:
                    continue

                lower_text = text.lower()

                if not any(
                    keyword in lower_text
                    for keyword in GOOD_KEYWORDS
                ):
                    continue

                # Skip legal docs
                legal_phrases = [
                    "all rights reserved",
                    "redistribution and use",
                    "permission is hereby granted",
                    "copyright",
                    "software is provided",
                    "warranty disclaimer"
                ]

                if any(
                    phrase in lower_text
                    for phrase in legal_phrases
                ):
                    continue

                out.write("\n<DOC_START>\n")
                out.write(path)
                out.write("\n\n")
                out.write(text)
                out.write("\n\n<DOC_END>\n\n")

                docs_saved += 1

            except Exception:
                pass

print(f"Useful docs saved: {docs_saved}")
print(f"Output: {OUTPUT_FILE}")