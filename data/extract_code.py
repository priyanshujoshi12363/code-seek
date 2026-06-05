import os

REPOS_DIR = "data/repos"
OUTPUT_FILE = "data/code_data/embedded_code.txt"

EXTENSIONS = {
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".ino"
}

os.makedirs("data/code_data", exist_ok=True)

files_processed = 0

with open(OUTPUT_FILE, "w", encoding="utf-8") as out:

    for root, dirs, files in os.walk(REPOS_DIR):

        # Skip useless folders
        dirs[:] = [
            d for d in dirs
            if d not in {
                ".git",
                "build",
                "__pycache__",
                ".github"
            }
        ]

        for file in files:

            ext = os.path.splitext(file)[1].lower()

            if ext not in EXTENSIONS:
                continue

            path = os.path.join(root, file)

            try:
                with open(
                    path,
                    "r",
                    encoding="utf-8",
                    errors="ignore"
                ) as f:

                    content = f.read()

                    if len(content.strip()) < 50:
                        continue

                    out.write("\n<FILE_START>\n")
                    out.write(path)
                    out.write("\n\n")

                    out.write(content)

                    out.write("\n\n<FILE_END>\n\n")

                    files_processed += 1

            except Exception:
                pass

print(f"Files processed: {files_processed}")
print(f"Saved to: {OUTPUT_FILE}")