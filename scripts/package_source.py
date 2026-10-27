"""Create the source-only ZIP expected by the Colab notebook; never includes data or secrets."""
from pathlib import Path
import hashlib
import zipfile


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "artifacts/upgaming-petseg-source.zip"
    output.parent.mkdir(exist_ok=True)
    files = [root / n for n in ["pyproject.toml", "README.md", "AI_USAGE.md", "app.py", "requirements-cpu.txt", "requirements-local-lock.txt", "VERIFICATION.md", "MANIFEST.in", ".gitignore", ".gitattributes"] if (root / n).exists()]
    extensions = {"petseg": ".py", "configs": ".json", "tests": ".py", "docs": ".md",
                  "notebooks": ".ipynb", "scripts": ".py"}
    for directory, suffix in extensions.items():
        files.extend(p for p in (root / directory).rglob("*" + suffix)
                     if p.is_file() and not p.is_symlink()
                     and not any(part.startswith(".") or part == "__pycache__"
                                 for part in p.relative_to(root).parts))
    files.append(root / ".streamlit/config.toml")
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(files)):
            archive.write(path, path.relative_to(root))
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(".zip.sha256").write_text(digest + "\n", encoding="ascii")
    print(output)


if __name__ == "__main__":
    main()
