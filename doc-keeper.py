import os
import google.generativeai as genai

# 🔑 Load your Gemini API key (recommended to use environment variable) #
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ⛔ Files and folders to skip during documentation generation
IGNORE_EXTENSIONS = ('.pyc', '.log', '.lock', '.env', '.sqlite3', '.db')
IGNORE_FILES = ('requirements.lock', '.env', 'secrets.json')
IGNORE_DIRS = ('.git', '__pycache__', 'venv', 'node_modules', 'dist', 'build', '.idea', '.vscode', '.pytest_cache')

def read_repo_files(base_path: str) -> dict:
    """
    Recursively reads all readable files in the repository directory, excluding
    ignored directories and extensions.

    Parameters:
        base_path (str): The root directory of the repository.

    Returns:
        dict: A dictionary mapping relative file paths to their content.
    """
    file_data = {}

    for root, dirs, files in os.walk(base_path):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

        for file in files:
            if file.endswith(IGNORE_EXTENSIONS) or file in IGNORE_FILES:
                continue

            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                relative_path = os.path.relpath(file_path, base_path)
                file_data[relative_path] = content
            except Exception as e:
                print(f"⚠️ Skipping {file_path}: {e}")
    
    return file_data

def generate_documentation(repo_files: dict) -> str:
    """
    Generates Markdown documentation using Gemini based on the provided codebase.

    Parameters:
        repo_files (dict): A mapping of file paths to file content.

    Returns:
        str: Generated documentation in Markdown format.
    """
    prompt = """("You are an expert software architect and technical writer.\n"
        "Given the following codebase, generate a **complete and detailed** documentation in Markdown format. Be exhaustive and helpful for developers.\n\n"
        "Include the following sections:\n"
        "- 🧾 Project Overview\n"
        "- ⚙️ Setup & Installation Instructions\n"
        - "- 🧩 Explanation of Key Modules, Classes, and Functions\n"
        "- 🗂 Folder & File Structure with Descriptions\n"
        "- 🔧 How to Use (with CLI or API examples if present)\n"
        "- 🤝 Contribution Guidelines (if applicable)\n"
        "- 🧪 Testing & Debugging Instructions (if test files exist)\n"
        "- 🧠 Add Python-style **docstrings** to all functions with descriptions of parameters and return types\n\n"
        "### Codebase Contents:\n")"""

    for filename, content in repo_files.items():
        trimmed_content = content[:3000]  # prevent token overflow
        prompt += f"\n#### FILE: {filename}\n```python\n{trimmed_content}\n```\n"

    model = genai.GenerativeModel('gemini-1.5-pro')
    response = model.generate_content(prompt)
    return response.text

def write_documentation(doc_text: str, output_file: str = "DOCUMENTATION.md") -> None:
    """
    Writes the generated documentation to a file.

    Parameters:
        doc_text (str): The documentation content to write.
        output_file (str): Output file name. Default is 'DOCUMENTATION.md'.
    """
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(doc_text)
    print(f"✅ Documentation written to {output_file}")

if __name__ == "__main__":
    repo_path = os.path.abspath(os.path.dirname(__file__))  # Root of the repo where doc-keeper.py lives
    print(f"📂 Scanning project directory: {repo_path}")
    repo_files = read_repo_files(repo_path)
    print("🤖 Using Gemini 2.0 to generate extensive documentation...")
    documentation = generate_documentation(repo_files)
    write_documentation(documentation)
