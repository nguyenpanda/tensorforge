# TensorForge 🔥

**An Interactive, Professional-Grade "LeetCode-Style" Educational Framework for Scientific Computing, Deep Learning, and GPU Kernel Engineering.**

---

## 1. THE STANDARD SETUP

Follow these instructions to initialize your interactive development environment:

1. **Install `uv`** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone the repository and enter the workspace**:
   ```bash
   git clone https://github.com/nguyenpanda/tensorforge.git
   cd tensorforge
   ```

3. **Run the automated bootstrap script**:
   ```bash
   python3 bootstrap.py
   ```
   *This detects your operating system and hardware architecture (macOS Metal vs. Linux CUDA) and installs all required packages into a local `.venv`.*

4. **Activate the virtual environment**:
   ```bash
   source .venv/bin/activate
   ```

---

## 2. ENABLING AUTOCOMPLETE

To enable command-line tab-completion for all CLI commands, curricula, tiers, and lessons, generate and source the completion script for your specific shell (`bash`, `zsh`, or `fish`):

```bash
uv run tforge autocomplete --shell <your_shell> > ~/.tforge-autocomplete.sh
source ~/.tforge-autocomplete.sh
```

To persist tab-completion across sessions, add the source command to your shell configuration file (`~/.bashrc`, `~/.zshrc`, or `~/.config/fish/config.fish`):

```bash
echo "source ~/.tforge-autocomplete.sh" >> ~/.zshrc  # Or ~/.bashrc / ~/.config/fish/config.fish
```

---

## 3. SYSTEM READINESS CHECK

Before starting curriculum exercises, verify that your environment is fully operational by running the verification suite in order:

```bash
# 1. Run static syntax and type checks
uv run tforge lint

# 2. Verify framework infrastructure and AST validators
uv run pytest tests/infra/ -v

# 3. Validate all golden baseline implementations
uv run tforge validate
```

If all commands pass without errors, your system is ready for development.

---

## 4. THE "NUCLEAR OPTION" (FACTORY RESET)

If your environment enters a corrupted state (e.g., mismatched CUDA driver versions, broken package caches, or dependency conflicts), perform a clean factory reset:

```bash
# Optional: Clear downloaded dataset caches
rm -rf ~/.cache/tensorforge/

# Remove the virtual environment and force clean dependency resolution
python3 bootstrap.py --reconfigure
```

*The `--reconfigure` flag deletes `.venv` and temporary build artifacts, re-evaluating hardware markers and forcing explicit PyTorch index resolution.*

---

## 5. THE STUDENT WORKFLOW

### Testing a Full Lesson Module
Run the automated verification suite against an entire lesson module:

```bash
uv run tforge check <curriculum> <tier> <lesson>
# Example: uv run tforge check arraysmith basic 01
```

### Testing a Specific Function (Granular Targeting)
Use the `-t` or `--target` flag to evaluate a single method during iterative development:

```bash
uv run tforge check arraysmith basic 01 -t create_integer_range
```

### Accessing Hints
If you encounter difficulties during implementation, call the built-in hint helper directly within your lesson code:

```python
from forge_core import show_hint

class ArrayCreation:
    
    @classmethod
    def exercise_01(cls) -> np.ndarray:
        show_hint()
```

---

## Contributing & Development

Refer to [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- Authoring curriculum lessons and reference baselines.
- Configuring AST inspection rules and custom feedback messages.
- Extending HPC backends and registering automated hints.

---

## Author Information

- **Author & Principal Architect:** nguyenpanda
- **Affiliation:** High Performance Computing Laboratory, HCMUT-VNU
