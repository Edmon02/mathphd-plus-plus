# Contributing to MathPhD++

Thank you for your interest in contributing to MathPhD++! This project aims to push the boundaries of AI-driven mathematical reasoning, and we welcome contributions from researchers, engineers, and mathematicians.

## How to Contribute

### Reporting Issues
- Use the [GitHub Issues](https://github.com/Edmon02/mathphd-plus-plus/issues) page
- Check existing issues before creating a new one
- Use the provided issue templates when available
- Include reproduction steps, expected behavior, and environment details

### Code Contributions

1. **Fork** the repository
2. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes** following the code style guidelines below
4. **Test your changes** — ensure training runs and evaluations pass
5. **Submit a Pull Request** with a clear description

### Code Style
- Follow PEP 8 for Python code
- Use type hints for function signatures
- Add docstrings to all public functions and classes
- Keep functions focused and under 100 lines where practical

### Areas We Need Help
- **Mathematical Datasets**: Curating high-quality math problem-solution pairs
- **Formal Verification**: Lean 4 integration for proof checking
- **Evaluation**: Creating harder benchmark problems
- **Inference Strategies**: Improving MCTS and Tree-of-Thoughts implementations
- **Documentation**: Tutorials, examples, and mathematical background

## Development Setup

```bash
git clone https://github.com/Edmon02/mathphd-plus-plus.git
cd mathphd-plus-plus
pip install -e ".[dev]"
```

## Code of Conduct

Please be respectful and constructive in all interactions. We follow the [Contributor Covenant](https://www.contributor-covenant.org/) Code of Conduct.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
