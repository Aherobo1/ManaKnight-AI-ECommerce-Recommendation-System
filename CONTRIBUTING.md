# Contributing to E-Commerce Product Recommendation System

Thank you for your interest in contributing to our project! This document provides guidelines and information for contributors.

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Git
- Virtual environment (recommended)

### Setting Up Development Environment
1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/ds_task_1ab.git`
3. Create a virtual environment: `python -m venv venv`
4. Activate virtual environment: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
5. Install dependencies: `pip install -r requirements.txt`
6. Copy `.env.example` to `.env` and configure your environment variables

## 📝 Development Guidelines

### Code Style
- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Add docstrings for all functions and classes
- Keep functions small and focused (max 50 lines)
- Use type hints where appropriate

### Commit Messages
- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit first line to 72 characters
- Reference issues and pull requests when applicable

Example:
```
Add OCR functionality for handwritten queries

- Implement Tesseract integration
- Add image preprocessing pipeline
- Include confidence scoring
- Fixes #123
```

### Branch Naming
- `feature/description` for new features
- `bugfix/description` for bug fixes
- `hotfix/description` for urgent fixes
- `docs/description` for documentation updates

## 🧪 Testing

### Running Tests
```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=services

# Run specific test file
python -m pytest tests/test_api.py
```

### Writing Tests
- Write tests for all new functionality
- Aim for >80% code coverage
- Use descriptive test names
- Follow AAA pattern (Arrange, Act, Assert)

## 📋 Pull Request Process

1. **Create an Issue**: For significant changes, create an issue first to discuss the approach
2. **Create a Branch**: Create a feature branch from `main`
3. **Make Changes**: Implement your changes following the guidelines
4. **Add Tests**: Write tests for new functionality
5. **Update Documentation**: Update README.md and docstrings as needed
6. **Run Tests**: Ensure all tests pass
7. **Submit PR**: Create a pull request with a clear description

### Pull Request Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings introduced
```

## 🐛 Reporting Issues

### Bug Reports
Include:
- Clear description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)
- Error messages/logs

### Feature Requests
Include:
- Clear description of the feature
- Use case/motivation
- Proposed implementation (if any)
- Alternatives considered

## 📚 Documentation

### Code Documentation
- Add docstrings to all public functions and classes
- Use Google-style docstrings
- Include parameter types and return types
- Provide usage examples for complex functions

### README Updates
- Keep installation instructions current
- Update API documentation for new endpoints
- Add examples for new features
- Update project status and roadmap

## 🏗 Project Structure

When adding new files, follow the existing structure:
- `services/` - Backend service classes
- `templates/` - HTML templates
- `static/` - CSS, JS, and image files
- `tests/` - Unit tests
- `notebooks/` - Jupyter notebooks for experimentation
- `models/` - Trained ML models

## 🤝 Code Review Process

### For Reviewers
- Check code quality and style
- Verify tests are adequate
- Test functionality manually if needed
- Provide constructive feedback
- Approve when ready

### For Contributors
- Respond to feedback promptly
- Make requested changes
- Ask questions if feedback is unclear
- Be open to suggestions

## 📞 Getting Help

- Create an issue for questions
- Join our discussions
- Check existing documentation
- Review similar implementations

## 🙏 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Project documentation

Thank you for contributing to our project! 🎉
