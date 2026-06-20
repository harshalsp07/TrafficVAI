# Contributing to TrafficAI

Thank you for your interest in contributing to TrafficAI! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [How to Contribute](#how-to-contribute)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Commit Messages](#commit-messages)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help create a welcoming environment for all contributors

## Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/your-username/TrafficVAI.git
   cd TrafficVAI
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/harshalsp07/TrafficVAI.git
   ```
4. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

### Prerequisites

- Python 3.11+
- Node.js 18+
- Git

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### Frontend

```bash
cd frontend
npm install
```

### Running Locally

```bash
# Terminal 1 — Backend
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2 — Frontend
cd frontend
npm run dev
```

## Project Structure

```
TrafficVAI/
├── backend/               # FastAPI backend
│   ├── app/
│   │   ├── main.py       # App entry point
│   │   ├── config.py     # Configuration
│   │   ├── api/          # REST API routes
│   │   ├── models/       # Pydantic schemas
│   │   ├── services/     # Business logic
│   │   ├── db/           # Database layer
│   │   └── utils/        # Utilities
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/              # React dashboard
│   ├── src/
│   │   ├── pages/        # Dashboard pages
│   │   ├── components/   # UI components
│   │   └── hooks/        # Custom hooks
│   ├── package.json
│   └── vite.config.js
├── training/              # ML training pipeline
│   ├── configs/          # Training configs
│   └── scripts/          # Training/export/eval scripts
├── docker-compose.yml
└── .env.example
```

## How to Contribute

### Types of Contributions

- **Bug Fixes** — Fix existing issues
- **Features** — Add new functionality
- **Documentation** — Improve docs, README, or code comments
- **Tests** — Add or improve test coverage
- **Performance** — Optimize existing code
- **UI/UX** — Improve the dashboard design

### Finding Issues

- Check the [Issues](https://github.com/harshalsp07/TrafficVAI/issues) tab for open tasks
- Look for issues labeled `good first issue` for beginner-friendly tasks
- Feel free to open a new issue if you find a bug or have a feature idea

## Pull Request Process

1. **Update your fork** with the latest upstream changes:
   ```bash
   git fetch upstream
   git merge upstream/main
   ```

2. **Make your changes** in small, focused commits

3. **Test your changes**:
   - Backend: Verify the API starts without errors
   - Frontend: Run `npm run build` to check for build errors
   - Training: Run `python scripts/train.py --help` to verify scripts

4. **Push your branch**:
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Open a Pull Request** on GitHub with:
   - A clear, descriptive title
   - Summary of changes
   - Screenshots/videos if applicable
   - Reference any related issues

6. **Respond to review feedback** — maintainers may request changes

## Coding Standards

### Python (Backend & Training)

- Follow [PEP 8](https://peps.python.org/pep-0008/) style guide
- Use type hints for function signatures
- Write docstrings for public functions
- Keep functions focused and concise
- Use async/await for I/O-bound operations

```python
async def get_violation(violation_id: str) -> ViolationOut:
    """Fetch a single violation record by ID."""
    ...
```

### JavaScript/JSX (Frontend)

- Use functional components with hooks
- Keep components focused and reusable
- Use meaningful variable and function names
- Prefer `const` over `let`

```jsx
const ViolationBadge = ({ type, confidence }) => {
  // ...
};
```

### YAML Configs

- Use 2-space indentation
- Add comments for non-obvious settings
- Keep training configs under 60 lines

## Commit Messages

Use clear, descriptive commit messages:

```
<type>: <short description>

<optional body>
```

### Types

| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation changes |
| `style` | Code style changes (formatting, no logic change) |
| `refactor` | Code restructuring (no feature/fix change) |
| `test` | Adding or updating tests |
| `chore` | Build, CI, or tooling changes |
| `perf` | Performance improvements |

### Examples

```
feat: add triple-riding detection to violation service

fix: resolve race condition in SSE stream handler

docs: update README with Docker deployment instructions

refactor: extract ANPR preprocessing into separate module
```

## Reporting Issues

When reporting bugs, please include:

1. **Environment** — OS, Python version, Node version
2. **Steps to reproduce** — Clear steps to trigger the issue
3. **Expected behavior** — What should happen
4. **Actual behavior** — What actually happens
5. **Screenshots** — If applicable
6. **Logs** — Relevant error messages or console output

## Questions?

Open a discussion or reach out to the maintainers. We're happy to help!

---

Thank you for contributing to TrafficAI! Every contribution helps make roads safer.
