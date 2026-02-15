import pytest


@pytest.fixture
def simple_messages():
    return [{"role": "user", "content": "What is 2+2?"}]


@pytest.fixture
def medium_messages():
    return [
        {"role": "system", "content": "You are a helpful coding assistant. You write clean, well-documented code and follow best practices. Always include type hints and error handling."},
        {"role": "user", "content": "I need to implement a data processing pipeline. Write a Python function that reads a CSV file, filters rows based on a condition, computes the average of a numeric column while handling missing values, and writes the results to a new file. The function should support different delimiter formats and handle encoding issues gracefully."},
    ]


@pytest.fixture
def complex_messages():
    return [
        {
            "role": "system",
            "content": "You are a senior software architect. " * 50,  # long system prompt
        },
        {"role": "user", "content": "Analyse the trade-offs between microservices and monolithic architecture for our e-commerce platform. Consider scalability, team structure, deployment complexity, and data consistency. Provide a step-by-step migration plan with comprehensive implementation details."},
        {"role": "assistant", "content": "Let me break this down...\n" + "detailed analysis " * 100},
        {"role": "user", "content": "Now compare this with a serverless approach and explain the pros and cons in depth, including cost analysis and performance implications for each architecture pattern."},
    ]


@pytest.fixture
def coding_messages():
    return [
        {"role": "user", "content": "Implement a Python class for a binary search tree with insert, delete, and search methods. Include unittest tests."},
    ]


@pytest.fixture
def tools_fixture():
    return [
        {"type": "function", "function": {"name": "get_weather", "parameters": {}}},
        {"type": "function", "function": {"name": "search_web", "parameters": {}}},
        {"type": "function", "function": {"name": "run_code", "parameters": {}}},
        {"type": "function", "function": {"name": "read_file", "parameters": {}}},
    ]


@pytest.fixture
def mock_models_response():
    return {
        "data": [
            {"id": "gemma-3-4b", "object": "model"},
            {"id": "qwen-3-4b", "object": "model"},
            {"id": "qwen2.5-coder-7b", "object": "model"},
            {"id": "Mistral-Small-3.2-24B", "object": "model"},
            {"id": "gemma-3-27b", "object": "model"},
            {"id": "qwen2.5-coder-32b", "object": "model"},
            {"id": "Qwen3-30B-A3B", "object": "model"},
            {"id": "nomic-embed-text:latest", "object": "model"},
        ],
        "object": "list",
    }
