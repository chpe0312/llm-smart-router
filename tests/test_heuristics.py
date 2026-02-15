from smart_router.heuristics import score_request


class TestHeuristicScoring:
    def test_simple_question_scores_low(self, simple_messages):
        result = score_request(simple_messages)
        assert result.score < 0.3
        assert result.confident

    def test_medium_request_scores_mid(self, medium_messages):
        result = score_request(medium_messages)
        assert 0.15 <= result.score <= 0.75

    def test_complex_request_scores_high(self, complex_messages):
        result = score_request(complex_messages)
        assert result.score >= 0.5

    def test_tools_increase_score(self, simple_messages, tools_fixture):
        without_tools = score_request(simple_messages)
        with_tools = score_request(simple_messages, tools=tools_fixture)
        assert with_tools.score > without_tools.score

    def test_code_blocks_increase_score(self):
        messages = [
            {"role": "user", "content": "Fix this:\n```python\ndef foo():\n    pass\n```\n```python\ndef bar():\n    pass\n```\n```python\ndef baz():\n    pass\n```"},
        ]
        result = score_request(messages)
        assert any("code" in r.lower() for r in result.reasons)

    def test_image_content_increases_score(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                ],
            },
        ]
        result = score_request(messages)
        assert any("image" in r.lower() for r in result.reasons)

    def test_many_turns_increase_score(self):
        messages = [
            {"role": "user", "content": f"Message {i}"}
            for i in range(12)
        ]
        result = score_request(messages)
        assert any("turn" in r.lower() or "conversation" in r.lower() for r in result.reasons)

    def test_simple_keywords_reduce_score(self):
        messages = [{"role": "user", "content": "Translate this to German: Hello"}]
        result = score_request(messages)
        assert result.score < 0.2

    def test_complex_keywords_increase_score(self):
        messages = [{"role": "user", "content": "Explain in detail step by step how to implement this"}]
        result = score_request(messages)
        assert any("complex" in r.lower() for r in result.reasons)

    def test_moderate_keywords_give_small_boost(self):
        messages = [{"role": "user", "content": "Analyse the trade-offs between X and Y"}]
        result = score_request(messages)
        assert any("moderate" in r.lower() for r in result.reasons)
        assert 0.1 <= result.score <= 0.5

    def test_moderate_keywords_cancel_simple_penalty(self):
        messages = [{"role": "user", "content": "Compare and list the differences"}]
        result = score_request(messages)
        assert not any("simple" in r.lower() for r in result.reasons)

    def test_german_moderate_keywords(self):
        messages = [{"role": "user", "content": "Analysiere E-Autos vs Verbrenner"}]
        result = score_request(messages)
        assert any("moderate" in r.lower() for r in result.reasons)
        assert result.score < 0.5

    def test_german_complex_keywords(self):
        messages = [{"role": "user", "content": "Analysiere die Vor- und Nachteile und erkläre Schritt für Schritt"}]
        result = score_request(messages)
        assert any("complex" in r.lower() for r in result.reasons)
        assert result.score >= 0.3

    def test_german_simple_keywords(self):
        messages = [{"role": "user", "content": "Übersetze das ins Englische: Hallo"}]
        result = score_request(messages)
        assert result.score < 0.2

    def test_german_simple_question(self):
        messages = [{"role": "user", "content": "Was ist die Hauptstadt von Frankreich?"}]
        result = score_request(messages)
        assert result.score < 0.2

    def test_german_complex_design_task(self):
        messages = [{"role": "user", "content": "Entwirf eine umfassende und detaillierte Architektur für eine Microservices-Plattform"}]
        result = score_request(messages)
        assert result.score >= 0.5

    def test_confidence_small(self):
        messages = [{"role": "user", "content": "Was ist die Hauptstadt von Frankreich?"}]
        result = score_request(messages)
        assert result.confident
        assert result.score < 0.2

    def test_confidence_medium(self):
        messages = [{"role": "user", "content": "analysiere e autos vs verbrenner, denke schritt für schritt"}]
        result = score_request(messages)
        assert result.confident
        assert 0.3 <= result.score <= 0.7

    def test_confidence_uncertain_in_transition(self):
        # Score in the 0.2-0.4 transition zone should be uncertain
        messages = [{"role": "user", "content": "implement something"}]
        result = score_request(messages)
        # "implement" is complex → 0.35, tokens short → 0.0, total ~0.35
        assert not result.confident

    def test_empty_messages(self):
        result = score_request([])
        assert result.score == 0.0

    def test_score_clamped_to_range(self, complex_messages, tools_fixture):
        result = score_request(complex_messages, tools=tools_fixture)
        assert 0.0 <= result.score <= 1.0
