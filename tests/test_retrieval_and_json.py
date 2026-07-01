import json

from development_intelligence.ollama_client import parse_json_response
from development_intelligence.retrieval import retrieve_chunks


def test_retrieval_prefers_relevant_chunk():
    chunks = [
        {"chunk_id": "a", "text": "School education and literacy improved."},
        {"chunk_id": "b", "text": "Coffee exports and national income shaped the economy."},
    ]
    result = retrieve_chunks("education literacy schooling", chunks, top_k=1)
    assert result[0]["chunk_id"] == "a"
    assert result[0]["retrieval_score"] > 0


def test_json_parser_accepts_strict_and_fenced_json():
    assert parse_json_response('{"score": 5}') == {"score": 5}
    assert parse_json_response('```json\n{"score": 4}\n```') == {"score": 4}


def test_json_parser_rejects_non_json():
    try:
        parse_json_response("This is not structured output")
    except json.JSONDecodeError:
        pass
    else:
        raise AssertionError("Expected JSONDecodeError")

