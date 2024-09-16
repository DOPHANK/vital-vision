import pytest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/mapper')))
from google_format import GoogleVisionMapper

@pytest.fixture
def sample_ocr_result():
    return [
        {
            "description": "Hello",
            "boundingPoly": {
                "vertices": [
                    {"x": 0, "y": 0},
                    {"x": 50, "y": 0},
                    {"x": 50, "y": 20},
                    {"x": 0, "y": 20}
                ]
            }
        },
        {
            "description": "World",
            "boundingPoly": {
                "vertices": [
                    {"x": 60, "y": 0},
                    {"x": 110, "y": 0},
                    {"x": 110, "y": 20},
                    {"x": 60, "y": 20}
                ]
            }
        }
    ]

def test_format_text_annotations(sample_ocr_result):
    mapper = GoogleVisionMapper()
    result = mapper.format_text_annotations(sample_ocr_result)
    
    assert len(result) == 3
    assert result[0]["description"] == "Hello World"
    assert result[0]["locale"] == ""
    assert result[0]["boundingPoly"]["vertices"] == [[{"x": None, "y": None} for _ in range(4)]]
    
    assert result[1]["description"] == "Hello"
    assert result[2]["description"] == "World"
    
    assert "confidence" not in result[1]
    assert "confidence" not in result[2]

def test_format_full_text_annotation(sample_ocr_result):
    mapper = GoogleVisionMapper()
    result = mapper.format_full_text_annotation(sample_ocr_result)
    
    assert "pages" in result
    assert len(result["pages"]) == 1
    assert result["text"] == "Hello World"
    
    page = result["pages"][0]
    assert "blocks" in page
    assert len(page["blocks"]) == 1
    
    block = page["blocks"][0]
    assert block["blockType"] == "TEXT"
    assert len(block["paragraphs"]) == 1
    
    paragraph = block["paragraphs"][0]
    assert len(paragraph["words"]) == 2
    
    word = paragraph["words"][0]
    assert len(word["symbols"]) == 5
    assert word["property"]["detectedLanguages"][0]["languageCode"] == "vi"

def test_map_to_google_vision(sample_ocr_result):
    mapper = GoogleVisionMapper()
    result = mapper.map_to_google_vision(sample_ocr_result)
    
    assert "responses" in result
    assert len(result["responses"]) == 1
    
    response = result["responses"][0]
    assert "textAnnotations" in response
    assert "fullTextAnnotation" in response
    
    assert len(response["textAnnotations"]) == 3
    assert response["fullTextAnnotation"]["text"] == "Hello World"

def test_create_symbol():
    from google_format import create_symbol
    
    symbols = create_symbol("Test")
    assert len(symbols) == 4
    assert symbols[-1]["property"]["detectedBreak"]["type"] == "SPACE"
    assert all(symbol["confidence"] is None for symbol in symbols)

def test_create_word():
    from google_format import create_word, create_symbol
    
    symbols = create_symbol("Test")
    bounding_box = {"vertices": [{"x": 0, "y": 0}, {"x": 40, "y": 0}, {"x": 40, "y": 20}, {"x": 0, "y": 20}]}
    word = create_word(symbols, bounding_box)
    
    assert word["property"]["detectedLanguages"][0]["languageCode"] == "vi"
    assert word["boundingBox"] == bounding_box
    assert len(word["symbols"]) == 4
    assert word["confidence"] is None

def test_create_paragraph():
    from google_format import create_paragraph, create_word, create_symbol
    
    symbols = create_symbol("Test")
    bounding_box = {"vertices": [{"x": 0, "y": 0}, {"x": 40, "y": 0}, {"x": 40, "y": 20}, {"x": 0, "y": 20}]}
    word = create_word(symbols, bounding_box)
    paragraph = create_paragraph([word])
    
    assert len(paragraph["words"]) == 1
    assert paragraph["confidence"] is None

def test_create_block():
    from google_format import create_block, create_paragraph, create_word, create_symbol
    
    symbols = create_symbol("Test")
    bounding_box = {"vertices": [{"x": 0, "y": 0}, {"x": 40, "y": 0}, {"x": 40, "y": 20}, {"x": 0, "y": 20}]}
    word = create_word(symbols, bounding_box)
    paragraph = create_paragraph([word])
    block = create_block([paragraph])
    
    assert block["blockType"] == "TEXT"
    assert len(block["paragraphs"]) == 1
    assert block["confidence"] is None

def test_create_page():
    from google_format import create_page, create_block, create_paragraph, create_word, create_symbol
    
    symbols = create_symbol("Test")
    bounding_box = {"vertices": [{"x": 0, "y": 0}, {"x": 40, "y": 0}, {"x": 40, "y": 20}, {"x": 0, "y": 20}]}
    word = create_word(symbols, bounding_box)
    paragraph = create_paragraph([word])
    block = create_block([paragraph])
    page = create_page([block])
    
    assert page["property"]["detectedLanguages"][0]["languageCode"] == "vi"
    assert len(page["blocks"]) == 1
    assert page["confidence"] is None