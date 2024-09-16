
class GoogleVisionMapper:
    @staticmethod
    def format_text_annotations(ocr_result: list) -> list:
        """
        Format OCR results into Google Vision API-like text annotations.

        This method takes a list of OCR results and formats them to match
        the structure of Google Vision API's text annotations.

        Args:
            ocr_result (list): A list of dictionaries containing OCR results.
                Each dictionary should have 'description' and 'boundingPoly' keys.

        Returns:
            list: A list of formatted text annotations. The first item contains
            the full text description, followed by individual text annotations.

        Note:
            - The first item in the returned list has empty 'locale' and null coordinates.
            - Individual annotations do not include confidence scores.
        """
        full_description = " ".join([item['description'] for item in ocr_result])
        
        # Create the first part of textAnnotations with full description
        text_annotations = [
            {
                "locale": "",
                "description": full_description,
                "boundingPoly": {
                    "vertices": [[{"x": None, "y": None} for _ in range(4)]]
                }
            }
        ]
        
        # Add individual text annotations without confidence
        for item in ocr_result:
            annotation = {
                "description": item['description'],
                "boundingPoly": {
                    "vertices": item['boundingPoly']['vertices']
                }
            }
            
            text_annotations.append(annotation)
        
        return text_annotations

    @staticmethod
    def format_full_text_annotation(ocr_result: list) -> dict:
        """Formats OCR result into full text annotation."""
        words = []
        
        for item in ocr_result:
            symbols = create_symbol(item["description"])
            bounding_box = {"vertices": item["boundingPoly"]["vertices"]}
            word = create_word(symbols, bounding_box)
            words.append(word)
        
        # Create a paragraph from words
        paragraph = create_paragraph(words)
        
        # Create a block from paragraphs
        block = create_block([paragraph])
        
        # Create a page from blocks
        page = create_page([block])
        
        # Compile full text annotation
        full_text_annotation = {
            "pages": [page],
            "text": " ".join([item['description'] for item in ocr_result])
        }

        return full_text_annotation

    def map_to_google_vision(self, ocr_result: list) -> dict:
        text_annotations = self.format_text_annotations(ocr_result)
        full_text_annotation = self.format_full_text_annotation(ocr_result)
        
        result = {
            "responses": [
                {
                    "textAnnotations": text_annotations,
                    "fullTextAnnotation": full_text_annotation
                }
            ]
        }
        
        return result
    
def create_symbol(description: str) -> list:
    """Creates a list of symbols from the description."""
    return [
        {
            **({'property': {
                    'detectedBreak': {'type': 'SPACE'}
                }} if i == len(description) - 1 else {}),
            'boundingBox': {None},
            'text': char,
            'confidence': None
        } for i,char in enumerate(description)
    ]

def create_word(symbols: list, bounding_box: dict) -> dict:
    """Creates a word object from the given symbols and bounding box."""
    return {
        "property": {
            "detectedLanguages": [
                {"languageCode": "vi", "confidence": None}
            ]
        },
        "boundingBox": bounding_box,
        "symbols": symbols,
        "confidence": None
    }

def create_paragraph(words: list) -> dict:
    """Creates a paragraph object from the given words."""
    return {
        "boundingBox": {None},
        "words": words,
        "confidence": None
    }

def create_block(paragraphs: list) -> dict:
    """Creates a block object from the given paragraphs."""
    return {
        "boundingBox": {None},
        "paragraphs": paragraphs,
        "blockType": "TEXT",
        "confidence": None
    }

def create_page(blocks: list) -> dict:
    """Creates a page object from the given blocks."""
    return {
        "property": {
            "detectedLanguages": [
                {"languageCode": "vi", "confidence": None}
            ]
        },
        "width": None,
        "height": None,
        "blocks": blocks,
        "confidence": None
    }