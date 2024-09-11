
from abc import ABC, abstractmethod

class OCRMapper(ABC):
    def __init__(self):
        """
        Initialize the OCRMapper with a GoogleVisionMapper instance.
        """
        self.google_vision_mapper = GoogleVisionMapper()
    
    @abstractmethod
    def extract_ocr_properties(self, ocr_result: list) -> dict:
        """
        Extract text, coordinates, and confidence from the OCR result.

        This abstract method should be implemented by subclasses for each specific OCR model.

        Args:
            ocr_result (list): The raw OCR result from a specific OCR model.

        Returns:
            dict: A dictionary containing extracted properties (text, coordinates, confidence).

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        pass
    
    def map_ocr_result(self, ocr_result: dict) -> dict:
        """
        Map the OCR result to Google Vision API format.

        This method extracts properties using the model-specific implementation
        and then delegates the final mapping to the GoogleVisionMapper.

        Args:
            ocr_result (dict): The raw OCR result from a specific OCR model.

        Returns:
            dict: The OCR result mapped to Google Vision API format.
        """
        # Extract properties from ocr model
        extracted_ocr = self.extract_ocr_properties(ocr_result)
        
        # Delegate the final mapping to the GoogleVisionMapper
        return self.google_vision_mapper.map_to_google_vision(extracted_ocr)




class EasyOCRMapper(OCRMapper):
	def extract_ocr_properties(self, ocr_result: list) -> dict:
		"""
        Extract fields: text, coordinates, and confidence from EasyOCR results.

        Args:
            ocr_result (list): The output from EasyOCR.

        Returns:
            dict: A dictionary containing the extracted properties.

        Example of EasyOCR output:
        [
            ([[454, 214], [629, 214], [629, 290], [454, 290]], 'HSR', 0.9931),
            ([[664, 222], [925, 222], [925, 302], [664, 302]], 'Station', 0.3260)
        ]
        """
		result_list = []
		for data in ocr_result:
			vertices_list, description, confidence = data
			vertices = [{"x": x, "y": y} for x, y in vertices_list]
			
			result = {
				"description": description,
				"boundingPoly": {
					"vertices": vertices
				},
				"confidence": confidence
			}
			
			result_list.append(result)
    
		return result_list
		
		
class GoogleVisionMapper:
    def format_text_annotations(self, ocr_result: list) -> list:
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

        
    def format_full_text_annotation(self, ocr_result: list) -> dict:
        symbols = []
        for item in ocr_result:
            symbol = []
            for index, char in enumerate(item["description"]):
                
                if index == len(item["description"]) - 1:
                    character = {
                        'property': {
                            'detectedBreak': {
                              'type': 'SPACE'
                            }
                          },
                        'boundingBox': {None},
                        'text': char,
                        'confidence': None
                    }
                else:
                    character = {
                        'boundingBox': {None},
                        'text': char,
                        'confidence': None
                    }
                symbol.append(character) 
            vertices = {"vertices": item["boundingPoly"]["vertices"] }
            symbols.append([vertices, symbol])
        
        words = []
        for i in range(len(symbols)):
            word = {
                "property":{
                    "detectedLanguages": [
                    {
                        "languageCode": "",
                        "confidence": None
                    }
                   ]
                },
                "boundingBox": symbols[i][0],
                "symbols": symbols[i][1],
                "confidence": None
            }
            words.append(word)
            
        full_text_annotation = {
            "pages": [
                {
                    "property":{
                      "detectedLanguages": [
                        {
                          "languageCode": "en",
                          "confidence": 1
                        }
                      ]
                    },
                    "width": None,
                    "height": None,
                    "blocks": [{
                        "boundingBox": {None},  
                        "paragraphs": [
                        {
                            "boundingBox": {None},
                            "words": words,
                            "confidence": None
                        }
                        ],
                        "blockType": "TEXT",
                        "confidence": None
                    }],
                    "confidence": None
                }
            ],
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
        

        
