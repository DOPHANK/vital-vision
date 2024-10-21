from ocr_mapper import OCRMapper
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/utils')))
from config import setup_logger

logger = setup_logger('pipeline')
class EasyOCRMapper(OCRMapper):
	@logger.catch
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
		logger.info("Starting extract_ocr_properties")
		result_list = []
		for data in ocr_result:
			if not isinstance(data, tuple) or len(data) != 3:
				logger.error("Invalid OCR result format: {}", data)
				continue
			vertices_list, description, confidence = data
			if not isinstance(vertices_list, list) or not isinstance(description, str) or not isinstance(confidence, float):
				logger.error("Invalid OCR result format: {}", data)
				continue
			# vertices_list, description, confidence = data
			vertices = [{"x": x, "y": y} for x, y in vertices_list]
			
			result = {
				"description": description,
				"boundingPoly": {
					"vertices": vertices
				},
				"confidence": confidence
			}
			
			result_list.append(result)
	
		# logger.debug(f"Processed OCR data: {result}")
		logger.info("Finishing extract_ocr_properties")
		return result_list
	
if __name__ == "__main__":
    ocr_mapper = EasyOCRMapper()
    # sample_ocr_result = [
    #     ([[454, 214], [629, 214], [629, 290], [454, 290]], 'HSR', 0.9931),
    #     ([[664, 222], [925, 222], [925, 302], [664, 302]], 'Station', 0.3260)
    # ]
    sample_ocr_result = []
    output = ocr_mapper.extract_ocr_properties(sample_ocr_result)
    print(output)