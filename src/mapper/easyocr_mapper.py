from ocr_mapper import OCRMapper

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
	
