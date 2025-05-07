import base64
import json
import os
import re
from io import BytesIO
from typing import Dict, Optional, Union

import pdfplumber
import pytesseract
import requests
from langchain_core.tools import tool
from PIL import Image


@tool
def analyze_image(input_image: Union[str, Image.Image]) -> str:
    """
    Extract text from an image using OCR and AI-based processing.

    Args:
        input_image: Path to image file (str) or already loaded PIL image.

    Returns:
        Extracted text or an error message.
    """
    try:
        extractor = RobustTextExtractor(os.getenv("OPENROUTER_API_KEY"))
        return extractor.get_text_from_image(input_image)
    except Exception as e:
        return f"Image analysis failed: {str(e)}"


@tool
def pdf_text_extractor(file_path: str) -> str:
    """
    Extracts and returns all text content from a local PDF file.

    Args:
        file_path (str): Path to the PDF file on the local filesystem.

    Returns:
        str: Concatenated text from every page, prefixed with the tool name,
             or an error message if validation/loading fails.
    """
    # 1. Validate file existence and extension
    if not os.path.isfile(file_path):
        return f"[Tool: pdf_text_extractor ERROR] File not found: {file_path}"
    if not file_path.lower().endswith(".pdf"):
        return f"[Tool: pdf_text_extractor ERROR] Unsupported file type: {file_path}"

    try:
        # 2. Open and extract text from every page
        text_chunks = []
        with pdfplumber.open(
            file_path
        ) as pdf:  # uses pdfplumber for page parsing :contentReference[oaicite:2]{index=2}
            if not pdf.pages:
                return f"[Tool: pdf_text_extractor] No pages found in {file_path}"
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text_chunks.append(page_text)

        # 3. Concatenate and return
        all_text = "\n\n".join(text_chunks)
        return f"[Tool: pdf_text_extractor]\n{all_text}"

    except Exception as e:
        # 4. Handle unexpected errors
        return f"[Tool: pdf_text_extractor ERROR] Failed to process {file_path}: {e}"


class RobustTextExtractor:
    """
    Enhanced image text extraction with:
    - pytesseract OCR fallback
    - Proper OpenRouter API error handling
    - Structured JSON output
    """

    def __init__(self, openrouter_api_key: Optional[str] = None):
        self.ocr_config = r"--oem 3 --psm 11"
        self.openrouter_api_key = openrouter_api_key
        self.vlm_enabled = bool(openrouter_api_key)

        # Configure VLM settings
        self.vlm_endpoint = "https://openrouter.ai/api/v1/chat/completions"
        self.vlm_headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "HTTP-Referer": "https://your-site-url.com",  # Required by OpenRouter
            "X-Title": "GAIA Image Processing",  # Optional but recommended
        }

        # Optimized VLM prompt
        self.vlm_prompt_template = """Extract and analyze text from this image:
        1. List all text elements with positions (x,y)
        2. Identify text hierarchy (headings, paragraphs)
        3. Note any ambiguous characters
        4. Output as JSON with:
        {
            "text_blocks": [{"text": "...", "position": [x,y,w,h]}],
            "formatting": {"bold": [...], "italic": [...]},
            "confidence": 0-1
        }"""

    def get_text_from_image(self, image_input) -> Dict[str, Union[str, dict]]:
        """
        Extract text with OCR + VLM (if available)

        Returns:
            {
                "ocr_text": "...",
                "vlm_analysis": {...} or None,
                "error": None or str,
                "combined_text": "..."
            }
        """
        result = {
            "ocr_text": "",
            "vlm_analysis": None,
            "error": None,
            "combined_text": "",
        }

        try:
            img = None
            if isinstance(image_input, str):
                # Preprocess image
                img = self._preprocess_image(image_input)
            else:
                img = image_input

            # Always run OCR
            ocr_data = self._run_ocr(img)
            result["ocr_text"] = ocr_data["text"]

            # Conditionally run VLM
            if self.vlm_enabled:
                try:
                    vlm_data = self._analyze_with_vlm(img)
                    if vlm_data:
                        result["vlm_analysis"] = vlm_data
                except Exception as vlm_error:
                    result["error"] = f"VLM analysis failed: {str(vlm_error)}"

            # Generate combined output
            result["combined_text"] = self._combine_results(result)

        except Exception as e:
            result["error"] = f"Processing failed: {str(e)}"

        return result

    def _preprocess_image(self, image_path: str) -> Image.Image:
        """Optimize image for text extraction"""
        image_path = re.sub(
            r"^(.*?\.(?:png|jpe?g)).*",
            r"\1",
            image_path,
            flags=re.DOTALL | re.IGNORECASE,
        )
        img = Image.open(image_path)
        return img.convert("L").point(lambda x: 0 if x < 128 else 255, "1")

    def _run_ocr(self, image: Image.Image) -> dict:
        """Run pytesseract with layout analysis"""
        data = pytesseract.image_to_data(
            image, config=self.ocr_config, output_type=pytesseract.Output.DICT
        )

        return {
            "text": pytesseract.image_to_string(image),
            "confidence": self._calculate_confidence(data["conf"]),
            "position_data": self._extract_position_data(data),
        }

    def _analyze_with_vlm(self, image: Image.Image) -> Optional[dict]:
        """Call OpenRouter VLM endpoint with proper error handling"""
        try:
            # Encode image
            img_base64 = self._image_to_base64(image)

            # Build payload according to OpenRouter specs :cite[3]
            payload = {
                "model": "qwen/qwen2.5-vl-3b-instruct:free",
                "messages": [
                    {"role": "system", "content": self.vlm_prompt_template},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": f"data:image/png;base64,{img_base64}",
                            }
                        ],
                    },
                ],
                # Reduced from 4000 to prevent 400 errors :cite[1]
                "max_tokens": 2000,
                "temperature": 0.2,
            }

            response = requests.post(
                self.vlm_endpoint, headers=self.vlm_headers, json=payload, timeout=30
            )

            # Handle API errors :cite[3]
            if response.status_code != 200:
                error_msg = (
                    response.json().get("error", {}).get("message", "Unknown VLM error")
                )
                raise Exception(f"API Error {response.status_code}: {error_msg}")

            return self._parse_vlm_response(response.json())

        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {str(e)}")

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL image to base64 string"""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    def _parse_vlm_response(self, response: dict) -> dict:
        """Extract structured data from VLM response"""
        try:
            content = response["choices"][0]["message"]["content"]
            if content.strip().startswith("{"):
                return json.loads(content)
            return {"raw_response": content}
        except Exception:
            return {"raw_response": "Could not parse VLM output"}

    def _calculate_confidence(self, conf_values: list) -> float:
        """Calculate average OCR confidence"""
        valid_confs = [float(c) for c in conf_values if c != "-1"]
        return sum(valid_confs) / len(valid_confs) if valid_confs else 0

    def _extract_position_data(self, ocr_data: dict) -> list:
        """Format OCR position data"""
        return [
            {
                "text": ocr_data["text"][i],
                "position": [
                    ocr_data["left"][i],
                    ocr_data["top"][i],
                    ocr_data["width"][i],
                    ocr_data["height"][i],
                ],
                "confidence": float(ocr_data["conf"][i]),
            }
            for i in range(len(ocr_data["text"]))
        ]

    def _parse_vlm_response(self, response: dict) -> dict:
        """Improved parsing of VLM responses with Markdown cleanup"""
        try:
            content = response["choices"][0]["message"]["content"]

            # Remove JSON code block formatting
            content = content.replace("```json", "").replace("```", "").strip()

            try:
                # Attempt to parse cleaned JSON
                parsed = json.loads(content)
                return {
                    "text_blocks": parsed.get("text_blocks", []),
                    "formatting": parsed.get("formatting", {}),
                    "confidence": parsed.get("confidence", 0.5),
                    "raw_json": parsed,
                }
            except json.JSONDecodeError:
                # Return cleaned text if JSON parsing fails
                return {"text_summary": content.split("\n", 1)[-1]}

        except Exception:
            return {"error": "Could not parse VLM output"}

    def _combine_results(self, results: dict) -> str:
        """Improved human-readable formatting"""
        output = []

        if results.get("vlm_analysis"):
            output.append("## Visual Language Model Analysis")

            analysis = results["vlm_analysis"]
            if "text_blocks" in analysis:
                output.append("### Text Elements:")
                for idx, block in enumerate(analysis["text_blocks"][:5], 1):
                    output.append(f"{idx}. {block.get('text', '')}")
                    output.append(f"   Position: {block.get('position', 'N/A')}")

            if "formatting" in analysis:
                output.append("\n### Text Formatting:")
                for style, items in analysis["formatting"].items():
                    output.append(f"- {style.capitalize()}: {len(items)} elements")

            if "text_summary" in analysis:
                output.append("\nSummary: " + analysis["text_summary"])

        output.append("\n## OCR Results")
        output.append(results["ocr_text"])

        if results.get("error"):
            output.append(f"\n[Note: {results['error']}]")

        return "\n".join(output)
