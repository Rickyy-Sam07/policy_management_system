"""
OCR Parser Module
================

Optimized OCR processing for images using multiple OCR engines.
"""

from typing import Optional, Dict, Any, List, Tuple, Union
from pathlib import Path
from dataclasses import dataclass
import io

try:
    from PIL import Image, ImageEnhance, ImageFilter
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

from loguru import logger


@dataclass
class OCRBoundingBox:
    """Bounding box for OCR text detection."""
    x: int
    y: int
    width: int
    height: int
    confidence: float


@dataclass
class OCRTextBlock:
    """OCR detected text block."""
    text: str
    confidence: float
    bounding_box: OCRBoundingBox
    language: Optional[str] = None


@dataclass
class ImageMetadata:
    """Metadata about the processed image."""
    width: int
    height: int
    channels: int
    format: str
    mode: Optional[str] = None
    has_transparency: bool = False
    file_size: int = 0


@dataclass
class OCRParseResult:
    """Result of OCR parsing operation."""
    text_content: str
    text_blocks: List[OCRTextBlock]
    metadata: ImageMetadata
    confidence_score: float = 0.0
    preprocessing_applied: List[str] = None
    extraction_method: str = ""
    detected_languages: List[str] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.preprocessing_applied is None:
            self.preprocessing_applied = []
        if self.detected_languages is None:
            self.detected_languages = []


class OCRParser:
    """
    Optimized OCR parser with multiple engines and preprocessing.
    """
    
    def __init__(self, prefer_paddle: bool = True, lang: str = 'en'):
        """
        Initialize OCR parser.
        
        Args:
            prefer_paddle: Whether to prefer PaddleOCR over Tesseract
            lang: Language for OCR (default: 'en')
        """
        self.prefer_paddle = prefer_paddle
        self.lang = lang
        
        # Check available libraries
        self.pil_available = PIL_AVAILABLE
        self.opencv_available = OPENCV_AVAILABLE
        self.paddleocr_available = PADDLEOCR_AVAILABLE
        self.tesseract_available = TESSERACT_AVAILABLE
        
        if not (self.paddleocr_available or self.tesseract_available):
            raise ImportError("Neither PaddleOCR nor Tesseract is available")
        
        # Initialize OCR engines
        self.paddle_ocr = None
        if self.paddleocr_available:
            try:
                self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)
                logger.info("PaddleOCR initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize PaddleOCR: {e}")
                self.paddleocr_available = False
        
        logger.info(f"OCR Parser initialized - PaddleOCR: {self.paddleocr_available}, "
                   f"Tesseract: {self.tesseract_available}, PIL: {self.pil_available}, "
                   f"OpenCV: {self.opencv_available}")
    
    def parse_image(self, file_path: Path, preprocess: bool = True) -> OCRParseResult:
        """
        Parse image using OCR.
        
        Args:
            file_path: Path to the image file
            preprocess: Whether to apply image preprocessing
            
        Returns:
            OCRParseResult with extracted text and metadata
        """
        logger.info(f"Starting OCR processing for {file_path}")
        
        try:
            # Load and analyze image
            image, metadata = self._load_image(file_path)
            
            # Apply preprocessing if requested
            preprocessing_steps = []
            if preprocess:
                image, preprocessing_steps = self._preprocess_image(image)
            
            # Choose OCR engine and perform extraction
            if self.prefer_paddle and self.paddleocr_available:
                try:
                    result = self._ocr_with_paddle(image, metadata, preprocessing_steps)
                    if result.confidence_score > 0.6:
                        logger.info(f"PaddleOCR successful with confidence {result.confidence_score:.2f}")
                        return result
                    else:
                        logger.info(f"PaddleOCR low confidence, trying Tesseract")
                except Exception as e:
                    logger.warning(f"PaddleOCR failed: {e}")
            
            # Fallback to Tesseract or use it as primary
            if self.tesseract_available:
                try:
                    result = self._ocr_with_tesseract(image, metadata, preprocessing_steps)
                    logger.info(f"Tesseract completed with confidence {result.confidence_score:.2f}")
                    return result
                except Exception as e:
                    logger.error(f"Tesseract also failed: {e}")
                    return OCRParseResult(
                        text_content="",
                        text_blocks=[],
                        metadata=metadata,
                        error_message=f"All OCR engines failed: {e}"
                    )
            
            return OCRParseResult(
                text_content="",
                text_blocks=[],
                metadata=metadata,
                error_message="No OCR engines available"
            )
            
        except Exception as e:
            logger.error(f"Image OCR failed: {e}")
            return OCRParseResult(
                text_content="",
                text_blocks=[],
                metadata=ImageMetadata(0, 0, 0, "unknown"),
                error_message=str(e)
            )
    
    def _load_image(self, file_path: Path) -> Tuple[Image.Image, ImageMetadata]:
        """Load image and extract metadata."""
        if not self.pil_available:
            raise ImportError("PIL is required for image loading")
        
        image = Image.open(file_path)
        
        # Extract metadata
        metadata = ImageMetadata(
            width=image.width,
            height=image.height,
            channels=len(image.getbands()),
            format=image.format or "unknown",
            mode=image.mode,
            has_transparency='transparency' in image.info or image.mode in ('RGBA', 'LA'),
            file_size=file_path.stat().st_size
        )
        
        # Convert to RGB if necessary
        if image.mode in ('RGBA', 'LA', 'P'):
            image = image.convert('RGB')
        elif image.mode == 'L':
            # Keep grayscale for OCR
            pass
        
        logger.debug(f"Loaded image: {metadata.width}x{metadata.height}, {metadata.channels} channels, {metadata.format}")
        return image, metadata
    
    def _preprocess_image(self, image: Image.Image) -> Tuple[Image.Image, List[str]]:
        """
        Apply preprocessing to improve OCR accuracy.
        
        Args:
            image: PIL Image to preprocess
            
        Returns:
            Tuple of (processed_image, preprocessing_steps_applied)
        """
        preprocessing_steps = []
        processed_image = image.copy()
        
        try:
            # Convert to grayscale if not already
            if processed_image.mode != 'L':
                processed_image = processed_image.convert('L')
                preprocessing_steps.append("convert_to_grayscale")
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(processed_image)
            processed_image = enhancer.enhance(1.5)
            preprocessing_steps.append("enhance_contrast")
            
            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(processed_image)
            processed_image = enhancer.enhance(1.3)
            preprocessing_steps.append("enhance_sharpness")
            
            # Apply denoising filter
            processed_image = processed_image.filter(ImageFilter.MedianFilter(size=3))
            preprocessing_steps.append("median_filter")
            
            # Scale up if image is too small (OCR works better on larger images)
            if min(processed_image.size) < 300:
                scale_factor = 300 / min(processed_image.size)
                new_size = (int(processed_image.width * scale_factor), 
                           int(processed_image.height * scale_factor))
                processed_image = processed_image.resize(new_size, Image.Resampling.LANCZOS)
                preprocessing_steps.append(f"upscale_{scale_factor:.1f}x")
            
            logger.debug(f"Applied preprocessing: {', '.join(preprocessing_steps)}")
            
        except Exception as e:
            logger.warning(f"Preprocessing failed: {e}")
            # Return original image if preprocessing fails
            return image, []
        
        return processed_image, preprocessing_steps
    
    def _ocr_with_paddle(self, image: Image.Image, metadata: ImageMetadata, 
                        preprocessing_steps: List[str]) -> OCRParseResult:
        """Perform OCR using PaddleOCR."""
        # Convert PIL Image to numpy array for PaddleOCR
        if self.opencv_available:
            img_array = np.array(image)
            if len(img_array.shape) == 2:  # Grayscale
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        else:
            # Fallback: save to bytes and reload
            img_bytes = io.BytesIO()
            image.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            img_array = img_bytes.getvalue()
        
        # Run PaddleOCR
        results = self.paddle_ocr.ocr(img_array, cls=True)
        
        # Process results
        text_blocks = []
        all_text = []
        total_confidence = 0
        valid_blocks = 0
        
        if results and results[0]:
            for line in results[0]:
                if line and len(line) >= 2:
                    # Extract bounding box and text info
                    bbox = line[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    text_info = line[1]  # (text, confidence)
                    
                    if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                        text, confidence = text_info[0], text_info[1]
                        
                        if text and text.strip():
                            # Calculate bounding box
                            x_coords = [point[0] for point in bbox]
                            y_coords = [point[1] for point in bbox]
                            
                            x = int(min(x_coords))
                            y = int(min(y_coords))
                            width = int(max(x_coords) - min(x_coords))
                            height = int(max(y_coords) - min(y_coords))
                            
                            bbox_obj = OCRBoundingBox(x, y, width, height, confidence)
                            
                            text_block = OCRTextBlock(
                                text=text.strip(),
                                confidence=confidence,
                                bounding_box=bbox_obj,
                                language=self.lang
                            )
                            
                            text_blocks.append(text_block)
                            all_text.append(text.strip())
                            total_confidence += confidence
                            valid_blocks += 1
        
        # Calculate overall confidence
        overall_confidence = total_confidence / valid_blocks if valid_blocks > 0 else 0.0
        
        # Combine all text
        combined_text = "\n".join(all_text)
        
        return OCRParseResult(
            text_content=combined_text,
            text_blocks=text_blocks,
            metadata=metadata,
            confidence_score=overall_confidence,
            preprocessing_applied=preprocessing_steps,
            extraction_method="paddleocr",
            detected_languages=[self.lang]
        )
    
    def _ocr_with_tesseract(self, image: Image.Image, metadata: ImageMetadata, 
                          preprocessing_steps: List[str]) -> OCRParseResult:
        """Perform OCR using Tesseract."""
        try:
            # Get text with confidence data
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, lang=self.lang)
            
            # Process results
            text_blocks = []
            all_text = []
            total_confidence = 0
            valid_blocks = 0
            
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                confidence = int(data['conf'][i])
                
                if text and confidence > 0:  # Filter out low confidence detections
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    
                    bbox = OCRBoundingBox(x, y, w, h, confidence / 100.0)
                    
                    text_block = OCRTextBlock(
                        text=text,
                        confidence=confidence / 100.0,
                        bounding_box=bbox,
                        language=self.lang
                    )
                    
                    text_blocks.append(text_block)
                    all_text.append(text)
                    total_confidence += confidence / 100.0
                    valid_blocks += 1
            
            # Also get clean text
            clean_text = pytesseract.image_to_string(image, lang=self.lang).strip()
            
            # Use clean text if it's longer than combined blocks
            if len(clean_text) > len(" ".join(all_text)):
                combined_text = clean_text
            else:
                combined_text = " ".join(all_text)
            
            # Calculate overall confidence
            overall_confidence = total_confidence / valid_blocks if valid_blocks > 0 else 0.0
            
            return OCRParseResult(
                text_content=combined_text,
                text_blocks=text_blocks,
                metadata=metadata,
                confidence_score=overall_confidence,
                preprocessing_applied=preprocessing_steps,
                extraction_method="tesseract",
                detected_languages=[self.lang]
            )
            
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}")
            raise
    
    def batch_ocr(self, image_paths: List[Path], preprocess: bool = True) -> List[OCRParseResult]:
        """
        Perform OCR on multiple images.
        
        Args:
            image_paths: List of image file paths
            preprocess: Whether to apply preprocessing
            
        Returns:
            List of OCRParseResult objects
        """
        results = []
        logger.info(f"Starting batch OCR for {len(image_paths)} images")
        
        for image_path in image_paths:
            try:
                result = self.parse_image(image_path, preprocess)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch OCR failed for {image_path}: {e}")
                error_result = OCRParseResult(
                    text_content="",
                    text_blocks=[],
                    metadata=ImageMetadata(0, 0, 0, "unknown"),
                    error_message=str(e)
                )
                results.append(error_result)
        
        logger.info(f"Batch OCR completed: {len(results)} results")
        return results
    



def parse_image_ocr(file_path: Path, lang: str = 'en', preprocess: bool = True) -> OCRParseResult:
    """
    Convenience function to perform OCR on an image.
    
    Args:
        file_path: Path to the image file
        lang: Language for OCR
        preprocess: Whether to apply image preprocessing
        
    Returns:
        OCRParseResult with extracted text
    """
    parser = OCRParser(lang=lang)
    return parser.parse_image(file_path, preprocess)
