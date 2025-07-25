"""
Email Parser Module
==================

Optimized email parsing for .eml and .msg files with attachment handling.
"""

from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
from dataclasses import dataclass
import email
import email.policy
from email.message import EmailMessage
import mimetypes

try:
    import mailparser
    MAILPARSER_AVAILABLE = True
except ImportError:
    MAILPARSER_AVAILABLE = False

try:
    import extract_msg
    EXTRACT_MSG_AVAILABLE = True
except ImportError:
    EXTRACT_MSG_AVAILABLE = False

from loguru import logger


@dataclass
class EmailAttachment:
    """Information about an email attachment."""
    filename: str
    content_type: str
    size_bytes: int
    data: Optional[bytes] = None
    is_inline: bool = False
    content_id: Optional[str] = None


@dataclass
class EmailMetadata:
    """Metadata extracted from email."""
    subject: Optional[str] = None
    sender: Optional[str] = None
    recipients: List[str] = None
    cc_recipients: List[str] = None
    bcc_recipients: List[str] = None
    date: Optional[str] = None
    message_id: Optional[str] = None
    in_reply_to: Optional[str] = None
    references: List[str] = None
    priority: Optional[str] = None
    total_attachments: int = 0
    has_html_body: bool = False
    has_text_body: bool = False
    
    def __post_init__(self):
        if self.recipients is None:
            self.recipients = []
        if self.cc_recipients is None:
            self.cc_recipients = []
        if self.bcc_recipients is None:
            self.bcc_recipients = []
        if self.references is None:
            self.references = []


@dataclass
class EmailParseResult:
    """Result of email parsing operation."""
    text_content: str
    html_content: str
    metadata: EmailMetadata
    attachments: List[EmailAttachment]
    structured_content: Dict[str, Any]
    confidence_score: float = 0.0
    extraction_method: str = ""
    error_message: Optional[str] = None


class EmailParser:
    """
    Optimized email parser supporting .eml and .msg formats.
    """
    
    def __init__(self):
        """Initialize email parser."""
        self.mailparser_available = MAILPARSER_AVAILABLE
        self.extract_msg_available = EXTRACT_MSG_AVAILABLE
        
        logger.info(f"Email Parser initialized - mailparser: {self.mailparser_available}, "
                   f"extract_msg: {self.extract_msg_available}")
    
    def parse_email(self, file_path: Path) -> EmailParseResult:
        """
        Parse email file (.eml or .msg).
        
        Args:
            file_path: Path to the email file
            
        Returns:
            EmailParseResult with extracted content and metadata
        """
        logger.info(f"Starting email parsing for {file_path}")
        
        file_ext = file_path.suffix.lower()
        
        if file_ext == '.msg':
            return self._parse_msg_file(file_path)
        elif file_ext == '.eml':
            return self._parse_eml_file(file_path)
        else:
            # Try to detect format by content
            return self._parse_unknown_email_format(file_path)
    
    def _parse_eml_file(self, file_path: Path) -> EmailParseResult:
        """Parse .eml file using standard email library."""
        try:
            # Try mailparser first if available
            if self.mailparser_available:
                try:
                    result = self._parse_with_mailparser(file_path)
                    if result.confidence_score > 0.7:
                        return result
                except Exception as e:
                    logger.warning(f"mailparser failed, falling back to standard library: {e}")
            
            # Fallback to standard email library
            return self._parse_with_email_library(file_path)
            
        except Exception as e:
            logger.error(f"EML parsing failed: {e}")
            return EmailParseResult(
                text_content="",
                html_content="",
                metadata=EmailMetadata(),
                attachments=[],
                structured_content={},
                error_message=str(e)
            )
    
    def _parse_msg_file(self, file_path: Path) -> EmailParseResult:
        """Parse .msg file using extract_msg library."""
        if not self.extract_msg_available:
            return EmailParseResult(
                text_content="",
                html_content="",
                metadata=EmailMetadata(),
                attachments=[],
                structured_content={},
                error_message="extract_msg library not available for .msg files"
            )
        
        try:
            msg = extract_msg.Message(file_path)
            
            # Extract basic content
            text_content = msg.body or ""
            html_content = msg.htmlBody or ""
            
            # Extract metadata
            metadata = self._extract_msg_metadata(msg)
            
            # Extract attachments
            attachments = self._extract_msg_attachments(msg)
            metadata.total_attachments = len(attachments)
            
            # Process image attachments with OCR if needed
            ocr_text = self._process_image_attachments(attachments)
            if ocr_text:
                text_content += "\n" + ocr_text
            
            # Create structured content
            structured_content = self._create_msg_structured_content(msg)
            
            # Calculate confidence
            confidence = self._calculate_confidence(text_content, html_content, metadata)
            
            logger.info(f"MSG parsing completed - {len(text_content)} text chars, {len(attachments)} attachments")
            
            return EmailParseResult(
                text_content=text_content,
                html_content=html_content,
                metadata=metadata,
                attachments=attachments,
                structured_content=structured_content,
                confidence_score=confidence,
                extraction_method="extract_msg"
            )
            
        except Exception as e:
            logger.error(f"MSG parsing failed: {e}")
            return EmailParseResult(
                text_content="",
                html_content="",
                metadata=EmailMetadata(),
                attachments=[],
                structured_content={},
                error_message=str(e)
            )
    
    def _parse_with_mailparser(self, file_path: Path) -> EmailParseResult:
        """Parse email using mailparser library."""
        mail = mailparser.parse_from_file(file_path)
        
        # Extract content
        text_content = mail.text_plain[0] if mail.text_plain else ""
        html_content = mail.text_html[0] if mail.text_html else ""
        
        # Extract metadata
        metadata = EmailMetadata(
            subject=mail.subject,
            sender=mail.from_[0][1] if mail.from_ else None,
            recipients=[addr[1] for addr in mail.to] if mail.to else [],
            cc_recipients=[addr[1] for addr in mail.cc] if mail.cc else [],
            bcc_recipients=[addr[1] for addr in mail.bcc] if mail.bcc else [],
            date=mail.date.isoformat() if mail.date else None,
            message_id=mail.message_id,
            has_html_body=bool(html_content),
            has_text_body=bool(text_content)
        )
        
        # Extract attachments
        attachments = self._extract_mailparser_attachments(mail)
        metadata.total_attachments = len(attachments)
        
        # Process image attachments with OCR if needed
        ocr_text = self._process_image_attachments(attachments)
        if ocr_text:
            text_content += "\n" + ocr_text
        
        # Create structured content
        structured_content = self._create_mailparser_structured_content(mail)
        
        # Calculate confidence
        confidence = self._calculate_confidence(text_content, html_content, metadata)
        
        return EmailParseResult(
            text_content=text_content,
            html_content=html_content,
            metadata=metadata,
            attachments=attachments,
            structured_content=structured_content,
            confidence_score=confidence,
            extraction_method="mailparser"
        )
    
    def _parse_with_email_library(self, file_path: Path) -> EmailParseResult:
        """Parse email using standard Python email library."""
        with open(file_path, 'rb') as f:
            msg = email.message_from_binary_file(f, policy=email.policy.default)
        
        # Extract content
        text_content = ""
        html_content = ""
        
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = part.get('Content-Disposition', '')
                
                if content_type == 'text/plain' and 'attachment' not in content_disposition:
                    text_content = part.get_content()
                elif content_type == 'text/html' and 'attachment' not in content_disposition:
                    html_content = part.get_content()
        else:
            content_type = msg.get_content_type()
            if content_type == 'text/plain':
                text_content = msg.get_content()
            elif content_type == 'text/html':
                html_content = msg.get_content()
        
        # Extract metadata
        metadata = self._extract_email_metadata(msg)
        
        # Extract attachments  
        attachments = self._extract_email_attachments(msg)
        metadata.total_attachments = len(attachments)
        
        # Process image attachments with OCR if needed
        ocr_text = self._process_image_attachments(attachments)
        if ocr_text:
            text_content += "\n" + ocr_text
            
        metadata.has_html_body = bool(html_content)
        metadata.has_text_body = bool(text_content)        # Create structured content
        structured_content = self._create_email_structured_content(msg)
        
        # Calculate confidence
        confidence = self._calculate_confidence(text_content, html_content, metadata)
        
        return EmailParseResult(
            text_content=text_content,
            html_content=html_content,
            metadata=metadata,
            attachments=attachments,
            structured_content=structured_content,
            confidence_score=confidence,
            extraction_method="email_library"
        )
    
    def _parse_unknown_email_format(self, file_path: Path) -> EmailParseResult:
        """Try to parse unknown email format."""
        # Try as EML first
        try:
            return self._parse_eml_file(file_path)
        except Exception as e:
            logger.warning(f"Failed to parse as EML: {e}")
        
        # If that fails, return error
        return EmailParseResult(
            text_content="",
            html_content="",
            metadata=EmailMetadata(),
            attachments=[],
            structured_content={},
            error_message=f"Unknown email format for {file_path}"
        )
    
    def _extract_email_metadata(self, msg: EmailMessage) -> EmailMetadata:
        """Extract metadata from email message."""
        return EmailMetadata(
            subject=msg.get('Subject'),
            sender=msg.get('From'),
            recipients=self._parse_email_addresses(msg.get('To', '')),
            cc_recipients=self._parse_email_addresses(msg.get('Cc', '')),
            bcc_recipients=self._parse_email_addresses(msg.get('Bcc', '')),
            date=msg.get('Date'),
            message_id=msg.get('Message-ID'),
            in_reply_to=msg.get('In-Reply-To'),
            references=self._parse_references(msg.get('References', '')),
            priority=msg.get('X-Priority')
        )
    
    def _extract_msg_metadata(self, msg) -> EmailMetadata:
        """Extract metadata from MSG object."""
        return EmailMetadata(
            subject=msg.subject,
            sender=msg.sender,
            recipients=self._parse_email_addresses(msg.to or ''),
            cc_recipients=self._parse_email_addresses(msg.cc or ''),
            bcc_recipients=self._parse_email_addresses(msg.bcc or ''),
            date=msg.date.isoformat() if hasattr(msg, 'date') and msg.date else None,
            message_id=getattr(msg, 'message_id', None),
            has_html_body=bool(getattr(msg, 'htmlBody', None)),
            has_text_body=bool(getattr(msg, 'body', None))
        )
    
    def _extract_email_attachments(self, msg: EmailMessage) -> List[EmailAttachment]:
        """Extract attachments from email message."""
        attachments = []
        
        for part in msg.walk():
            content_disposition = part.get('Content-Disposition', '')
            
            if 'attachment' in content_disposition or part.get_filename():
                filename = part.get_filename() or 'unknown_attachment'
                content_type = part.get_content_type()
                
                try:
                    data = part.get_content()
                    if isinstance(data, str):
                        data = data.encode('utf-8')
                    
                    attachment = EmailAttachment(
                        filename=filename,
                        content_type=content_type,
                        size_bytes=len(data) if data else 0,
                        data=data,
                        is_inline='inline' in content_disposition,
                        content_id=part.get('Content-ID')
                    )
                    attachments.append(attachment)
                    
                except Exception as e:
                    logger.warning(f"Failed to extract attachment {filename}: {e}")
                    continue
        
        return attachments
    
    def _extract_msg_attachments(self, msg) -> List[EmailAttachment]:
        """Extract attachments from MSG object."""
        attachments = []
        
        try:
            for attachment in msg.attachments:
                filename = attachment.longFilename or attachment.shortFilename or 'unknown'
                
                # Guess content type from filename
                content_type, _ = mimetypes.guess_type(filename)
                content_type = content_type or 'application/octet-stream'
                
                data = attachment.data
                
                email_attachment = EmailAttachment(
                    filename=filename,
                    content_type=content_type,
                    size_bytes=len(data) if data else 0,
                    data=data
                )
                attachments.append(email_attachment)
                
        except Exception as e:
            logger.warning(f"Failed to extract MSG attachments: {e}")
        
        return attachments
    
    def _process_image_attachments(self, attachments: List[EmailAttachment]) -> str:
        """Process image attachments with OCR if needed."""
        ocr_text = ""
        
        for attachment in attachments:
            if attachment.content_type.startswith('image/'):
                try:
                    from .ocr_parser import OCRParser
                    
                    # Save attachment temporarily for OCR
                    temp_path = Path(f"temp_{attachment.filename}")
                    temp_path.write_bytes(attachment.data)
                    
                    ocr_parser = OCRParser()
                    result = ocr_parser.parse_file(temp_path)
                    
                    if result.success and result.text_content:
                        ocr_text += f"\n[OCR from {attachment.filename}]\n{result.text_content}\n"
                        logger.info(f"Extracted text from image attachment: {attachment.filename}")
                    
                    # Clean up temp file
                    temp_path.unlink(missing_ok=True)
                    
                except Exception as e:
                    logger.warning(f"Failed to process image attachment {attachment.filename}: {e}")
        
        return ocr_text
    
    def _extract_mailparser_attachments(self, mail) -> List[EmailAttachment]:
        """Extract attachments using mailparser."""
        attachments = []
        
        for attachment in mail.attachments:
            email_attachment = EmailAttachment(
                filename=attachment.get('filename', 'unknown'),
                content_type=attachment.get('mail_content_type', 'application/octet-stream'),
                size_bytes=len(attachment.get('payload', b'')),
                data=attachment.get('payload')
            )
            attachments.append(email_attachment)
        
        return attachments
    
    def _parse_email_addresses(self, address_string: str) -> List[str]:
        """Parse email addresses from string."""
        if not address_string:
            return []
        
        # Simple parsing - can be enhanced
        addresses = [addr.strip() for addr in address_string.split(',')]
        return [addr for addr in addresses if addr]
    
    def _parse_references(self, references_string: str) -> List[str]:
        """Parse message references."""
        if not references_string:
            return []
        
        return [ref.strip() for ref in references_string.split() if ref.strip()]
    
    def _create_email_structured_content(self, msg: EmailMessage) -> Dict[str, Any]:
        """Create structured content from email message."""
        return {
            "headers": dict(msg.items()),
            "multipart": msg.is_multipart(),
            "parts": len(list(msg.walk())) if msg.is_multipart() else 1,
            "content_types": [part.get_content_type() for part in msg.walk()]
        }
    
    def _create_msg_structured_content(self, msg) -> Dict[str, Any]:
        """Create structured content from MSG object."""
        return {
            "has_attachments": len(getattr(msg, 'attachments', [])) > 0,
            "attachment_count": len(getattr(msg, 'attachments', [])),
            "has_html": bool(getattr(msg, 'htmlBody', None)),
            "has_text": bool(getattr(msg, 'body', None))
        }
    
    def _create_mailparser_structured_content(self, mail) -> Dict[str, Any]:
        """Create structured content from mailparser object."""
        return {
            "has_attachments": len(mail.attachments) > 0,
            "attachment_count": len(mail.attachments),
            "has_html": bool(mail.text_html),
            "has_text": bool(mail.text_plain),
            "headers": mail.headers
        }
    
    def _calculate_confidence(self, text_content: str, html_content: str, 
                            metadata: EmailMetadata) -> float:
        """Calculate confidence score for email extraction."""
        confidence_factors = []
        
        # Factor 1: Content presence
        has_content = bool(text_content.strip() or html_content.strip())
        confidence_factors.append(0.8 if has_content else 0.2)
        
        # Factor 2: Metadata completeness
        metadata_score = 0
        if metadata.subject:
            metadata_score += 0.3
        if metadata.sender:
            metadata_score += 0.3
        if metadata.recipients:
            metadata_score += 0.2
        if metadata.date:
            metadata_score += 0.2
        
        confidence_factors.append(metadata_score)
        
        # Factor 3: Structure
        if metadata.has_text_body or metadata.has_html_body:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.3)
        
        return sum(confidence_factors) / len(confidence_factors)
    
    def save_attachments(self, attachments: List[EmailAttachment], 
                        output_dir: Path) -> List[Path]:
        """
        Save email attachments to disk.
        
        Args:
            attachments: List of email attachments
            output_dir: Directory to save attachments
            
        Returns:
            List of paths to saved attachments
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_paths = []
        
        for i, attachment in enumerate(attachments):
            try:
                if attachment.data:
                    # Create safe filename
                    safe_filename = self._make_safe_filename(attachment.filename)
                    if not safe_filename:
                        safe_filename = f"attachment_{i:03d}"
                    
                    output_path = output_dir / safe_filename
                    
                    # Handle duplicate filenames
                    counter = 1
                    original_path = output_path
                    while output_path.exists():
                        stem = original_path.stem
                        suffix = original_path.suffix
                        output_path = output_dir / f"{stem}_{counter}{suffix}"
                        counter += 1
                    
                    with open(output_path, 'wb') as f:
                        f.write(attachment.data)
                    
                    saved_paths.append(output_path)
                    logger.debug(f"Saved attachment: {output_path}")
            
            except Exception as e:
                logger.warning(f"Failed to save attachment {attachment.filename}: {e}")
                continue
        
        logger.info(f"Saved {len(saved_paths)} attachments to {output_dir}")
        return saved_paths
    
    def _make_safe_filename(self, filename: str) -> str:
        """Make filename safe for filesystem."""
        import re
        # Remove or replace unsafe characters
        safe_filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        return safe_filename.strip()


def parse_email(file_path: Path) -> EmailParseResult:
    """
    Convenience function to parse an email file.
    
    Args:
        file_path: Path to the email file
        
    Returns:
        EmailParseResult with extracted content
    """
    parser = EmailParser()
    return parser.parse_email(file_path)
