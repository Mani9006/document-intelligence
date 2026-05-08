"""
Named Entity Extraction Module.

Extracts structured entities from document text using regex patterns
and rule-based NER-like logic. Supports dates, monetary amounts,
person names, organization names, email addresses, phone numbers,
and custom entity types.

The module provides a pluggable architecture where new entity extractors
can be registered and combined for comprehensive information extraction.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Pattern, Tuple

logger = logging.getLogger(__name__)


class EntityType(str, Enum):
    """Supported entity types for extraction."""

    DATE = "date"
    AMOUNT = "amount"
    PERSON = "person"
    ORGANIZATION = "organization"
    EMAIL = "email"
    PHONE = "phone"
    ADDRESS = "address"
    INVOICE_NUMBER = "invoice_number"
    PO_NUMBER = "po_number"


@dataclass
class ExtractedEntity:
    """A single extracted entity with metadata."""

    text: str
    entity_type: EntityType
    start_pos: int
    end_pos: int
    confidence: float = 1.0
    normalized_value: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to a serializable dictionary."""
        return {
            "text": self.text,
            "type": self.entity_type.value,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "confidence": round(self.confidence, 4),
            "normalized_value": self.normalized_value,
            "metadata": self.metadata,
        }


class BaseExtractor(ABC):
    """Abstract base class for entity extractors."""

    @abstractmethod
    def extract(self, text: str) -> List[ExtractedEntity]:
        """Extract entities from the given text."""
        ...

    @abstractmethod
    def get_entity_type(self) -> EntityType:
        """Return the entity type this extractor handles."""
        ...


class DateExtractor(BaseExtractor):
    """
    Extract dates from text using multiple date format patterns.

    Supports standard formats: MM/DD/YYYY, DD-MM-YYYY, YYYY-MM-DD,
    Month DD, YYYY, etc. Also handles relative date references.
    """

    # Comprehensive date pattern collection
    PATTERNS: List[Tuple[str, Pattern[str], float]] = [
        # MM/DD/YYYY or MM-DD-YYYY
        (
            "us_numeric",
            re.compile(
                r"\b(0[1-9]|1[0-2])[/-](0[1-9]|[12][0-9]|3[01])[/-](\d{4})\b"
            ),
            0.95,
        ),
        # DD/MM/YYYY or DD-MM-YYYY
        (
            "eu_numeric",
            re.compile(
                r"\b(0[1-9]|[12][0-9]|3[01])[/-](0[1-9]|1[0-2])[/-](\d{4})\b"
            ),
            0.90,
        ),
        # YYYY-MM-DD (ISO format)
        (
            "iso",
            re.compile(r"\b(\d{4})-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])\b"),
            0.98,
        ),
        # Month DD, YYYY - e.g., "January 15, 2024"
        (
            "month_day_year",
            re.compile(
                r"\b(January|February|March|April|May|June|July|August|"
                r"September|October|November|December|Jan|Feb|Mar|Apr|Jun|"
                r"Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+"
                r"(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b",
                re.IGNORECASE,
            ),
            0.97,
        ),
        # DD Month YYYY - e.g., "15 January 2024"
        (
            "day_month_year",
            re.compile(
                r"\b(\d{1,2})(?:st|nd|rd|th)?\s+"
                r"(January|February|March|April|May|June|July|August|"
                r"September|October|November|December|Jan|Feb|Mar|Apr|Jun|"
                r"Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+(\d{4})\b",
                re.IGNORECASE,
            ),
            0.96,
        ),
        # Month YYYY - e.g., "January 2024"
        (
            "month_year",
            re.compile(
                r"\b(January|February|March|April|May|June|July|August|"
                r"September|October|November|December|Jan|Feb|Mar|Apr|Jun|"
                r"Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+(\d{4})\b",
                re.IGNORECASE,
            ),
            0.85,
        ),
        # QX YYYY - e.g., "Q1 2024"
        (
            "quarter_year",
            re.compile(r"\b(Q[1-4])\s+(\d{4})\b", re.IGNORECASE),
            0.80,
        ),
        # FY YYYY - e.g., "FY2024"
        (
            "fiscal_year",
            re.compile(r"\bFY\s*(\d{4})\b", re.IGNORECASE),
            0.75,
        ),
    ]

    MONTH_MAP = {
        "january": "01",
        "jan": "01",
        "february": "02",
        "feb": "02",
        "march": "03",
        "mar": "03",
        "april": "04",
        "apr": "04",
        "may": "05",
        "june": "06",
        "jun": "06",
        "july": "07",
        "jul": "07",
        "august": "08",
        "aug": "08",
        "september": "09",
        "sept": "09",
        "sep": "09",
        "october": "10",
        "oct": "10",
        "november": "11",
        "nov": "11",
        "december": "12",
        "dec": "12",
    }

    def get_entity_type(self) -> EntityType:
        return EntityType.DATE

    def _normalize_date(self, pattern_name: str, match: re.Match) -> Optional[str]:
        """
        Normalize a matched date to ISO format YYYY-MM-DD.

        Args:
            pattern_name: Name of the pattern that matched.
            match: The regex match object.

        Returns:
            Normalized date string or None if parsing fails.
        """
        try:
            groups = match.groups()

            if pattern_name == "us_numeric":
                month, day, year = groups[0], groups[1], groups[2]
                return f"{year}-{month}-{day.zfill(2)}"

            elif pattern_name == "eu_numeric":
                day, month, year = groups[0], groups[1], groups[2]
                return f"{year}-{month}-{day.zfill(2)}"

            elif pattern_name == "iso":
                return match.group(0)

            elif pattern_name == "month_day_year":
                month_str = groups[0].lower()
                day = groups[1].zfill(2)
                year = groups[2]
                month = self.MONTH_MAP.get(month_str, "01")
                return f"{year}-{month}-{day}"

            elif pattern_name == "day_month_year":
                day = groups[0].zfill(2)
                month_str = groups[1].lower()
                year = groups[2]
                month = self.MONTH_MAP.get(month_str, "01")
                return f"{year}-{month}-{day}"

            elif pattern_name == "month_year":
                month_str = groups[0].lower()
                year = groups[1]
                month = self.MONTH_MAP.get(month_str, "01")
                return f"{year}-{month}"

            elif pattern_name in ("quarter_year", "fiscal_year"):
                return match.group(0)

        except (IndexError, ValueError) as e:
            logger.debug("Date normalization failed: %s", e)

        return None

    def extract(self, text: str) -> List[ExtractedEntity]:
        """
        Extract all date entities from text.

        Args:
            text: Source text to search.

        Returns:
            List of ExtractedEntity objects for dates.
        """
        entities = []
        seen_spans: set = set()

        for pattern_name, pattern, confidence in self.PATTERNS:
            for match in pattern.finditer(text):
                start, end = match.start(), match.end()
                # Avoid overlapping matches
                if any(start < e and end > s for s, e in seen_spans):
                    continue
                seen_spans.add((start, end))

                normalized = self._normalize_date(pattern_name, match)
                entities.append(
                    ExtractedEntity(
                        text=match.group(0),
                        entity_type=EntityType.DATE,
                        start_pos=start,
                        end_pos=end,
                        confidence=confidence,
                        normalized_value=normalized,
                        metadata={"pattern": pattern_name},
                    )
                )

        return entities


class AmountExtractor(BaseExtractor):
    """
    Extract monetary amounts from text.

    Handles dollar amounts, euro amounts, percentages, and generic
    numeric amounts with currency symbols or currency codes.
    """

    PATTERNS: List[Tuple[str, Pattern[str], float]] = [
        # Dollar amounts with optional decimals: $1,234.56 or $1234
        (
            "dollar_with_cents",
            re.compile(
                r"\$\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})\b",
            ),
            0.98,
        ),
        # Dollar amounts without cents: $1,234 or $1234
        (
            "dollar_no_cents",
            re.compile(
                r"\$\s*\d{1,3}(?:,\d{3})+(?!\.\d)\b",
            ),
            0.95,
        ),
        # Generic $N
        (
            "dollar_simple",
            re.compile(r"\$\s*\d+(?:\.\d{2})?\b"),
            0.90,
        ),
        # Euro amounts
        (
            "euro",
            re.compile(r"\€\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?\b"),
            0.95,
        ),
        # GBP amounts
        (
            "gbp",
            re.compile(r"\£\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?\b"),
            0.95,
        ),
        # Amounts with USD/EUR/GBP suffix
        (
            "currency_code_suffix",
            re.compile(
                r"\b\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s*(USD|EUR|GBP|CAD|AUD|JPY)\b",
                re.IGNORECASE,
            ),
            0.93,
        ),
        # Percentages
        (
            "percentage",
            re.compile(r"\b\d{1,3}(?:\.\d+)?%\s*(?:discount|tax|vat|gst|fee|rate|interest|apr)?\b", re.IGNORECASE),
            0.85,
        ),
        # Written amounts: "one thousand dollars"
        (
            "written_amount",
            re.compile(
                r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|"
                r"eleven|twelve|twenty|thirty|forty|fifty|hundred|thousand|"
                r"million|billion)\s+(?:dollars?|euros?|pounds?)\b",
                re.IGNORECASE,
            ),
            0.80,
        ),
        # Generic numbers in context (total, amount, balance)
        (
            "contextual_amount",
            re.compile(
                r"(?:total(?:\s+amount)?|amount|balance|subtotal|grand\s+total|"
                r"net\s+total|gross)\s*[:\-]?\s*\$?\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?",
                re.IGNORECASE,
            ),
            0.88,
        ),
    ]

    def get_entity_type(self) -> EntityType:
        return EntityType.AMOUNT

    def _normalize_amount(self, raw_text: str) -> Optional[str]:
        """
        Normalize an amount string to a standard numeric format.

        Args:
            raw_text: The raw matched amount text.

        Returns:
            Normalized amount string or None.
        """
        try:
            # Remove currency symbols and whitespace
            cleaned = re.sub(r"[\$\€\£\s,%]", "", raw_text.lower())
            # Remove currency codes
            cleaned = re.sub(r"usd|eur|gbp|cad|aud|jpy", "", cleaned)
            # Try to extract the numeric portion
            match = re.search(r"\d+(?:\.\d+)?", cleaned)
            if match:
                return match.group(0)
        except Exception as e:
            logger.debug("Amount normalization failed for '%s': %s", raw_text, e)
        return None

    def extract(self, text: str) -> List[ExtractedEntity]:
        """
        Extract all monetary amount entities from text.

        Args:
            text: Source text to search.

        Returns:
            List of ExtractedEntity objects for amounts.
        """
        entities = []
        seen_spans: set = set()

        for pattern_name, pattern, confidence in self.PATTERNS:
            for match in pattern.finditer(text):
                start, end = match.start(), match.end()
                if any(start < e and end > s for s, e in seen_spans):
                    continue
                seen_spans.add((start, end))

                normalized = self._normalize_amount(match.group(0))
                entities.append(
                    ExtractedEntity(
                        text=match.group(0),
                        entity_type=EntityType.AMOUNT,
                        start_pos=start,
                        end_pos=end,
                        confidence=confidence,
                        normalized_value=normalized,
                        metadata={"pattern": pattern_name},
                    )
                )

        return entities


class PersonExtractor(BaseExtractor):
    """
    Extract person names from text.

    Uses pattern matching for common name structures, including
    honorifics, middle initials, and suffixes. Also detects names
    in signature blocks and greeting lines.
    """

    PATTERNS: List[Tuple[str, Pattern[str], float]] = [
        # Honorific + First + Last
        (
            "honorific_full",
            re.compile(
                r"\b(Mr\.?|Mrs\.?|Ms\.?|Miss|Dr\.?|Prof\.?|Professor|"
                r"Rev\.?|Hon\.?|Sir|Dame|Lord|Lady)"
                r"\s+([A-Z][a-zA-Z]+)\s+([A-Z][a-zA-Z]+)\b"
            ),
            0.92,
        ),
        # First + Middle Initial + Last
        (
            "first_initial_last",
            re.compile(
                r"\b([A-Z][a-zA-Z]{1,20})\s+([A-Z])\.?\s+"
                r"([A-Z][a-zA-Z]{1,20})\b"
            ),
            0.88,
        ),
        # First + Last (simple)
        (
            "first_last",
            re.compile(
                r"\b([A-Z][a-zA-Z]{1,20})\s+([A-Z][a-zA-Z]{1,25})\b"
            ),
            0.80,
        ),
        # Signature line: "Sincerely, Name"
        (
            "signature",
            re.compile(
                r"(?:Sincerely|Yours sincerely|Yours faithfully|Best regards|"
                r"Kind regards|Regards|Respectfully),?\s*\n?\s*"
                r"([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)",
                re.IGNORECASE,
            ),
            0.90,
        ),
        # From/To field names
        (
            "from_to_field",
            re.compile(
                r"(?:From|To|Attn|Attention|Contact|Prepared by|Author)"
                r"[:\-]\s*([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)",
                re.IGNORECASE,
            ),
            0.85,
        ),
    ]

    # Common words to exclude from name matches
    EXCLUDE_WORDS = {
        "the", "and", "for", "from", "with", "that", "this", "have",
        "has", "had", "not", "are", "were", "been", "their", "they",
        "will", "would", "could", "should", "may", "might", "shall",
        "about", "into", "through", "during", "before", "after",
        "above", "below", "between", "among", "within", "without",
        "invoice", "report", "resume", "letter", "total", "amount",
        "payment", "balance", "company", "inc", "llc", "ltd", "corp",
    }

    def get_entity_type(self) -> EntityType:
        return EntityType.PERSON

    def extract(self, text: str) -> List[ExtractedEntity]:
        """
        Extract person name entities from text.

        Args:
            text: Source text to search.

        Returns:
            List of ExtractedEntity objects for person names.
        """
        entities = []
        seen_spans: set = set()

        for pattern_name, pattern, confidence in self.PATTERNS:
            for match in pattern.finditer(text):
                start, end = match.start(), match.end()
                if any(start < e and end > s for s, e in seen_spans):
                    continue
                seen_spans.add((start, end))

                # For multi-group patterns, extract the name portion
                if pattern_name == "honorific_full":
                    name_text = f"{match.group(2)} {match.group(3)}"
                elif pattern_name == "first_initial_last":
                    name_text = match.group(0)
                elif pattern_name == "first_last":
                    groups = match.groups()
                    # Filter out common non-name words
                    if (
                        groups[0].lower() in self.EXCLUDE_WORDS
                        or groups[1].lower() in self.EXCLUDE_WORDS
                    ):
                        continue
                    name_text = match.group(0)
                elif pattern_name in ("signature", "from_to_field"):
                    name_text = match.group(1)
                else:
                    name_text = match.group(0)

                # Additional validation: exclude all-uppercase and all-lowercase
                if name_text.isupper() or name_text.islower():
                    continue

                entities.append(
                    ExtractedEntity(
                        text=name_text,
                        entity_type=EntityType.PERSON,
                        start_pos=start,
                        end_pos=end,
                        confidence=confidence,
                        normalized_value=name_text.title(),
                        metadata={"pattern": pattern_name},
                    )
                )

        return entities


class OrganizationExtractor(BaseExtractor):
    """
    Extract organization/company names from text.

    Detects business entities by legal suffix patterns and context clues
    like "Inc.", "LLC", "Ltd", "Corp" etc. Also identifies organizations
    from labeled fields.
    """

    LEGAL_SUFFIXES = (
        r"Inc\.?|LLC|L\.L\.C\.|Ltd\.?|Limited|Corp\.?|Corporation|"
        r"PLC|GmbH|AG|BV|S\.A\.|SAS|LLP|LP|Co\.?|Company|"
        r"Holdings|Group|Partners|Associates|Enterprises|Solutions|"
        r"Technologies|Systems|Services|Industries|International|Global"
    )

    PATTERNS: List[Tuple[str, Pattern[str], float]] = [
        # Company with legal suffix
        (
            "legal_suffix",
            re.compile(
                rf"\b([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*){{0,5}}\s+"
                rf"(?:{LEGAL_SUFFIXES}))\b"
            ),
            0.94,
        ),
        # Labeled company fields
        (
            "company_field",
            re.compile(
                r"(?:Company|Organization|Organisation|Firm|Vendor|Client|"
                r"Supplier|Contractor|Employer|Business)"
                r"[:\-]\s*([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]+){0,6})",
                re.IGNORECASE,
            ),
            0.88,
        ),
        # Bill to / Ship to company
        (
            "billing_company",
            re.compile(
                r"(?:Bill To|Ship To|Sold To|Invoice To|Remit To)"
                r"[:\-]?\s*\n?\s*([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]+){0,6})",
                re.IGNORECASE,
            ),
            0.90,
        ),
        # All-caps company abbreviations (3+ chars)
        (
            "acronym_company",
            re.compile(
                r"\b([A-Z]{2,6}(?:\s+[A-Z]{2,6}){0,3}\s+(?:"
                + LEGAL_SUFFIXES.replace(r"\.?", "").replace(r"\.", "")
                + r"))\b"
            ),
            0.82,
        ),
    ]

    def get_entity_type(self) -> EntityType:
        return EntityType.ORGANIZATION

    def extract(self, text: str) -> List[ExtractedEntity]:
        """
        Extract organization entities from text.

        Args:
            text: Source text to search.

        Returns:
            List of ExtractedEntity objects for organizations.
        """
        entities = []
        seen_spans: set = set()

        for pattern_name, pattern, confidence in self.PATTERNS:
            for match in pattern.finditer(text):
                start, end = match.start(), match.end()
                if any(start < e and end > s for s, e in seen_spans):
                    continue
                seen_spans.add((start, end))

                if pattern_name in ("company_field", "billing_company") and match.lastindex:
                    org_text = match.group(1)
                else:
                    org_text = match.group(0)

                # Filter out very short matches
                if len(org_text) < 4:
                    continue

                entities.append(
                    ExtractedEntity(
                        text=org_text,
                        entity_type=EntityType.ORGANIZATION,
                        start_pos=start,
                        end_pos=end,
                        confidence=confidence,
                        normalized_value=org_text.title(),
                        metadata={"pattern": pattern_name},
                    )
                )

        return entities


class ContactExtractor(BaseExtractor):
    """
    Extract contact information: emails and phone numbers.
    """

    EMAIL_PATTERN = re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    )
    PHONE_PATTERNS: List[Tuple[str, Pattern[str], float]] = [
        # US format: (555) 123-4567
        (
            "us_formatted",
            re.compile(r"\(\d{3}\)\s*\d{3}[\s\-]?\d{4}"),
            0.97,
        ),
        # US format with dots: 555.123.4567
        (
            "us_dotted",
            re.compile(r"\b\d{3}\.\d{3}\.\d{4}\b"),
            0.95,
        ),
        # International: +1-555-123-4567
        (
            "international",
            re.compile(r"\+\d{1,3}[\s\-]?\(\d{1,4}\)[\s\-]?\d{3}[\s\-]?\d{4}"),
            0.92,
        ),
        # E.164-like: +15551234567
        (
            "e164",
            re.compile(r"\+\d{10,15}\b"),
            0.90,
        ),
        # Generic with dashes: 555-123-4567
        (
            "generic_dashed",
            re.compile(r"\b\d{3}[\s\-]\d{3}[\s\-]\d{4}\b"),
            0.88,
        ),
    ]

    def get_entity_type(self) -> EntityType:
        return EntityType.EMAIL  # default, overridden per-match

    def extract(self, text: str) -> List[ExtractedEntity]:
        """
        Extract email and phone entities from text.

        Args:
            text: Source text to search.

        Returns:
            List of ExtractedEntity objects for emails and phones.
        """
        entities = []

        # Extract emails
        for match in self.EMAIL_PATTERN.finditer(text):
            entities.append(
                ExtractedEntity(
                    text=match.group(0),
                    entity_type=EntityType.EMAIL,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=0.98,
                    normalized_value=match.group(0).lower(),
                    metadata={"pattern": "email"},
                )
            )

        # Extract phones
        seen_spans: set = set()
        for pattern_name, pattern, confidence in self.PHONE_PATTERNS:
            for match in pattern.finditer(text):
                start, end = match.start(), match.end()
                if any(start < e and end > s for s, e in seen_spans):
                    continue
                seen_spans.add((start, end))

                normalized = re.sub(r"[^\d+]", "", match.group(0))
                entities.append(
                    ExtractedEntity(
                        text=match.group(0),
                        entity_type=EntityType.PHONE,
                        start_pos=start,
                        end_pos=end,
                        confidence=confidence,
                        normalized_value=normalized,
                        metadata={"pattern": pattern_name},
                    )
                )

        return entities


class InvoiceNumberExtractor(BaseExtractor):
    """Extract invoice numbers, PO numbers, and similar identifiers."""

    PATTERNS: List[Tuple[str, Pattern[str], float]] = [
        # Invoice Number patterns
        (
            "inv_num",
            re.compile(
                r"(?:Invoice\s*(?:No|Number|#)[:\s]*)"
                r"([A-Z]{0,4}\d{2,10}[A-Z]{0,4})\b",
                re.IGNORECASE,
            ),
            0.97,
        ),
        # Generic INV-XXXX
        (
            "inv_prefix",
            re.compile(r"\b(INV[\-\_]\d{3,10})\b", re.IGNORECASE),
            0.95,
        ),
        # PO Number patterns
        (
            "po_num",
            re.compile(
                r"(?:PO\s*(?:No|Number|#)[:\s]*|Purchase\s+Order[:\s]*)"
                r"([A-Z]{0,4}\d{2,10}[A-Z]{0,4})\b",
                re.IGNORECASE,
            ),
            0.96,
        ),
        # Generic PO-XXXX
        (
            "po_prefix",
            re.compile(r"\b(PO[\-\_]\d{3,10})\b", re.IGNORECASE),
            0.94,
        ),
    ]

    def get_entity_type(self) -> EntityType:
        return EntityType.INVOICE_NUMBER

    def extract(self, text: str) -> List[ExtractedEntity]:
        """
        Extract invoice and PO number entities.

        Args:
            text: Source text to search.

        Returns:
            List of ExtractedEntity objects.
        """
        entities = []

        for pattern_name, pattern, confidence in self.PATTERNS:
            for match in pattern.finditer(text):
                entity_type = (
                    EntityType.INVOICE_NUMBER
                    if "inv" in pattern_name
                    else EntityType.PO_NUMBER
                )
                value = match.group(1) if match.lastindex else match.group(0)

                entities.append(
                    ExtractedEntity(
                        text=value,
                        entity_type=entity_type,
                        start_pos=match.start(),
                        end_pos=match.end(),
                        confidence=confidence,
                        normalized_value=value.upper(),
                        metadata={"pattern": pattern_name},
                    )
                )

        return entities


class EntityExtractionPipeline:
    """
    Unified entity extraction pipeline combining all extractors.

    Provides a single interface to run multiple entity extractors
    and merge their results with deduplication.
    """

    def __init__(self, extractors: Optional[List[BaseExtractor]] = None) -> None:
        """
        Initialize the pipeline.

        Args:
            extractors: Optional list of extractors. Uses all default
                extractors if None.
        """
        self.extractors = extractors or self._default_extractors()

    def _default_extractors(self) -> List[BaseExtractor]:
        """Return the default set of entity extractors."""
        return [
            DateExtractor(),
            AmountExtractor(),
            PersonExtractor(),
            OrganizationExtractor(),
            ContactExtractor(),
            InvoiceNumberExtractor(),
        ]

    def extract_all(self, text: str) -> List[ExtractedEntity]:
        """
        Extract all entity types from the given text.

        Args:
            text: Source text to analyze.

        Returns:
            List of all extracted entities, sorted by position.
        """
        if not text:
            return []

        all_entities: List[ExtractedEntity] = []

        for extractor in self.extractors:
            try:
                entities = extractor.extract(text)
                all_entities.extend(entities)
            except Exception as e:
                logger.error(
                    "Extractor %s failed: %s",
                    extractor.__class__.__name__,
                    e,
                )

        # Sort by position and then by confidence (higher first for same position)
        all_entities.sort(key=lambda e: (e.start_pos, -e.confidence))

        # Deduplicate: remove overlapping entities with lower confidence
        deduplicated = self._deduplicate(all_entities)

        return deduplicated

    def _deduplicate(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """
        Remove overlapping entities, keeping higher-confidence ones.

        Args:
            entities: List of potentially overlapping entities.

        Returns:
            Deduplicated list.
        """
        if not entities:
            return []

        result = []
        for entity in entities:
            overlap = False
            for existing in result:
                # Check for overlap
                if (
                    entity.start_pos < existing.end_pos
                    and entity.end_pos > existing.start_pos
                ):
                    overlap = True
                    break
            if not overlap:
                result.append(entity)

        return result

    def extract_by_type(
        self, text: str, entity_type: EntityType
    ) -> List[ExtractedEntity]:
        """
        Extract only entities of a specific type.

        Args:
            text: Source text.
            entity_type: Type of entities to extract.

        Returns:
            Filtered list of entities.
        """
        all_entities = self.extract_all(text)
        return [e for e in all_entities if e.entity_type == entity_type]

    def get_summary(self, text: str) -> Dict[str, Any]:
        """
        Get a summary of all extracted entities.

        Args:
            text: Source text.

        Returns:
            Dictionary with entity counts and grouped entities.
        """
        entities = self.extract_all(text)

        by_type: Dict[str, List[Dict[str, Any]]] = {}
        type_counts: Dict[str, int] = {}

        for entity in entities:
            type_key = entity.entity_type.value
            if type_key not in by_type:
                by_type[type_key] = []
                type_counts[type_key] = 0
            by_type[type_key].append(entity.to_dict())
            type_counts[type_key] += 1

        return {
            "total_entities": len(entities),
            "by_type": type_counts,
            "entities": by_type,
        }
