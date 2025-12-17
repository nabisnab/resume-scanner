"""
NLP Processing module for resume analysis.

This module provides comprehensive NLP functionality for resume processing including:
- Text preprocessing and normalization
- Skills extraction
- Experience detection
- Action verb extraction
- Resume section detection
- Text quality analysis
- Keyword extraction
"""

import re
from typing import List, Dict, Set, Tuple, Optional
from collections import Counter
import string


class TextPreprocessor:
    """Handles text preprocessing and normalization."""

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep alphanumeric and basic punctuation
        text = re.sub(r'[^\w\s\.\,\-\+\(\)\/]', '', text)
        return text.strip()

    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Normalize text to lowercase for processing.
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text
        """
        return text.lower()

    @staticmethod
    def remove_stopwords(text: str, stopwords: Optional[Set[str]] = None) -> str:
        """
        Remove common stopwords from text.
        
        Args:
            text: Text to process
            stopwords: Set of stopwords to remove
            
        Returns:
            Text with stopwords removed
        """
        if stopwords is None:
            stopwords = TextPreprocessor.get_default_stopwords()
        
        words = text.split()
        filtered_words = [w for w in words if w.lower() not in stopwords]
        return ' '.join(filtered_words)

    @staticmethod
    def get_default_stopwords() -> Set[str]:
        """Get default set of English stopwords."""
        return {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'or', 'that',
            'the', 'to', 'was', 'will', 'with', 'this', 'but', 'they', 'have',
            'had', 'would', 'could', 'should', 'about', 'which', 'who', 'when'
        }

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        # Split by whitespace and punctuation
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens


class SkillsExtractor:
    """Extracts skills from resume text."""

    # Common technical and professional skills
    COMMON_SKILLS = {
        # Programming Languages
        'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go', 'rust',
        'php', 'ruby', 'swift', 'kotlin', 'scala', 'r', 'matlab', 'sql',
        
        # Web Technologies
        'html', 'css', 'react', 'angular', 'vue', 'nodejs', 'express', 'django',
        'flask', 'spring', 'fastapi', 'asp.net', 'webpack', 'babel',
        
        # Databases
        'mongodb', 'postgresql', 'mysql', 'sqlite', 'redis', 'cassandra',
        'elasticsearch', 'dynamodb', 'firestore', 'oracle', 'sqlserver',
        
        # Cloud & DevOps
        'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'gitlab',
        'github', 'ci/cd', 'terraform', 'ansible', 'cloudformation',
        
        # Data & ML
        'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch', 'keras',
        'hadoop', 'spark', 'machine learning', 'deep learning', 'nlp',
        'data analysis', 'data science',
        
        # Soft Skills
        'communication', 'leadership', 'teamwork', 'project management',
        'problem solving', 'critical thinking', 'time management',
        'collaboration', 'adaptability', 'attention to detail'
    }

    @classmethod
    def extract_skills(cls, text: str) -> List[str]:
        """
        Extract skills from text.
        
        Args:
            text: Resume text
            
        Returns:
            List of detected skills
        """
        normalized_text = text.lower()
        detected_skills = []
        
        for skill in cls.COMMON_SKILLS:
            # Create pattern to match whole words
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, normalized_text):
                detected_skills.append(skill)
        
        return list(set(detected_skills))  # Remove duplicates

    @classmethod
    def extract_custom_skills(cls, text: str, custom_skill_list: List[str]) -> List[str]:
        """
        Extract custom skills from text.
        
        Args:
            text: Resume text
            custom_skill_list: List of skills to search for
            
        Returns:
            List of detected custom skills
        """
        normalized_text = text.lower()
        detected_skills = []
        
        for skill in custom_skill_list:
            pattern = r'\b' + re.escape(skill.lower()) + r'\b'
            if re.search(pattern, normalized_text):
                detected_skills.append(skill)
        
        return list(set(detected_skills))


class ExperienceDetector:
    """Detects work experience and duration from resume text."""

    # Date patterns for experience
    DATE_PATTERNS = {
        'month_year': r'(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{4})',
        'year_only': r'(\d{4})',
        'duration': r'(\d+)\s*(?:years?|yrs?)',
        'duration_short': r'(\d+)\s*(?:months?|mos?)',
    }

    @staticmethod
    def extract_dates(text: str) -> List[str]:
        """
        Extract dates from text.
        
        Args:
            text: Resume text
            
        Returns:
            List of found dates
        """
        dates = []
        for pattern in ExperienceDetector.DATE_PATTERNS.values():
            matches = re.findall(pattern, text)
            dates.extend(matches)
        return dates

    @staticmethod
    def extract_job_titles(text: str) -> List[str]:
        """
        Extract potential job titles from text.
        
        Args:
            text: Resume text
            
        Returns:
            List of potential job titles
        """
        common_titles = [
            'software engineer', 'developer', 'data scientist', 'analyst',
            'manager', 'director', 'coordinator', 'specialist', 'consultant',
            'architect', 'engineer', 'administrator', 'support', 'lead',
            'principal', 'senior', 'junior', 'intern', 'associate'
        ]
        
        found_titles = []
        normalized_text = text.lower()
        
        for title in common_titles:
            if title in normalized_text:
                found_titles.append(title)
        
        return list(set(found_titles))

    @staticmethod
    def estimate_experience_years(text: str) -> Optional[int]:
        """
        Estimate total years of experience.
        
        Args:
            text: Resume text
            
        Returns:
            Estimated years of experience or None
        """
        pattern = r'(\d+)\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)'
        matches = re.findall(pattern, text.lower())
        
        if matches:
            return max(int(m) for m in matches)
        return None


class ActionVerbExtractor:
    """Extracts action verbs from resume text."""

    STRONG_ACTION_VERBS = {
        # Leadership
        'led', 'managed', 'directed', 'oversaw', 'coordinated', 'supervised',
        'mentored', 'coached', 'motivated', 'inspired',
        
        # Achievement
        'accomplished', 'achieved', 'delivered', 'completed', 'executed',
        'implemented', 'established', 'founded', 'launched', 'initiated',
        
        # Improvement
        'improved', 'enhanced', 'optimized', 'streamlined', 'accelerated',
        'increased', 'boosted', 'amplified', 'elevated', 'strengthened',
        
        # Analysis
        'analyzed', 'evaluated', 'assessed', 'examined', 'investigated',
        'researched', 'identified', 'determined', 'diagnosed',
        
        # Technical
        'developed', 'designed', 'built', 'created', 'engineered',
        'architected', 'programmed', 'deployed', 'integrated',
        
        # Communication
        'presented', 'reported', 'communicated', 'collaborated', 'partnered',
        'negotiated', 'convinced', 'persuaded',
        
        # Financial
        'reduced', 'saved', 'generated', 'earned', 'budgeted', 'forecasted',
        'increased revenue', 'decreased costs'
    }

    @classmethod
    def extract_action_verbs(cls, text: str) -> List[str]:
        """
        Extract action verbs from text.
        
        Args:
            text: Resume text
            
        Returns:
            List of detected action verbs
        """
        normalized_text = text.lower()
        found_verbs = []
        
        for verb in cls.STRONG_ACTION_VERBS:
            pattern = r'\b' + re.escape(verb) + r'\b'
            if re.search(pattern, normalized_text):
                found_verbs.append(verb)
        
        return list(set(found_verbs))

    @classmethod
    def count_action_verbs(cls, text: str) -> Dict[str, int]:
        """
        Count occurrences of action verbs.
        
        Args:
            text: Resume text
            
        Returns:
            Dictionary with verb counts
        """
        normalized_text = text.lower()
        verb_counts = {}
        
        for verb in cls.STRONG_ACTION_VERBS:
            pattern = r'\b' + re.escape(verb) + r'\b'
            matches = re.findall(pattern, normalized_text)
            if matches:
                verb_counts[verb] = len(matches)
        
        return verb_counts


class ResumeSectionDetector:
    """Detects and extracts resume sections."""

    SECTION_HEADERS = {
        'summary': r'(summary|objective|professional summary|about)',
        'experience': r'(work experience|professional experience|employment history|experience)',
        'education': r'(education|academic|qualifications|degrees)',
        'skills': r'(skills|technical skills|competencies|expertise)',
        'projects': r'(projects|portfolio|accomplishments)',
        'certifications': r'(certifications|licenses|credentials|certifications?)',
        'awards': r'(awards|honors|recognition)',
        'languages': r'(languages|language proficiency)',
        'volunteering': r'(volunteer|volunteering|community service)',
        'publications': r'(publications|papers|articles)'
    }

    @classmethod
    def detect_sections(cls, text: str) -> Dict[str, bool]:
        """
        Detect which sections are present in resume.
        
        Args:
            text: Resume text
            
        Returns:
            Dictionary with section presence
        """
        normalized_text = text.lower()
        sections = {}
        
        for section_name, pattern in cls.SECTION_HEADERS.items():
            sections[section_name] = bool(re.search(pattern, normalized_text))
        
        return sections

    @classmethod
    def extract_section_content(cls, text: str, section_name: str) -> Optional[str]:
        """
        Extract content of a specific section.
        
        Args:
            text: Resume text
            section_name: Name of section to extract
            
        Returns:
            Section content or None if not found
        """
        if section_name not in cls.SECTION_HEADERS:
            return None
        
        pattern = cls.SECTION_HEADERS[section_name]
        # Find the section header
        header_match = re.search(pattern, text, re.IGNORECASE)
        if not header_match:
            return None
        
        # Extract content until next section
        start_pos = header_match.end()
        next_section = None
        
        for other_section_name, other_pattern in cls.SECTION_HEADERS.items():
            if other_section_name != section_name:
                match = re.search(other_pattern, text[start_pos:], re.IGNORECASE)
                if match:
                    pos = start_pos + match.start()
                    if next_section is None or pos < next_section:
                        next_section = pos
        
        if next_section:
            return text[start_pos:next_section].strip()
        return text[start_pos:].strip()


class TextQualityAnalyzer:
    """Analyzes text quality and readability metrics."""

    @staticmethod
    def analyze_quality(text: str) -> Dict[str, any]:
        """
        Analyze overall text quality.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with quality metrics
        """
        if not text:
            return {}
        
        words = text.split()
        sentences = re.split(r'[.!?]', text)
        
        return {
            'word_count': len(words),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'avg_word_length': sum(len(w) for w in words) / len(words) if words else 0,
            'avg_sentence_length': len(words) / len([s for s in sentences if s.strip()]) if sentences else 0,
            'unique_words': len(set(words)),
            'lexical_diversity': len(set(words)) / len(words) if words else 0
        }

    @staticmethod
    def check_spelling_complexity(text: str) -> Dict[str, any]:
        """
        Analyze spelling and complexity.
        
        Args:
            text: Text to analyze
            
        Returns:
            Complexity metrics
        """
        words = text.split()
        long_words = [w for w in words if len(w) > 6]
        
        return {
            'long_word_count': len(long_words),
            'long_word_ratio': len(long_words) / len(words) if words else 0,
            'has_numbers': bool(re.search(r'\d', text)),
            'has_special_chars': bool(re.search(r'[^\w\s]', text))
        }

    @staticmethod
    def get_readability_score(text: str) -> float:
        """
        Calculate a simple readability score (0-100).
        
        Args:
            text: Text to analyze
            
        Returns:
            Readability score
        """
        words = text.split()
        if len(words) == 0:
            return 0
        
        sentences = len(re.split(r'[.!?]', text))
        word_count = len(words)
        
        # Flesch Reading Ease approximation
        if sentences == 0:
            sentences = 1
        
        score = 206.835 - 1.015 * (word_count / sentences) - 84.6 * (len([w for w in words if len(w) > 6]) / word_count)
        return max(0, min(100, score))  # Clamp between 0-100


class KeywordExtractor:
    """Extracts and analyzes keywords from text."""

    @staticmethod
    def extract_keywords(text: str, num_keywords: int = 10) -> List[Tuple[str, int]]:
        """
        Extract most frequent keywords from text.
        
        Args:
            text: Text to analyze
            num_keywords: Number of keywords to return
            
        Returns:
            List of (keyword, frequency) tuples
        """
        # Tokenize and clean
        tokens = re.findall(r'\b\w+\b', text.lower())
        
        # Remove stopwords
        stopwords = TextPreprocessor.get_default_stopwords()
        filtered_tokens = [t for t in tokens if t not in stopwords and len(t) > 2]
        
        # Count frequencies
        counter = Counter(filtered_tokens)
        return counter.most_common(num_keywords)

    @staticmethod
    def extract_ngrams(text: str, n: int = 2, num_ngrams: int = 10) -> List[Tuple[str, int]]:
        """
        Extract n-grams from text.
        
        Args:
            text: Text to analyze
            n: Size of n-grams
            num_ngrams: Number of n-grams to return
            
        Returns:
            List of (n-gram, frequency) tuples
        """
        tokens = re.findall(r'\b\w+\b', text.lower())
        
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = ' '.join(tokens[i:i+n])
            ngrams.append(ngram)
        
        counter = Counter(ngrams)
        return counter.most_common(num_ngrams)

    @staticmethod
    def calculate_keyword_density(text: str, keyword: str) -> float:
        """
        Calculate keyword density as percentage.
        
        Args:
            text: Text to analyze
            keyword: Keyword to find
            
        Returns:
            Keyword density as percentage
        """
        tokens = re.findall(r'\b\w+\b', text.lower())
        if not tokens:
            return 0
        
        keyword_lower = keyword.lower()
        count = sum(1 for t in tokens if t == keyword_lower)
        return (count / len(tokens)) * 100


class NLPProcessor:
    """Main NLP processor combining all functionality."""

    def __init__(self):
        """Initialize the NLP processor."""
        self.text_preprocessor = TextPreprocessor()
        self.skills_extractor = SkillsExtractor()
        self.experience_detector = ExperienceDetector()
        self.action_verb_extractor = ActionVerbExtractor()
        self.section_detector = ResumeSectionDetector()
        self.quality_analyzer = TextQualityAnalyzer()
        self.keyword_extractor = KeywordExtractor()

    def process_resume(self, text: str) -> Dict:
        """
        Perform comprehensive NLP analysis on resume text.
        
        Args:
            text: Raw resume text
            
        Returns:
            Dictionary containing all analysis results
        """
        # Clean text
        cleaned_text = self.text_preprocessor.clean_text(text)
        normalized_text = self.text_preprocessor.normalize_text(cleaned_text)
        
        # Extract various elements
        results = {
            'cleaned_text': cleaned_text,
            'skills': self.skills_extractor.extract_skills(normalized_text),
            'job_titles': self.experience_detector.extract_job_titles(normalized_text),
            'dates': self.experience_detector.extract_dates(text),
            'experience_years': self.experience_detector.estimate_experience_years(text),
            'action_verbs': self.action_verb_extractor.extract_action_verbs(normalized_text),
            'action_verb_counts': self.action_verb_extractor.count_action_verbs(normalized_text),
            'sections': self.section_detector.detect_sections(normalized_text),
            'quality_metrics': self.quality_analyzer.analyze_quality(cleaned_text),
            'complexity_metrics': self.quality_analyzer.check_spelling_complexity(cleaned_text),
            'readability_score': self.quality_analyzer.get_readability_score(cleaned_text),
            'keywords': self.keyword_extractor.extract_keywords(normalized_text),
            'bigrams': self.keyword_extractor.extract_ngrams(normalized_text, n=2),
            'trigrams': self.keyword_extractor.extract_ngrams(normalized_text, n=3),
        }
        
        return results

    def extract_section_analysis(self, text: str, section_name: str) -> Dict:
        """
        Analyze a specific section of the resume.
        
        Args:
            text: Resume text
            section_name: Name of section to analyze
            
        Returns:
            Analysis results for the section
        """
        section_content = self.section_detector.extract_section_content(text, section_name)
        if not section_content:
            return {'section': section_name, 'found': False}
        
        cleaned_text = self.text_preprocessor.clean_text(section_content)
        normalized_text = self.text_preprocessor.normalize_text(cleaned_text)
        
        return {
            'section': section_name,
            'found': True,
            'content': section_content[:200] + '...' if len(section_content) > 200 else section_content,
            'skills': self.skills_extractor.extract_skills(normalized_text),
            'action_verbs': self.action_verb_extractor.extract_action_verbs(normalized_text),
            'quality_metrics': self.quality_analyzer.analyze_quality(cleaned_text),
            'keywords': self.keyword_extractor.extract_keywords(normalized_text, num_keywords=5)
        }
