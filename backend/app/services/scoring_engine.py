"""
Scoring Engine for Resume Analysis

This module provides comprehensive scoring and evaluation of resumes based on:
- Skills match
- Experience assessment
- Education evaluation
- Keyword relevance
- Content quality
- Section completeness
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import re
from collections import Counter


class SkillLevel(Enum):
    """Enumeration for skill proficiency levels."""
    EXPERT = 1.0
    ADVANCED = 0.8
    INTERMEDIATE = 0.6
    BEGINNER = 0.4
    UNKNOWN = 0.2


@dataclass
class SkillMatch:
    """Data class representing a skill match."""
    skill: str
    found: bool
    proficiency_level: SkillLevel
    context: Optional[str] = None


@dataclass
class ScoringResult:
    """Data class for complete scoring results."""
    overall_score: float
    skills_score: float
    experience_score: float
    education_score: float
    keywords_score: float
    content_quality_score: float
    section_completeness_score: float
    skill_matches: List[SkillMatch]
    missing_skills: List[str]
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]


class ScoringEngine:
    """
    Main scoring engine for resume evaluation.
    
    Evaluates resumes across multiple dimensions and provides detailed
    scoring breakdown and recommendations.
    """
    
    # Weights for overall score calculation
    SKILL_WEIGHT = 0.30
    EXPERIENCE_WEIGHT = 0.25
    EDUCATION_WEIGHT = 0.15
    KEYWORDS_WEIGHT = 0.15
    CONTENT_QUALITY_WEIGHT = 0.10
    SECTION_COMPLETENESS_WEIGHT = 0.05
    
    # Required sections for a complete resume
    REQUIRED_SECTIONS = {
        'contact_info',
        'professional_summary',
        'experience',
        'education',
        'skills'
    }
    
    # Optional but recommended sections
    OPTIONAL_SECTIONS = {
        'certifications',
        'projects',
        'languages',
        'achievements',
        'references'
    }
    
    # Common relevant keywords by category
    INDUSTRY_KEYWORDS = {
        'technical': [
            'python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'php', 'go',
            'rust', 'kotlin', 'swift', 'typescript', 'react', 'angular', 'vue',
            'nodejs', 'django', 'flask', 'spring', 'fastapi', 'express',
            'sql', 'nosql', 'mongodb', 'postgresql', 'mysql', 'redis',
            'docker', 'kubernetes', 'aws', 'azure', 'gcp', 'jenkins',
            'git', 'ci/cd', 'agile', 'scrum', 'jira', 'linux', 'unix'
        ],
        'soft_skills': [
            'communication', 'leadership', 'teamwork', 'collaboration',
            'problem-solving', 'critical thinking', 'time management',
            'project management', 'attention to detail', 'adaptability',
            'creativity', 'analytical', 'strategic', 'decision-making'
        ],
        'achievement_indicators': [
            'led', 'managed', 'developed', 'improved', 'increased',
            'reduced', 'optimized', 'designed', 'implemented', 'delivered',
            'achieved', 'successfully', 'award', 'recognition', 'promoted'
        ]
    }
    
    def __init__(self, required_skills: Optional[List[str]] = None):
        """
        Initialize the scoring engine.
        
        Args:
            required_skills: List of skills to evaluate against resume.
        """
        self.required_skills = required_skills or []
    
    def score_resume(self, resume_data: Dict) -> ScoringResult:
        """
        Perform comprehensive scoring of a resume.
        
        Args:
            resume_data: Dictionary containing parsed resume information with keys:
                - text: Full resume text
                - contact_info: Dict with name, email, phone, location
                - professional_summary: String summary
                - experience: List of Dict with job details
                - education: List of Dict with education details
                - skills: List of skills mentioned
                - certifications: List of certifications
                - projects: List of projects
                - languages: List of languages
                - sections_found: List of section names present
        
        Returns:
            ScoringResult with detailed scoring breakdown.
        """
        # Extract and normalize resume components
        resume_text = resume_data.get('text', '').lower()
        sections_found = set(resume_data.get('sections_found', []))
        skills_mentioned = set(resume_data.get('skills', []))
        experience = resume_data.get('experience', [])
        education = resume_data.get('education', [])
        
        # Calculate individual scores
        skills_score, skill_matches, missing_skills = self._score_skills(
            resume_text, skills_mentioned
        )
        experience_score = self._score_experience(experience, resume_text)
        education_score = self._score_education(education)
        keywords_score = self._score_keywords(resume_text)
        content_quality_score = self._score_content_quality(resume_data)
        section_completeness_score = self._score_section_completeness(
            sections_found
        )
        
        # Calculate weighted overall score
        overall_score = (
            skills_score * self.SKILL_WEIGHT +
            experience_score * self.EXPERIENCE_WEIGHT +
            education_score * self.EDUCATION_WEIGHT +
            keywords_score * self.KEYWORDS_WEIGHT +
            content_quality_score * self.CONTENT_QUALITY_WEIGHT +
            section_completeness_score * self.SECTION_COMPLETENESS_WEIGHT
        )
        
        # Identify strengths and weaknesses
        strengths = self._identify_strengths(
            skills_score, experience_score, education_score,
            keywords_score, content_quality_score
        )
        weaknesses = self._identify_weaknesses(
            skills_score, experience_score, education_score,
            keywords_score, content_quality_score, section_completeness_score
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            weaknesses, missing_skills, sections_found, resume_text
        )
        
        return ScoringResult(
            overall_score=round(overall_score * 100, 2),
            skills_score=round(skills_score * 100, 2),
            experience_score=round(experience_score * 100, 2),
            education_score=round(education_score * 100, 2),
            keywords_score=round(keywords_score * 100, 2),
            content_quality_score=round(content_quality_score * 100, 2),
            section_completeness_score=round(section_completeness_score * 100, 2),
            skill_matches=skill_matches,
            missing_skills=missing_skills,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations
        )
    
    def _score_skills(
        self, resume_text: str, skills_mentioned: set
    ) -> Tuple[float, List[SkillMatch], List[str]]:
        """
        Score skills section and match against required skills.
        
        Args:
            resume_text: Lowercase resume text
            skills_mentioned: Set of skills extracted from resume
        
        Returns:
            Tuple of (skills_score, skill_matches, missing_skills)
        """
        skill_matches = []
        match_count = 0
        missing_skills = []
        
        if not self.required_skills:
            # If no required skills specified, score based on skills mentioned
            skill_diversity_score = min(len(skills_mentioned) / 10, 1.0)
            
            # Bonus for technical skills
            technical_skills = [
                s for s in skills_mentioned
                if any(keyword in s.lower() for keyword in
                       self.INDUSTRY_KEYWORDS['technical'])
            ]
            technical_bonus = min(len(technical_skills) / 5, 0.2)
            
            skills_score = skill_diversity_score + technical_bonus
        else:
            # Match against required skills
            for skill in self.required_skills:
                skill_lower = skill.lower()
                found = skill_lower in skills_mentioned or \
                        any(skill_lower in s.lower() for s in skills_mentioned)
                
                # Determine proficiency level from context
                proficiency = self._assess_skill_proficiency(
                    skill_lower, resume_text
                )
                
                skill_matches.append(
                    SkillMatch(
                        skill=skill,
                        found=found,
                        proficiency_level=proficiency
                    )
                )
                
                if found:
                    match_count += 1
            
            # Calculate skills score
            base_score = match_count / len(self.required_skills)
            proficiency_bonus = sum(
                m.proficiency_level.value / len(self.required_skills)
                for m in skill_matches if m.found
            )
            skills_score = min(base_score + proficiency_bonus * 0.2, 1.0)
            
            # Identify missing skills
            missing_skills = [
                m.skill for m in skill_matches if not m.found
            ]
        
        return min(skills_score, 1.0), skill_matches, missing_skills
    
    def _assess_skill_proficiency(
        self, skill: str, resume_text: str
    ) -> SkillLevel:
        """
        Assess proficiency level of a skill from context.
        
        Args:
            skill: The skill to assess
            resume_text: Resume text to analyze
        
        Returns:
            SkillLevel enum value
        """
        # Look for proficiency indicators in context around skill
        context_patterns = {
            SkillLevel.EXPERT: [
                f'expert {skill}', f'mastery of {skill}',
                f'advanced {skill}', f'led.*{skill}', f'architected.*{skill}'
            ],
            SkillLevel.ADVANCED: [
                f'advanced {skill}', f'proficient {skill}',
                f'extensive {skill}', f'deep {skill}'
            ],
            SkillLevel.INTERMEDIATE: [
                f'intermediate {skill}', f'familiar {skill}',
                f'experience with {skill}', f'worked with {skill}'
            ],
            SkillLevel.BEGINNER: [
                f'basic {skill}', f'learning {skill}',
                f'introductory {skill}'
            ]
        }
        
        for level, patterns in context_patterns.items():
            for pattern in patterns:
                if re.search(pattern, resume_text):
                    return level
        
        return SkillLevel.UNKNOWN
    
    def _score_experience(
        self, experience: List[Dict], resume_text: str
    ) -> float:
        """
        Score professional experience section.
        
        Args:
            experience: List of experience entries
            resume_text: Resume text for additional analysis
        
        Returns:
            Experience score (0.0 to 1.0)
        """
        if not experience:
            return 0.0
        
        score = 0.0
        
        # Score based on number of positions (ideal: 3-5)
        position_count = len(experience)
        if position_count >= 3:
            position_score = 1.0 if position_count <= 5 else 0.9
        elif position_count == 2:
            position_score = 0.7
        else:
            position_score = 0.5
        
        # Score based on description quality
        description_quality = 0.0
        achievement_count = 0
        
        for exp in experience:
            description = exp.get('description', '').lower()
            if not description:
                continue
            
            # Check for achievement indicators
            achievements = sum(
                1 for indicator in self.INDUSTRY_KEYWORDS['achievement_indicators']
                if indicator in description
            )
            achievement_count += achievements
            
            # Check for quantifiable metrics
            has_metrics = bool(re.search(r'\d+%|\$\d+|increased|decreased', description))
            
            # Calculate description score
            words = len(description.split())
            if words >= 30:  # Substantial description
                exp_score = 1.0 if has_metrics else 0.8
            elif words >= 15:
                exp_score = 0.7 if has_metrics else 0.6
            else:
                exp_score = 0.4
            
            description_quality += exp_score
        
        description_score = (
            (description_quality / position_count) if position_count > 0 else 0.0
        )
        
        # Combine scores
        score = (position_score * 0.4 + description_score * 0.6)
        
        return min(score, 1.0)
    
    def _score_education(self, education: List[Dict]) -> float:
        """
        Score education section.
        
        Args:
            education: List of education entries
        
        Returns:
            Education score (0.0 to 1.0)
        """
        if not education:
            return 0.3  # Partial credit for no education listed
        
        score = 0.0
        max_score = 0.0
        
        for edu in education:
            degree = edu.get('degree', '').lower()
            institution = edu.get('institution', '').lower()
            
            # Score based on degree level
            degree_score = 0.0
            if any(d in degree for d in ['phd', 'doctorate']):
                degree_score = 1.0
            elif any(d in degree for d in ['master', 'mba', 'ms', 'ma']):
                degree_score = 0.95
            elif any(d in degree for d in ['bachelor', 'bs', 'ba', 'bsc']):
                degree_score = 0.85
            elif any(d in degree for d in ['associate', 'diploma']):
                degree_score = 0.65
            elif any(d in degree for d in ['certificate', 'bootcamp']):
                degree_score = 0.55
            else:
                degree_score = 0.4
            
            # Bonus for recognized institution
            institution_bonus = 0.0
            if institution and len(institution) > 0:
                institution_bonus = 0.05
            
            score += degree_score + institution_bonus
            max_score += 1.0
        
        return min(score / max_score if max_score > 0 else 0, 1.0)
    
    def _score_keywords(self, resume_text: str) -> float:
        """
        Score based on industry keyword presence.
        
        Args:
            resume_text: Lowercase resume text
        
        Returns:
            Keywords score (0.0 to 1.0)
        """
        score = 0.0
        
        # Count technical skills
        technical_matches = sum(
            1 for keyword in self.INDUSTRY_KEYWORDS['technical']
            if keyword in resume_text
        )
        technical_score = min(technical_matches / 5, 0.4)
        
        # Count soft skills
        soft_skill_matches = sum(
            1 for keyword in self.INDUSTRY_KEYWORDS['soft_skills']
            if keyword in resume_text
        )
        soft_score = min(soft_skill_matches / 4, 0.3)
        
        # Count achievement indicators
        achievement_matches = sum(
            1 for keyword in self.INDUSTRY_KEYWORDS['achievement_indicators']
            if keyword in resume_text
        )
        achievement_score = min(achievement_matches / 5, 0.3)
        
        score = technical_score + soft_score + achievement_score
        
        return min(score, 1.0)
    
    def _score_content_quality(self, resume_data: Dict) -> float:
        """
        Score overall content quality.
        
        Args:
            resume_data: Resume data dictionary
        
        Returns:
            Content quality score (0.0 to 1.0)
        """
        score = 0.0
        components = 0
        
        # Check resume length (ideal: 400-1000 words)
        resume_text = resume_data.get('text', '')
        word_count = len(resume_text.split())
        if 400 <= word_count <= 1000:
            length_score = 1.0
        elif 200 <= word_count < 400 or 1000 < word_count <= 1500:
            length_score = 0.8
        elif word_count < 200 or word_count > 1500:
            length_score = 0.5
        else:
            length_score = 0.3
        score += length_score
        components += 1
        
        # Check for professional summary
        summary = resume_data.get('professional_summary', '')
        summary_score = 0.8 if summary and len(summary) > 30 else 0.3
        score += summary_score
        components += 1
        
        # Check for grammatical quality (simple heuristic)
        contact_info = resume_data.get('contact_info', {})
        has_email = bool(contact_info.get('email'))
        has_phone = bool(contact_info.get('phone'))
        contact_score = 1.0 if (has_email and has_phone) else 0.6
        score += contact_score
        components += 1
        
        # Check for formatting consistency
        experience = resume_data.get('experience', [])
        has_consistent_format = all(
            exp.get('title') and exp.get('description')
            for exp in experience
        )
        format_score = 0.9 if has_consistent_format else 0.6
        score += format_score
        components += 1
        
        return min(score / components if components > 0 else 0, 1.0)
    
    def _score_section_completeness(self, sections_found: set) -> float:
        """
        Score completeness of resume sections.
        
        Args:
            sections_found: Set of section names present in resume
        
        Returns:
            Section completeness score (0.0 to 1.0)
        """
        required_found = len(sections_found & self.REQUIRED_SECTIONS)
        required_count = len(self.REQUIRED_SECTIONS)
        
        optional_found = len(sections_found & self.OPTIONAL_SECTIONS)
        optional_count = len(self.OPTIONAL_SECTIONS)
        
        # Required sections are weighted more heavily
        required_score = required_found / required_count if required_count > 0 else 0.0
        optional_score = min(optional_found / 2, 1.0) * 0.2  # Capped bonus
        
        return min(required_score + optional_score, 1.0)
    
    def _identify_strengths(
        self, skills_score: float, experience_score: float,
        education_score: float, keywords_score: float,
        content_quality_score: float
    ) -> List[str]:
        """Identify resume strengths based on scores."""
        strengths = []
        
        if skills_score >= 0.8:
            strengths.append("Strong technical skills coverage")
        if experience_score >= 0.8:
            strengths.append("Well-detailed professional experience")
        if education_score >= 0.85:
            strengths.append("Strong educational background")
        if keywords_score >= 0.8:
            strengths.append("Good use of industry keywords")
        if content_quality_score >= 0.85:
            strengths.append("Well-formatted and professional content")
        
        return strengths if strengths else ["Resume has foundational elements"]
    
    def _identify_weaknesses(
        self, skills_score: float, experience_score: float,
        education_score: float, keywords_score: float,
        content_quality_score: float, section_completeness_score: float
    ) -> List[str]:
        """Identify resume weaknesses based on scores."""
        weaknesses = []
        
        if skills_score < 0.5:
            weaknesses.append("Skills section needs expansion or clarity")
        if experience_score < 0.5:
            weaknesses.append("Work experience descriptions lack detail or impact")
        if education_score < 0.5:
            weaknesses.append("Education section could be stronger")
        if keywords_score < 0.5:
            weaknesses.append("Limited use of industry-relevant keywords")
        if content_quality_score < 0.6:
            weaknesses.append("Content quality and formatting need improvement")
        if section_completeness_score < 0.8:
            weaknesses.append("Some important sections are missing")
        
        return weaknesses if weaknesses else ["Minor areas for improvement"]
    
    def _generate_recommendations(
        self, weaknesses: List[str], missing_skills: List[str],
        sections_found: set, resume_text: str
    ) -> List[str]:
        """Generate specific recommendations for improvement."""
        recommendations = []
        
        # Skill-based recommendations
        if missing_skills and len(missing_skills) <= 3:
            skills_str = ', '.join(missing_skills[:3])
            recommendations.append(f"Add or highlight experience with: {skills_str}")
        
        # Section-based recommendations
        missing_sections = self.REQUIRED_SECTIONS - sections_found
        if missing_sections:
            section_str = ', '.join(missing_sections)
            recommendations.append(f"Add missing sections: {section_str}")
        
        # Content improvements
        if 'work experience descriptions lack detail or impact' in weaknesses:
            recommendations.append(
                "Use action verbs and quantifiable metrics in job descriptions"
            )
        
        if 'limited use of industry-relevant keywords' in weaknesses:
            recommendations.append(
                "Incorporate more industry-specific keywords relevant to your field"
            )
        
        # Additional suggestions
        if len(resume_text) < 400:
            recommendations.append("Expand resume content for better coverage")
        
        if not any(
            keyword in resume_text
            for keyword in ['certified', 'award', 'recognition']
        ):
            recommendations.append("Add certifications or awards if applicable")
        
        return recommendations if recommendations else ["Resume is well-structured"]
