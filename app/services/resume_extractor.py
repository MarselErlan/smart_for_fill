"""
RESUME EXTRACTOR MODULE - Advanced PDF Resume Processing

This module provides comprehensive tools to:
1. Extract text from PDF resumes with precise formatting preservation
2. Identify and structure resume sections (experience, education, skills, etc.)
3. Extract contact information with intelligent pattern recognition
4. Generate embeddings for AI-powered job matching
5. Analyze resume sections for completeness and quality
6. Detect key skills and competencies with advanced NLP

The module supports both basic text extraction and advanced AI-powered analysis.
"""

import fitz  # PyMuPDF - Library for reading PDF files
import re
import os
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path

# Optional imports with graceful fallback
try:
    import openai
    from loguru import logger
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    import logging as logger

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_openai import OpenAIEmbeddings
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Constants for resume section identification
RESUME_SECTIONS = {
    "contact": ["contact", "contact information", "personal information", "personal details"],
    "summary": ["summary", "professional summary", "profile", "about me", "overview"],
    "experience": ["experience", "work experience", "professional experience", "employment history", "work history"],
    "education": ["education", "academic background", "academic history", "qualifications"],
    "skills": ["skills", "technical skills", "core competencies", "competencies", "expertise"],
    "projects": ["projects", "personal projects", "key projects", "professional projects"],
    "certifications": ["certifications", "certificates", "professional certifications", "credentials"],
    "languages": ["languages", "language proficiency", "language skills"],
    "achievements": ["achievements", "awards", "honors", "recognitions"],
    "interests": ["interests", "hobbies", "activities", "volunteer", "volunteering"]
}

# Contact information patterns
CONTACT_PATTERNS = {
    "email": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
    "phone": r"(?:\+\d{1,3}[\s-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}",
    "linkedin": r"linkedin\.com\/in\/[\w\-]+\/?",
    "github": r"github\.com\/[\w\-]+\/?",
    "website": r"https?:\/\/(?:www\.)?[\w\-]+\.[\w\-]+(?:\.[\w\-]+)*(?:\/[\w\-\.\/\?\%\&\=\#]*)?",
    "address": r"\d+\s+[\w\s]+,\s+[\w\s]+,\s+[A-Z]{2}\s+\d{5}"
}

# Common job titles for matching
COMMON_JOB_TITLES = [
    "software engineer", "developer", "data scientist", "data analyst", 
    "product manager", "project manager", "engineer", "analyst", "designer",
    "researcher", "consultant", "architect", "administrator", "specialist",
    "director", "manager", "lead", "head", "chief", "officer", "intern"
]

@dataclass
class Contact:
    """Contact information extracted from resume"""
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None
    website: Optional[str] = None
    address: Optional[str] = None
    
@dataclass
class Experience:
    """Work experience entry"""
    company: str
    title: str
    start_date: str
    end_date: Optional[str] = None
    description: Optional[str] = None
    location: Optional[str] = None
    is_current: bool = False
    
@dataclass
class Education:
    """Education entry"""
    institution: str
    degree: Optional[str] = None
    field: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    gpa: Optional[str] = None
    location: Optional[str] = None
    
@dataclass
class Skill:
    """Skill entry with optional category"""
    name: str
    category: Optional[str] = None
    level: Optional[str] = None
    
@dataclass
class ResumeStructure:
    """Complete parsed resume structure"""
    raw_text: str
    contact: Contact
    summary: Optional[str] = None
    experience: List[Experience] = None
    education: List[Education] = None
    skills: List[Skill] = None
    projects: Optional[List[Dict]] = None
    certifications: Optional[List[Dict]] = None
    languages: Optional[List[Dict]] = None
    achievements: Optional[List[str]] = None
    interests: Optional[List[str]] = None
    sections: Dict[str, str] = None
    
    def __post_init__(self):
        if self.experience is None:
            self.experience = []
        if self.education is None:
            self.education = []
        if self.skills is None:
            self.skills = []
        if self.sections is None:
            self.sections = {}

def extract_text_from_resume(pdf_path: str) -> str:
    """
    Extract text content from a PDF resume file.
    
    Args:
        pdf_path (str): Path to the PDF resume file
        
    Returns:
        str: Extracted and cleaned text from the PDF
    """
    logger.info(f"Starting text extraction from PDF: {pdf_path}")
    
    try:
        text = ""
        with fitz.open(pdf_path) as doc:
            total_pages = len(doc)
            logger.info(f"PDF has {total_pages} pages")
            
            # Loop through each page in the PDF
            for page_num, page in enumerate(doc, 1):
                logger.debug(f"Processing page {page_num}/{total_pages}")
                # Extract text from current page and add to our text string
                page_text = page.get_text()
                text += page_text
            
        # Remove extra whitespace and return clean text
        cleaned_text = text.strip()
        logger.info(f"Successfully extracted {len(cleaned_text)} characters from PDF")
        return cleaned_text
        
    except Exception as e:
        logger.error(f"Error extracting text from PDF {pdf_path}: {str(e)}")
        raise

def extract_contact_information(text: str) -> Contact:
    """
    Extract contact information from resume text.
    
    Args:
        text (str): Complete resume text
        
    Returns:
        Contact: Structured contact information
    """
    contact = Contact()
    
    # Extract email
    email_match = re.search(CONTACT_PATTERNS["email"], text)
    if email_match:
        contact.email = email_match.group(0)
    
    # Extract phone
    phone_match = re.search(CONTACT_PATTERNS["phone"], text)
    if phone_match:
        contact.phone = phone_match.group(0)
    
    # Extract LinkedIn profile
    linkedin_match = re.search(CONTACT_PATTERNS["linkedin"], text)
    if linkedin_match:
        contact.linkedin = "https://www." + linkedin_match.group(0)
    
    # Extract GitHub profile
    github_match = re.search(CONTACT_PATTERNS["github"], text)
    if github_match:
        contact.github = "https://www." + github_match.group(0)
    
    # Extract website
    website_match = re.search(CONTACT_PATTERNS["website"], text)
    if website_match:
        contact.website = website_match.group(0)
    
    # Try to extract name from first lines of resume
    # This is a simple heuristic that assumes the name is in the first 5 lines
    first_lines = text.split('\n')[:5]
    for line in first_lines:
        # Look for single line with just a name (likely the header)
        line = line.strip()
        if len(line.split()) <= 3 and len(line) > 3 and line.lower() not in [
            'resume', 'curriculum vitae', 'cv', 'profile', 'contact'
        ]:
            contact.name = line
            break
    
    return contact

def identify_resume_sections(text: str) -> Dict[str, str]:
    """
    Identify standard resume sections from the text.
    
    Args:
        text (str): Complete resume text
        
    Returns:
        Dict[str, str]: Dictionary of section name -> section content
    """
    sections = {}
    lines = text.split('\n')
    
    # First pass: identify potential section headers
    potential_headers = []
    for i, line in enumerate(lines):
        line = line.strip().lower()
        if not line:
            continue
            
        # Check if this line matches any known section
        for section_name, variations in RESUME_SECTIONS.items():
            if any(line.startswith(var.lower()) or line == var.lower() for var in variations):
                potential_headers.append((i, section_name))
    
    # Second pass: extract section content
    for i in range(len(potential_headers)):
        section_idx, section_name = potential_headers[i]
        
        # Determine section end
        if i < len(potential_headers) - 1:
            next_section_idx = potential_headers[i + 1][0]
            section_content = '\n'.join(lines[section_idx+1:next_section_idx])
        else:
            section_content = '\n'.join(lines[section_idx+1:])
        
        sections[section_name] = section_content.strip()
    
    return sections

def extract_skills(text: str) -> List[Skill]:
    """
    Extract skills from resume text.
    
    Args:
        text (str): Complete resume text or skills section text
        
    Returns:
        List[Skill]: List of identified skills
    """
    skills = []
    
    # Common skill categories
    tech_skills = ["python", "java", "javascript", "c++", "sql", "react", "angular", "tensorflow", 
                  "pytorch", "aws", "azure", "docker", "kubernetes", "html", "css", "git"]
    soft_skills = ["leadership", "communication", "teamwork", "problem solving", "critical thinking",
                  "time management", "adaptability", "creativity", "collaboration"]
    
    # Simple skill extraction by keyword matching
    words = re.findall(r'\b\w+(?:-\w+)*\b', text.lower())
    
    # Extract tech skills
    for skill in tech_skills:
        if skill in text.lower():
            skills.append(Skill(name=skill, category="Technical"))
    
    # Extract soft skills
    for skill in soft_skills:
        if skill in text.lower():
            skills.append(Skill(name=skill, category="Soft"))
    
    return skills

def extract_experiences(experience_text: str) -> List[Experience]:
    """
    Extract work experience entries from experience section.
    
    Args:
        experience_text (str): Experience section text
        
    Returns:
        List[Experience]: List of parsed experience entries
    """
    experiences = []
    
    # Basic pattern to identify experience entries (can be improved for more complex formats)
    # This assumes each experience entry starts with company name or job title
    experience_blocks = re.split(r'\n\s*\n', experience_text)
    
    for block in experience_blocks:
        if not block.strip():
            continue
            
        lines = block.split('\n')
        if not lines:
            continue
            
        # Try to extract company and title
        company = ""
        title = ""
        dates = ""
        description = ""
        
        # First line typically contains title and company
        first_line = lines[0].strip()
        for job_title in COMMON_JOB_TITLES:
            if job_title in first_line.lower():
                parts = first_line.split(job_title, 1)
                if len(parts) == 2:
                    company = parts[0].strip()
                    title = job_title + parts[1].strip()
                break
        
        # If we couldn't extract company/title, use simple heuristics
        if not company or not title:
            if '|' in first_line:
                parts = first_line.split('|')
                title = parts[0].strip()
                company = parts[1].strip()
            elif '-' in first_line:
                parts = first_line.split('-')
                title = parts[0].strip()
                company = parts[1].strip()
            elif ',' in first_line:
                parts = first_line.split(',')
                title = parts[0].strip()
                company = ','.join(parts[1:]).strip()
            else:
                title = first_line
                if len(lines) > 1:
                    company = lines[1].strip()
        
        # Look for dates in the first 3 lines
        date_pattern = r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?|[0-9]{1,2}/[0-9]{1,2})[\s,]*[0-9]{4}\s*(?:-|to|–|until)\s*(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?|[0-9]{1,2}/[0-9]{1,2}|Present|Current|Now)[\s,]*(?:[0-9]{4})?'
        
        for i in range(min(3, len(lines))):
            date_match = re.search(date_pattern, lines[i], re.IGNORECASE)
            if date_match:
                dates = date_match.group(0).strip()
                break
                
        # Extract start and end dates
        start_date = ""
        end_date = None
        is_current = False
        
        if dates:
            if "present" in dates.lower() or "current" in dates.lower() or "now" in dates.lower():
                is_current = True
                date_parts = re.split(r'\s*(?:-|to|–|until)\s*', dates, maxsplit=1)
                if len(date_parts) == 2:
                    start_date = date_parts[0].strip()
            else:
                date_parts = re.split(r'\s*(?:-|to|–|until)\s*', dates, maxsplit=1)
                if len(date_parts) == 2:
                    start_date = date_parts[0].strip()
                    end_date = date_parts[1].strip()
        
        # Get description from remaining lines
        if len(lines) > 2:
            description = '\n'.join(lines[2:]).strip()
        
        if company and title:
            experiences.append(Experience(
                company=company,
                title=title,
                start_date=start_date,
                end_date=end_date,
                description=description,
                is_current=is_current
            ))
    
    return experiences

def extract_education(education_text: str) -> List[Education]:
    """
    Extract education entries from education section.
    
    Args:
        education_text (str): Education section text
        
    Returns:
        List[Education]: List of parsed education entries
    """
    education_entries = []
    
    # Common degree abbreviations and their full forms
    degree_patterns = [
        "Bachelor", "BS", "B.S.", "BA", "B.A.", "Master", "MS", "M.S.", "MA", "M.A.",
        "PhD", "Ph.D.", "MD", "M.D.", "JD", "J.D.", "MBA", "Associate", "Certificate"
    ]
    
    # Split into blocks
    education_blocks = re.split(r'\n\s*\n', education_text)
    
    for block in education_blocks:
        if not block.strip():
            continue
            
        lines = block.split('\n')
        if not lines:
            continue
            
        institution = ""
        degree = ""
        field = ""
        dates = ""
        gpa = ""
        
        # First line typically contains institution
        institution = lines[0].strip()
        
        # Look for degree and field in the next lines
        for i in range(1, min(3, len(lines))):
            line = lines[i].strip()
            
            # Look for degree patterns
            for pattern in degree_patterns:
                if pattern in line:
                    degree = line
                    break
            
            # Look for date patterns
            date_pattern = r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?|[0-9]{1,2}/[0-9]{1,2})[\s,]*[0-9]{4}\s*(?:-|to|–|until)\s*(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?|[0-9]{1,2}/[0-9]{1,2}|Present|Current|Now)[\s,]*(?:[0-9]{4})?'
            date_match = re.search(date_pattern, line, re.IGNORECASE)
            if date_match:
                dates = date_match.group(0).strip()
            
            # Look for GPA
            gpa_match = re.search(r'GPA:?\s*([0-9]\.[0-9]{1,2})', line, re.IGNORECASE)
            if gpa_match:
                gpa = gpa_match.group(1)
        
        # Extract field of study if applicable
        if degree and "in" in degree:
            parts = degree.split("in", 1)
            if len(parts) == 2:
                degree = parts[0].strip()
                field = parts[1].strip()
        
        # Extract start and end date
        start_date = None
        end_date = None
        if dates:
            date_parts = re.split(r'\s*(?:-|to|–|until)\s*', dates, maxsplit=1)
            if len(date_parts) == 2:
                start_date = date_parts[0].strip()
                end_date = date_parts[1].strip()
        
        if institution:
            education_entries.append(Education(
                institution=institution,
                degree=degree,
                field=field,
                start_date=start_date,
                end_date=end_date,
                gpa=gpa
            ))
    
    return education_entries

def analyze_resume(pdf_path: str) -> ResumeStructure:
    """
    Analyze a resume PDF and extract structured information.
    
    Args:
        pdf_path (str): Path to the PDF resume file
        
    Returns:
        ResumeStructure: Structured resume information
    """
    # Extract text from PDF
    text = extract_text_from_resume(pdf_path)
    
    # Extract basic contact information
    contact = extract_contact_information(text)
    
    # Identify resume sections
    sections = identify_resume_sections(text)
    
    # Extract skills (basic)
    skills = extract_skills(text)
    
    # Extract experiences if available
    experiences = []
    if "experience" in sections:
        experiences = extract_experiences(sections["experience"])
    
    # Extract education if available
    education = []
    if "education" in sections:
        education = extract_education(sections["education"])
    
    # Create resume structure
    resume = ResumeStructure(
        raw_text=text,
        contact=contact,
        summary=sections.get("summary"),
        experience=experiences,
        education=education,
        skills=skills,
        sections=sections
    )
    
    return resume

def generate_embeddings(text: str) -> Optional[List[float]]:
    """
    Generate embeddings for resume text for AI-powered matching.
    
    Args:
        text (str): The text to generate embeddings for
        
    Returns:
        Optional[List[float]]: Embedding vector or None if not available
    """
    if not OPENAI_AVAILABLE:
        logger.warning("OpenAI package not available - cannot generate embeddings")
        return None
    
    try:
        # Initialize OpenAI client
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not found in environment variables")
            return None
            
        client = openai.OpenAI(api_key=api_key)
        
        # Generate embeddings
        response = client.embeddings.create(
            input=[text],
            model="text-embedding-ada-002"
        )
        
        # Extract the embedding vector
        embedding = response.data[0].embedding
        logger.info(f"Generated embedding vector with {len(embedding)} dimensions")
        return embedding
        
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        return None

def save_resume_as_json(resume: ResumeStructure, output_path: str = None) -> str:
    """
    Save resume data as a JSON file.
    
    Args:
        resume (ResumeStructure): The resume structure to save
        output_path (str): Path to save the JSON file (optional)
        
    Returns:
        str: Path to the saved JSON file
    """
    # Convert resume structure to dictionary
    resume_dict = {
        "contact": {
            "name": resume.contact.name,
            "email": resume.contact.email,
            "phone": resume.contact.phone,
            "linkedin": resume.contact.linkedin,
            "github": resume.contact.github,
            "website": resume.contact.website,
            "address": resume.contact.address
        },
        "summary": resume.summary,
        "experience": [
            {
                "company": exp.company,
                "title": exp.title,
                "start_date": exp.start_date,
                "end_date": exp.end_date,
                "description": exp.description,
                "is_current": exp.is_current,
            } for exp in resume.experience
        ],
        "education": [
            {
                "institution": edu.institution,
                "degree": edu.degree,
                "field": edu.field,
                "start_date": edu.start_date,
                "end_date": edu.end_date,
                "gpa": edu.gpa
            } for edu in resume.education
        ],
        "skills": [
            {
                "name": skill.name,
                "category": skill.category
            } for skill in resume.skills
        ],
        "sections": resume.sections
    }
    
    # Generate output path if not provided
    if not output_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"resume_data_{timestamp}.json"
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Write to JSON file
    with open(output_path, 'w') as f:
        json.dump(resume_dict, f, indent=2)
    
    logger.info(f"Resume data saved to {output_path}")
    return output_path

class ResumeExtractor:
    def extract(self, file_path: str) -> Dict[str, Any]:
        """
        Parse the resume file and extract structured information (name, email, skills, etc.)
        """
        # TODO: Implement actual resume parsing logic (PDF, DOCX, etc.)
        # Example return: {"name": ..., "email": ..., "skills": [...], ...}
        return {}

class PersonalInfoVectorDB:
    def __init__(self, vector_db_client):
        self.client = vector_db_client

    def vectorize(self, personal_info: dict) -> list:
        # Convert personal info to vector(s)
        pass

    def store(self, user_id: str, vector: list, metadata: dict):
        # Store vector in DB
        pass

    def search(self, query_vector: list, top_k: int = 5):
        # Search similar vectors
        pass

if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        if not os.path.exists(pdf_path):
            print(f"Error: File {pdf_path} does not exist")
            sys.exit(1)
            
        print(f"Processing resume: {pdf_path}")
        resume = analyze_resume(pdf_path)
        
        # Display basic information
        print(f"\nName: {resume.contact.name}")
        print(f"Email: {resume.contact.email}")
        print(f"Phone: {resume.contact.phone}")
        print(f"\nFound {len(resume.experience)} experience entries")
        print(f"Found {len(resume.education)} education entries")
        print(f"Found {len(resume.skills)} skills")
        
        # Save to JSON
        output_path = pdf_path.replace(".pdf", "_data.json")
        save_path = save_resume_as_json(resume, output_path)
        print(f"\nSaved resume data to {save_path}")
    else:
        print("Usage: python resume_extractor.py path/to/resume.pdf") 
