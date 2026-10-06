"""Selectable career categories and local fallback requirements for target roles."""


CAREER_CATEGORIES = {
    "Technology": [
        "Data Analyst", "Data Scientist", "ML Engineer", "AI Engineer", "Python Developer",
        "Software Developer", "Cloud Engineer", "DevOps Engineer", "Cybersecurity Analyst",
    ],
    "Finance": [
        "Financial Analyst", "Investment Analyst", "Risk Analyst", "Banking Analyst",
        "Financial Planning Analyst",
    ],
    "Marketing": [
        "Marketing Analyst", "Digital Marketing Specialist", "SEO Specialist", "Brand Manager",
        "Marketing Manager",
    ],
    "Business": [
        "Business Analyst", "Product Analyst", "Product Manager", "Management Consultant",
        "Operations Analyst",
    ],
    "HR": ["HR Analyst", "Talent Acquisition Specialist", "HR Business Partner"],
    "Sales": ["Sales Analyst", "Business Development Executive", "Sales Manager"],
    "Design / Creative": ["UI/UX Designer", "Graphic Designer", "Content Strategist"],
    "Supply Chain / Operations": [
        "Supply Chain Analyst", "Procurement Analyst", "Operations Manager", "Logistics Analyst",
    ],
    "Healthcare": ["Healthcare Data Analyst", "Healthcare Business Analyst", "Health Informatics Analyst"],
}


# These are broad, locally stored skill baselines for newly selectable roles.
# Existing role documents continue to take precedence wherever they are present.
_CATEGORY_REQUIREMENTS = {
    "Technology": (
        "software development, Python, testing, Git/GitHub, APIs, SQL, data structures, cloud fundamentals",
        "Docker, CI/CD, system design, security fundamentals",
        "problem solving, communication, teamwork",
    ),
    "Finance": (
        "financial analysis, Excel, accounting, budgeting, financial modeling, statistics, data visualization",
        "SQL, forecasting, financial software",
        "attention to detail, communication, ethical judgment",
    ),
    "Marketing": (
        "marketing strategy, market research, campaign analytics, digital marketing, content development, SEO, Excel",
        "SQL, A/B testing, marketing automation, data visualization",
        "communication, creativity, project management",
    ),
    "Business": (
        "business analysis, requirements analysis, process modeling, data analysis, Excel, project management, SQL",
        "stakeholder research, data visualization, change management",
        "communication, structured thinking, collaboration",
    ),
    "HR": (
        "human resources, recruitment, interviewing, HRIS, HR analytics, reporting, Excel",
        "workforce planning, employment practices, data visualization",
        "communication, discretion, relationship building",
    ),
    "Sales": (
        "CRM, prospecting, negotiation, pipeline management, sales analytics, Excel, customer relationship management",
        "forecasting, account planning, presentation",
        "communication, persistence, relationship building",
    ),
    "Design / Creative": (
        "visual design, user experience, design tools, prototyping, accessibility, portfolio development",
        "user research, design systems, content strategy",
        "creativity, communication, collaboration",
    ),
    "Supply Chain / Operations": (
        "supply chain management, logistics, inventory management, procurement, Excel, ERP, forecasting",
        "SQL, process improvement, data visualization",
        "problem solving, communication, attention to detail",
    ),
    "Healthcare": (
        "healthcare operations, health informatics, data analysis, SQL, privacy and compliance, electronic health records, statistics",
        "data visualization, healthcare quality, interoperability",
        "communication, discretion, attention to detail",
    ),
}


def local_role_context(role: str) -> str:
    """Return a transparent category-level baseline when no role document exists."""
    requested = (role or "").strip().casefold()
    category = next(
        (name for name, roles in CAREER_CATEGORIES.items() if any(item.casefold() == requested for item in roles)),
        None,
    )
    if category is None:
        return ""

    required, preferred, soft = _CATEGORY_REQUIREMENTS[category]
    return (
        f"{role}\n"
        f"Source: Local {category} category-level baseline (not a retrieved job posting).\n"
        f"Job description: General preparation baseline for {role} within {category}.\n"
        f"Required technical skills: {required}.\n"
        f"Preferred skills: {preferred}.\n"
        f"Soft skills: {soft}.\n"
        f"Beginner topics: fundamentals of {role}, role-specific tools, and practical exercises.\n"
        f"Intermediate topics: applied {role} workflows, quality checks, and communicating outcomes.\n"
        f"Suggested learning topics: build a small role-relevant project and document its decisions and results."
    )
