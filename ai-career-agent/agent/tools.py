"""Focused Python tools available to the career agent."""
import json
import re

from agent.prompts import (
    CAREER_RECOMMENDATION_PROMPT,
    RESUME_ANALYSIS_PROMPT,
    ROADMAP_PROMPT,
    SKILL_GAP_PROMPT,
)


def analyze_resume(client, resume_text: str) -> dict:
    """Extract evidence-based resume details as structured JSON."""
    return client.generate_json(RESUME_ANALYSIS_PROMPT, resume_text, temperature=0.1)


def calculate_skill_gap(client, profile: dict, context: str) -> dict | list:
    """Compatibility wrapper for callers that need the deterministic comparison."""
    return calculate_skill_gap_fallback(profile, context)


_SKILL_ALIASES = {
    "python": ("python",),
    "r": ("r", "r programming"),
    "sql": ("sql", "structured query language"),
    "excel / spreadsheets": ("excel", "spreadsheets", "spreadsheet analysis", "spreadsheet formulas"),
    "statistics": ("statistics", "statistical analysis", "descriptive statistics", "probability"),
    "data cleaning": ("data cleaning", "data cleansing", "cleaning datasets", "cleaned datasets", "data validation", "data wrangling", "data manipulation", "data preparation"),
    "data visualization": ("data visualization", "visualization", "power bi", "tableau", "dashboard", "dashboards", "charts"),
    "communicating analytical results": ("communicating analytical results", "communicating insights", "presenting insights", "data storytelling", "stakeholder communication"),
    "machine learning fundamentals": ("machine learning fundamentals", "machine learning", "ml fundamentals", "ml basics"),
    "model evaluation": ("model evaluation", "model metrics", "evaluating models", "evaluation metrics"),
    "software engineering": ("software engineering", "software development", "maintainable code"),
    "data pipeline concepts": ("data pipeline concepts", "data pipelines", "pipelines", "etl"),
    "object-oriented programming": ("object-oriented programming", "object oriented programming", "oop"),
    "requirements analysis": ("requirements analysis", "requirements gathering", "requirements writing", "documented requirements"),
    "process modeling": ("process modeling", "process mapping", "process maps", "bpmn"),
    "presentation": ("presentation", "presentations", "presenting"),
    "communication": ("communication", "communicated", "stakeholder communication"),
    "testing": ("testing", "tests", "pytest", "unit testing"),
    "apis": ("apis", "api", "http api", "rest api"),
    "llm concepts": ("llm concepts", "llm", "large language models", "language models"),
    "prompt design": ("prompt design", "prompting", "prompt construction", "prompt engineering"),
    "data handling": ("data handling", "data processing", "data management"),
    "embeddings": ("embeddings", "embedding"),
    "vector search": ("vector search", "vector database", "vector store", "faiss"),
    "rag": ("rag", "retrieval augmented generation", "retrieval-augmented generation"),
    "git": ("git", "github", "version control"),
    "pandas": ("pandas",),
    "numpy": ("numpy",),
    "power bi": ("power bi",),
    "tableau": ("tableau",),
    "scikit-learn": ("scikit-learn", "sklearn"),
    "docker": ("docker", "containers", "containerization"),
    "cloud/aws": ("aws", "amazon web services", "cloud", "cloud computing", "cloud services", "cloud platform"),
    "git/github": ("git", "github", "git/github", "version control"),
}

_LEVELS = {"Basic": 1, "Intermediate": 2, "Advanced": 3}
_PROFILE_SKILL_FIELDS = ("technical_skills", "soft_skills", "skills")
_REQUIREMENT_SECTIONS = {
    "required technical skills": ("required", "Intermediate"),
    "soft skills": ("required", "Intermediate"),
    "preferred skills": ("preferred", "Basic"),
    "beginner topics": ("required", "Basic"),
    "intermediate topics": ("required", "Intermediate"),
    "advanced topics": ("required", "Advanced"),
}


def has_structured_role_requirements(context: str) -> bool:
    text = context or ""
    requirements = _extract_role_requirements(text)
    return (
        bool(re.search(r"(?i)\brequired technical skills:", text))
        and bool(re.search(r"(?i)\bsoft skills:", text))
        and len(requirements) >= 6
    )


def calculate_skill_gap_fallback(profile: dict, context: str, resume_text: str = "") -> dict:
    """Build a deterministic comparison from explicit resume evidence and role requirements."""
    return _compare_resume_requirements(profile, resume_text, _extract_role_requirements(context))


_DATA_ANALYST_REQUIREMENTS = (
    ("Python", "Intermediate", "High"),
    ("SQL", "Advanced", "High"),
    ("Excel", "Intermediate", "High"),
    ("Power BI", "Intermediate", "High"),
    ("Statistics", "Intermediate", "High"),
    ("Data Visualization", "Intermediate", "High"),
    ("Pandas", "Intermediate", "High"),
    ("NumPy", "Basic", "Medium"),
    ("Machine Learning", "Basic", "Medium"),
    ("Tableau", "Basic", "Medium"),
    ("Cloud/AWS", "Basic", "Medium"),
    ("Git/GitHub", "Basic", "Low"),
)


def data_analyst_fallback_context() -> str:
    """Describe the emergency role baseline in the existing retrieved-context panel."""
    return "Data Analyst presentation fallback requirements:\n" + "\n".join(
        f"{skill}: {level} (priority {priority})" for skill, level, priority in _DATA_ANALYST_REQUIREMENTS
    )


def calculate_data_analyst_fallback(profile: dict, resume_text: str = "") -> dict:
    requirements = []
    for skill, level, priority in _DATA_ANALYST_REQUIREMENTS:
        requirements.append({
            "skill": skill,
            "alternatives": [skill],
            "required_level": level,
            "requirement_type": "required",
            "priority": priority,
        })
    comparison = _compare_resume_requirements(profile, resume_text, requirements)
    learning_order = comparison.get("learning_order", [])
    if learning_order:
        comparison["recommendations"].append(
            "Recommended learning order: " + " → ".join(learning_order)
        )
    return comparison


def _compare_resume_requirements(profile: dict, resume_text: str, requirements: list[dict]) -> dict:
    candidate_entries = _candidate_skill_entries(profile)
    resume = resume_text if isinstance(resume_text, str) else ""
    matched, missing, improve = [], [], []
    seen = set()

    for requirement in requirements:
        skill = requirement["skill"]
        key = _normalized(skill)
        if key in seen:
            continue
        seen.add(key)
        evidence = _find_resume_evidence(skill, requirement["alternatives"], candidate_entries, resume)
        candidate_level = _candidate_level(skill, evidence, candidate_entries, resume) if evidence else "Missing"
        required_level = requirement["required_level"]
        gap = _skill_gap(candidate_level, required_level)
        status = "Missing" if candidate_level == "Missing" else "Available" if gap == "None" else "Needs improvement"
        if evidence:
            reason = f"Resume evidence for {evidence['matched_skill']}: {evidence['snippet']} Required level is {required_level}; candidate level is {candidate_level}."
        else:
            reason = f"The resume does not mention {skill}; the selected role lists it as a requirement at {required_level} level."
        if requirement.get("priority") and gap != "None":
            reason += f" Priority: {requirement['priority']}."
        row = {
            "skill": skill,
            "candidate_level": candidate_level,
            "required_level": required_level,
            "gap": gap,
            "reason": reason,
            "status": status,
            "explanation": reason,
            "evidence": evidence["snippet"] if evidence else "",
            "requirement_type": requirement["requirement_type"],
            "priority": requirement.get("priority", ""),
        }
        if candidate_level == "Missing":
            missing.append(row)
        elif gap == "None":
            matched.append(row)
        else:
            improve.append(row)

    all_rows = matched + improve + missing
    priority_value = {"High": 3, "Medium": 2, "Low": 1}
    priority_rows = sorted(
        all_rows,
        key=lambda row: (
            -(priority_value.get(row.get("priority"), 0) or (3 if row["gap"] == "High" else 2 if row["gap"] == "Medium" else 1)),
            row["skill"].casefold(),
        ),
    )
    strong_skills = [row["skill"] for row in matched if row["candidate_level"] == "Strong"]
    high_gaps = [row["skill"] for row in priority_rows if row["gap"] == "High"]
    summary_parts = []
    if strong_skills:
        summary_parts.append("Strong skills: " + ", ".join(strong_skills[:5]))
    if high_gaps:
        summary_parts.append("Highest-priority gaps: " + ", ".join(high_gaps[:5]))
    skill_summary = ". ".join(summary_parts) or f"The comparison covers {len(all_rows)} skills required for the selected role."
    recommendations = [
        f"Recommended: build {row['skill']} to {row['required_level']} level with a Data Analyst exercise, then add that work to your portfolio."
        for row in priority_rows if row["gap"] != "None"
    ][:5]
    if not recommendations and all_rows:
        recommendations = [
            f"Recommended: demonstrate {row['skill']} at {row['required_level']} level in a Data Analyst portfolio project."
            for row in matched[:3]
        ]
    return {
        "matching_skills": matched,
        "missing_skills": missing,
        "skills_to_improve": improve,
        "recommendations": recommendations,
        "skill_gaps": all_rows,
        "learning_order": [row["skill"] for row in priority_rows if row["gap"] != "None"],
        "skill_summary": skill_summary,
    }


def explain_skill_gap(client, role: str, comparison: dict, context: str) -> dict:
    """Ask Gemini to explain only the deterministic, evidence-grounded comparison."""
    payload = {
        "role": role,
        "comparison": comparison.get("skill_gaps", []),
        "retrieved_role_requirements": context,
    }
    system_prompt = SKILL_GAP_PROMPT.format(
        role=role,
        comparison=json.dumps(payload["comparison"], ensure_ascii=False),
        context=context,
    )
    return _json_completion(client, system_prompt, "Return the requested JSON explanation for this supplied, fixed comparison.")


def apply_skill_gap_explanation(comparison: dict, explanation: dict) -> dict:
    """Merge validated natural-language reasons and advice without changing evidence labels."""
    if not isinstance(explanation, dict):
        return comparison
    reason_map = {}
    for item in explanation.get("gap_explanations", []):
        if isinstance(item, dict) and item.get("skill") and item.get("reason"):
            reason_map[_normalized(item["skill"])] = str(item["reason"]).strip()
    for row in comparison.get("skill_gaps", []):
        reason = reason_map.get(_normalized(row["skill"]))
        if reason:
            row["reason"] = reason
        row["explanation"] = (
            f"{row['reason']} Candidate level: {row['candidate_level']}; "
            f"required level: {row['required_level']}; gap: {row['gap']}."
        )

    recs = explanation.get("recommendations")
    if isinstance(recs, list):
        recs = [str(item.get("recommendation") or item.get("action") or item.get("text") or "").strip()
                if isinstance(item, dict) else str(item).strip() for item in recs]
        recs = [item for item in recs if item]
        if recs:
            comparison["recommendations"] = recs[:6]
    learning_order = explanation.get("learning_order")
    if isinstance(learning_order, list):
        known = {_normalized(row["skill"]): row["skill"] for row in comparison.get("skill_gaps", [])}
        normalized_order = []
        for item in learning_order:
            name = known.get(_normalized(item))
            if name and name not in normalized_order:
                normalized_order.append(name)
        if normalized_order:
            comparison["learning_order"] = normalized_order
    comparison["skill_summary"] = str(explanation.get("skill_summary") or comparison.get("skill_summary") or "").strip()
    if comparison["skill_summary"]:
        comparison["recommendations"] = [f"Skill summary: {comparison['skill_summary']}"] + comparison.get("recommendations", [])
    if comparison.get("learning_order"):
        comparison["recommendations"] = comparison.get("recommendations", []) + [
            "Recommended learning order: " + " → ".join(comparison["learning_order"])
        ]
    return comparison


def _extract_role_requirements(context: str) -> list[dict]:
    requirements = []
    section_names = "required technical skills|soft skills|preferred skills|beginner topics|intermediate topics|advanced topics"
    boundaries = section_names + r"|job description|responsibilities|education|common tools|suggested learning topics"
    pattern = re.compile(rf"({section_names}):\s*(.*?)(?=\s+(?:{boundaries}):\s|$)", re.I | re.S)
    for match in pattern.finditer(context or ""):
        section, content = match.groups()
        requirement_type, default_level = _REQUIREMENT_SECTIONS[section.lower()]
        # Role knowledge files use sentences for each labeled section; ignore descriptions after the first.
        full_content = content
        content = re.split(r"\.\s", content.strip())[0].rstrip(". ")
        if section.lower() == "required technical skills":
            common = re.search(r"(?i)\b(python|sql)\s+is commonly required", full_content)
            if common:
                content += ", " + common.group(1)
        for part in re.split(r"\s*(?:,|;|\band\b)\s*", content, flags=re.I):
            original_part = part
            part = re.sub(r"(?i)^(?:strong|basic|advanced|intermediate|proficient in|experience with)\s+", "", part).strip(" .:-")
            if not part:
                continue
            alternatives = [value.strip() for value in re.split(r"\s*\bor\b\s*|\s*/\s*", part, flags=re.I) if value.strip()]
            canonical = list(dict.fromkeys(_canonical_skill(value) for value in alternatives))
            canonical = [value for value in canonical if value]
            if not canonical:
                continue
            display = " or ".join(canonical)
            level = _required_level(original_part, section.lower(), default_level)
            requirements.append({"skill": display, "alternatives": alternatives, "required_level": level, "requirement_type": requirement_type})
    return requirements


def _candidate_skill_entries(profile: dict) -> list[dict]:
    entries = []
    if not isinstance(profile, dict):
        return entries
    for field in _PROFILE_SKILL_FIELDS:
        values = profile.get(field, [])
        if isinstance(values, (str, dict)):
            values = [values]
        if isinstance(values, list) and len(values) == 1 and isinstance(values[0], dict):
            nested = next((values[0].get(key) for key in ("skills", "items", "technical_skills", "values") if isinstance(values[0].get(key), list)), None)
            if nested is not None:
                values = nested
        if not isinstance(values, list):
            continue
        for value in values:
            if isinstance(value, dict):
                name = value.get("skill") or value.get("name") or value.get("title") or value.get("text") or ""
                level = value.get("level") or value.get("proficiency") or value.get("status") or ""
                detail = value.get("evidence") or value.get("details") or value.get("description") or ""
            else:
                name, level, detail = str(value), "", ""
            if name:
                entries.append({"name": str(name), "level": str(level), "detail": str(detail)})
    return entries


def _find_resume_evidence(skill, alternatives, candidate_entries, resume_text):
    terms = []
    for alternative in alternatives:
        terms.extend(_skill_terms(alternative))
    resume_matches = []
    for sentence in _resume_evidence_segments(resume_text):
        for term in sorted(set(terms), key=len, reverse=True):
            if len(term) == 1:
                continue
            if re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", sentence, re.I):
                alternative = next((alt for alt in alternatives if term in _skill_terms(alt)), skill)
                resume_matches.append({"matched_skill": alternative, "snippet": _clean_snippet(sentence), "source": "resume"})
                break
    demonstrated = next((item for item in resume_matches if _demonstrated_use(item["snippet"])), None)
    if demonstrated:
        return demonstrated
    if resume_matches:
        return resume_matches[0]
    for alternative in alternatives:
        canonical = _canonical_skill(alternative)
        for entry in candidate_entries:
            if canonical == _canonical_skill(entry["name"]):
                snippet = entry["detail"] or entry["name"]
                return {"matched_skill": entry["name"], "snippet": _clean_snippet(snippet), "source": "profile"}
    return None


def _candidate_level(skill, evidence, candidate_entries, resume_text):
    profile_level = next((entry["level"] for entry in candidate_entries if _canonical_skill(entry["name"]) == _canonical_skill(evidence["matched_skill"]) and entry["level"]), "")
    evidence_text = " ".join((profile_level, evidence["snippet"]))
    terms = _skill_terms(evidence["matched_skill"])
    for term in terms:
        if len(term) == 1:
            continue
        nearby = re.findall(rf"(?:\b(?:strong|advanced|expert|proficient|highly skilled|extensive)\b.{{0,30}}\b{re.escape(term)}\b|\b{re.escape(term)}\b.{{0,30}}\b(?:strong|advanced|expert|proficient|highly skilled|extensive)\b)", evidence_text, re.I)
        if nearby:
            return "Strong"
        nearby = re.findall(rf"(?:\b(?:intermediate|working knowledge)\b.{{0,30}}\b{re.escape(term)}\b|\b{re.escape(term)}\b.{{0,30}}\b(?:intermediate|working knowledge)\b)", evidence_text, re.I)
        if nearby:
            return "Intermediate"
        nearby = re.findall(rf"(?:\b(?:basic|beginner|foundational|fundamental|introductory)\b.{{0,30}}\b{re.escape(term)}\b|\b{re.escape(term)}\b.{{0,30}}\b(?:basic|beginner|foundational|fundamental|introductory)\b)", evidence_text, re.I)
        if nearby:
            return "Basic"
    if profile_level:
        normalized_level = profile_level.casefold()
        if any(label in normalized_level for label in ("strong", "advanced", "expert", "proficient")):
            return "Strong"
        if "intermediate" in normalized_level:
            return "Intermediate"
        if any(label in normalized_level for label in ("basic", "beginner", "foundational")):
            return "Basic"
    if _demonstrated_use(evidence["snippet"]):
        return "Intermediate"
    return "Basic"


def _demonstrated_use(text):
    return bool(re.search(r"(?i)\b(built|developed|created|used|applied|analyzed|cleaned|queried|implemented|automated|designed|tested|maintained|deployed|evaluated|prepared|produced|managed)\b", text or ""))


def _resume_evidence_segments(resume_text):
    skill_heading = re.compile(r"(?i)^\s*(?:(?:(?:technical|key|core)\s+)?skills?|core competencies|technical proficiencies|tools and technologies)\s*:?\s*(.*)$")
    other_heading = re.compile(r"(?i)^\s*(?:work experience|professional experience|experience|employment|projects|education|certifications|summary|profile|objective|references)\s*:?\s*(.*)$")
    in_skills = False
    for line in (resume_text or "").splitlines():
        candidate = re.sub(r"^[\s•·*\-]+", "", line).strip()
        if not candidate:
            continue
        skill_match = skill_heading.match(candidate)
        if skill_match:
            in_skills = True
            content = skill_match.group(1).strip()
            if content:
                yield content
            continue
        other_match = other_heading.match(candidate)
        if other_match:
            in_skills = False
            candidate = other_match.group(1).strip()
            if not candidate:
                continue
        for sentence in re.split(r"(?<=[.!?])\s+", candidate):
            if in_skills or _demonstrated_use(sentence):
                yield sentence


def _skill_gap(candidate_level, required_level):
    if candidate_level == "Missing":
        return "High"
    difference = _LEVELS[required_level] - _LEVELS[candidate_level]
    if difference <= 0:
        return "None"
    if difference >= 2 or (candidate_level == "Intermediate" and required_level == "Advanced"):
        return "High"
    if candidate_level == "Basic" and required_level == "Intermediate":
        return "Medium"
    return "Low"


def _required_level(skill, section, default_level):
    text = f"{section} {skill}".casefold()
    if "advanced" in text:
        return "Advanced"
    if "intermediate" in text:
        return "Intermediate"
    if "beginner" in text or "basic" in text:
        return "Basic"
    if any(marker in text for marker in ("strong", "advanced", "expert", "proficient")):
        return "Advanced"
    return default_level


def _canonical_skill(value):
    normalized = _normalized(value)
    for canonical, aliases in _SKILL_ALIASES.items():
        if normalized in {_normalized(alias) for alias in aliases} or normalized == _normalized(canonical):
            return canonical
    return re.sub(r"\s+", " ", str(value).strip()).title()


def _skill_terms(value):
    canonical = _canonical_skill(value)
    aliases = _SKILL_ALIASES.get(canonical.casefold(), (canonical,))
    return [str(term).casefold() for term in aliases]


def _normalized(value):
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def _clean_snippet(value):
    return re.sub(r"\s+", " ", str(value or "")).strip(" .\n")[:220]


def recommend_roles(client, profile: dict, context: str) -> list:
    data = _json_completion(
        client,
        CAREER_RECOMMENDATION_PROMPT,
        f"Resume: {json.dumps(profile)}\nRetrieved context: {context}",
    )
    return data.get("recommendations", data.get("roles", [])) if isinstance(data, dict) else data


def generate_learning_roadmap(client, role: str, gaps: list, skills: list, context: str) -> str:
    prompt = ROADMAP_PROMPT.format(
        role=role,
        gaps=json.dumps(gaps),
        skills=json.dumps(skills),
        context=context,
    )
    return client.generate_response(prompt, temperature=0.4)


def _json_completion(client, system_prompt: str, user_prompt: str) -> dict:
    return client.generate_json(system_prompt, user_prompt, temperature=0.2)
