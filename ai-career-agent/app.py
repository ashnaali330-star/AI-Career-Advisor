"""Streamlit interface for the AI Career Advisor Agent."""
import hashlib
import json
import re
from uuid import uuid4
from pathlib import Path

import streamlit as st

from agent.career_agent import CareerAgent
from agent.career_catalog import CAREER_CATEGORIES
from rag.retriever import RoleRetriever
from resume.parser import parse_resume
from resume.analyzer import get_profile_section
from utils.helpers import create_llm_client, load_settings, safe_error_message
from utils.voice_ui import clean_text_for_speech, voice_defaults, voice_player, voice_settings, voice_toolbar
from utils.speech_services import transcribe_audio_async

ROOT = Path(__file__).parent
ROLES = ["Data Analyst", "Data Scientist", "AI Engineer", "ML Engineer", "Python Developer", "Business Analyst"]
st.set_page_config(
    page_title="AI Career Advisor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource(show_spinner="Preparing role knowledge search…")
def make_retriever(model_name: str):
    return RoleRetriever(str(ROOT / "data" / "job_roles"), str(ROOT / "vector_db"), model_name)


@st.cache_resource(show_spinner=False)
def get_agent(settings):
    client = create_llm_client(settings)
    if client is None:
        return None
    return CareerAgent(client, make_retriever(settings["embedding_model"]))


@st.cache_resource(show_spinner=False)
def _app_instance_id():
    """Keep a stable startup marker across reruns, but renew it after restarts."""
    return uuid4().hex


def _normal_key(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _find_value(data, aliases, fallback="Not specified"):
    if not isinstance(data, dict):
        return fallback
    normalized = {_normal_key(key): value for key, value in data.items()}
    for alias in aliases:
        value = normalized.get(_normal_key(alias))
        if value not in (None, "", [], {}):
            return value
    return fallback


def _plain_text(value):
    if value is None or value == "":
        return ""
    if isinstance(value, (str, int, float)):
        return str(value).strip()
    if isinstance(value, list):
        return ", ".join(text for item in value if (text := _plain_text(item)))
    if isinstance(value, dict):
        preferred = ("name", "skill", "title", "value", "text", "description", "summary")
        selected = _find_value(value, preferred, fallback=None)
        if selected is not None:
            return _plain_text(selected)
        return ", ".join(f"{key}: {_plain_text(item)}" for key, item in value.items() if _plain_text(item))
    return str(value)


def _as_items(value, wrapper_keys=()):
    if value in (None, "", []):
        return []
    if isinstance(value, dict):
        wrapped = _find_value(value, wrapper_keys, fallback=None) if wrapper_keys else None
        if wrapped is not None:
            return _as_items(wrapped, wrapper_keys)
        return [value]
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        if value.startswith("[") or value.startswith("{"):
            try:
                return _as_items(json.loads(value), wrapper_keys)
            except json.JSONDecodeError:
                pass
        return [line.strip(" •-*\t") for line in value.splitlines() if line.strip(" •-*\t")]
    return [value]


def _render_bullets(values, empty_message="Not specified"):
    items = [_plain_text(item) for item in values]
    items = [item for item in items if item]
    if items:
        for item in items:
            st.markdown(f"- {item}")
    else:
        st.caption(empty_message)


def _render_profile(profile):
    candidate = _find_value(profile, ("candidate_profile", "candidate", "profile"), fallback={})
    if not isinstance(candidate, dict):
        candidate = {"professional_title": candidate} if candidate else {}

    st.header("👤 1. Candidate profile")
    fields = [
        ("Name", ("name", "full_name", "candidate_name")),
        ("Professional title", ("professional_title", "title", "headline", "current_title")),
        ("Location", ("location", "city", "address")),
        ("Email", ("email", "email_address")),
        ("Phone", ("phone", "phone_number", "mobile")),
        ("LinkedIn", ("linkedin", "linkedin_url", "linkedin_profile")),
    ]
    with st.container(border=True):
        for label, aliases in fields:
            value = _find_value(candidate, aliases, fallback=None)
            if value is None:
                value = _find_value(profile, aliases)
            st.markdown(f"**{label}:** {_plain_text(value) or 'Not specified'}")

    st.header("🧠 2. Skills")
    technical = _as_items(_find_value(profile, ("technical_skills", "skills", "hard_skills")), ("technical_skills", "skills", "items"))
    soft = _as_items(_find_value(profile, ("soft_skills", "interpersonal_skills")), ("soft_skills", "skills", "items"))
    left, right = st.columns(2)
    with left:
        st.subheader("Technical skills")
        _render_bullets(technical)
    with right:
        st.subheader("People and communication skills")
        _render_bullets(soft)

    st.header("💼 3. Experience & projects")
    st.subheader("Work experience")
    _render_records(
        _find_value(profile, ("work_experience", "experience", "employment_history")),
        heading_keys=("title", "role", "position", "project_name", "name", "company"),
        field_labels={"company": "Company", "employer": "Company", "dates": "Dates", "duration": "Duration", "location": "Location", "summary": "Summary", "description": "Details", "responsibilities": "Responsibilities", "achievements": "Achievements"},
    )
    st.subheader("Projects")
    _render_records(
        _find_value(profile, ("projects", "portfolio_projects")),
        heading_keys=("project_name", "name", "title"),
        field_labels={"description": "Description", "summary": "Summary", "role": "Role", "technologies": "Tools and technologies", "tools": "Tools and technologies", "skills": "Skills used", "outcome": "Outcome", "impact": "Impact"},
    )

    st.header("🎓 4. Education & credentials")
    st.subheader("Education")
    _render_records(
        _find_value(profile, ("education", "educational_background", "degrees")),
        heading_keys=("degree", "qualification", "program", "title", "name"),
        field_labels={"institution": "Institution", "school": "Institution", "university": "Institution", "dates": "Dates", "graduation_year": "Graduation year", "field": "Field of study", "major": "Field of study", "details": "Details"},
    )
    st.subheader("Certifications")
    _render_records(
        _find_value(profile, ("certifications", "credentials", "licenses")),
        heading_keys=("name", "title", "certification", "credential"),
        field_labels={"issuer": "Issuer", "organization": "Issuer", "date": "Date", "year": "Year", "expiry": "Expiry"},
    )


def _render_records(value, heading_keys, field_labels):
    records = _as_items(value, ("items", "records", "experiences", "projects", "education", "certifications"))
    if not records:
        st.caption("Not specified")
        return

    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            st.markdown(f"- {_plain_text(record)}")
            continue
        heading = _plain_text(_find_value(record, heading_keys, fallback=None))
        heading = heading or f"Entry {index}"
        with st.container(border=True):
            st.markdown(f"**{heading}**")
            detail_lines = []
            for key, item in record.items():
                label = field_labels.get(_normal_key(key))
                text = _plain_text(item)
                if label and text and _normal_key(key) not in {_normal_key(alias) for alias in heading_keys}:
                    detail_lines.append(f"**{label}:** {text}")
                elif not label and _normal_key(key) not in {_normal_key(alias) for alias in heading_keys}:
                    text = _plain_text(item)
                    if text:
                        detail_lines.append(text)
            if detail_lines:
                for line in detail_lines:
                    st.markdown(f"- {line}")
            else:
                st.caption("No additional details provided")


def _render_skill_gaps(analysis, role="the selected role"):
    if isinstance(analysis, dict) and any(key in analysis for key in ("matching_skills", "missing_skills", "skills_to_improve")):
        available = rows_as_skill_entries(_as_items(analysis.get("matching_skills", [])))
        missing = rows_as_skill_entries(_as_items(analysis.get("missing_skills", [])))
        improve = rows_as_skill_entries(_as_items(analysis.get("skills_to_improve", [])))
        rows = available + missing + improve
    else:
        rows = rows_as_skill_entries(_as_items(analysis, ("skill_gaps", "skills", "items")))
        available, missing, improve = [], [], []
        for row in rows:
            status = _normal_key(row.get("status", ""))
            if any(token in status for token in ("available", "match", "have", "present")):
                available.append(row)
            elif any(token in status for token in ("improv", "developing", "partial", "intermediate")):
                improve.append(row)
            else:
                missing.append(row)

    _render_readiness_and_comparison(analysis, role)

    metric_columns = st.columns(3)
    metric_columns[0].metric("Skills matched", len(available))
    metric_columns[1].metric("Skills to improve", len(improve))
    metric_columns[2].metric("Missing skills", len(missing))

    have_column, missing_column, improve_column = st.columns(3)
    with have_column:
        st.subheader("🟢 Strong skills / already aligned")
        _render_gap_entries(available, "No matching skills were identified.")
    with missing_column:
        st.subheader("🔴 Missing skills")
        _render_gap_entries(missing, "No missing skills were identified.")
    with improve_column:
        st.subheader("🟡 Skills to improve")
        _render_gap_entries(improve, "No skills were marked for improvement.")

    with st.expander("All required skills and evidence", expanded=False):
        _render_gap_entries(rows, "No structured skill requirements were returned.", show_status=True)
    st.subheader("🎯 A few useful next steps")
    recommendations = _as_items(analysis.get("recommendations", [])) if isinstance(analysis, dict) else []
    _render_bullets(_unique_advice(recommendations), "No additional actions were returned.")


_LEVEL_RANK = {"missing": 0, "none": 0, "basic": 1, "beginner": 1, "foundational": 1,
               "intermediate": 2, "advanced": 3, "strong": 3, "expert": 3, "proficient": 3}
_PRIORITY_WEIGHT = {"high": 3, "medium": 2, "low": 1}


def _comparison_rows(analysis):
    """Normalize deterministic skill comparison rows for readiness and charting."""
    if not isinstance(analysis, dict):
        return []
    raw_rows = _as_items(analysis.get("skill_gaps", []))
    if not raw_rows:
        raw_rows = sum(
            (_as_items(analysis.get(key, [])) for key in ("matching_skills", "skills_to_improve", "missing_skills")),
            [],
        )

    rows = []
    for source in raw_rows:
        if not isinstance(source, dict):
            continue
        requirement_type = _normal_key(source.get("requirement_type", "required"))
        if "preferred" in requirement_type:
            continue
        skill = _plain_text(source.get("skill", ""))
        required_label = _plain_text(source.get("required_level", ""))
        required_rank = _LEVEL_RANK.get(required_label.casefold())
        if not skill or not required_rank:
            continue

        candidate_label = _plain_text(source.get("candidate_level", ""))
        status = _normal_key(source.get("status", ""))
        candidate_rank = _LEVEL_RANK.get(candidate_label.casefold())
        if candidate_rank is None:
            if "missing" in status:
                candidate_rank = 0
                candidate_label = "Missing"
            elif any(token in status for token in ("available", "match", "have", "present")):
                candidate_rank = required_rank
                candidate_label = required_label
            else:
                continue

        priority = _plain_text(source.get("priority", "")).casefold()
        weight = _PRIORITY_WEIGHT.get(priority, 1)
        ratio = min(candidate_rank / required_rank, 1.0)
        rows.append({
            "skill": skill,
            "candidate_level": candidate_label,
            "required_level": required_label,
            "candidate_rank": candidate_rank,
            "required_rank": required_rank,
            "candidate_percent": round(candidate_rank / 3 * 100),
            "required_percent": round(required_rank / 3 * 100),
            "ratio": ratio,
            "weight": weight,
            "gap_share": 0.0,
            "priority": priority.title() if priority else "Standard",
        })

    total_weight = sum(row["weight"] for row in rows)
    if not rows or not total_weight:
        return []
    for row in rows:
        row["gap_share"] = row["weight"] * (1 - row["ratio"]) / total_weight * 100
    earned = sum(row["weight"] * row["ratio"] for row in rows)
    readiness = round(earned / total_weight * 100)
    return rows, readiness


def _render_readiness_and_comparison(analysis, role):
    result = _comparison_rows(analysis)
    if not result:
        st.info("A level-based readiness score is unavailable for this comparison. Review the skill matches and gaps below.")
        return

    rows, readiness = result
    remaining = 100 - readiness
    st.subheader("🎯 Job Readiness")
    with st.container(horizontal=True):
        st.metric("Estimated readiness", f"{readiness}%", border=True)
        st.metric("Remaining gap", f"{remaining}%", border=True)
    st.progress(readiness / 100, text=f"{readiness}% aligned with {role}'s required skill levels")
    st.caption(
        "Deterministic estimate from the Python skill comparison: Missing = 0, Basic = 1, "
        "Intermediate = 2, Advanced/Strong = 3. Each candidate level is capped at the required level; "
        "existing High/Medium/Low priorities receive weights of 3/2/1. This is a planning estimate, not a test score."
    )

    st.subheader("📊 Your Skill Gap")
    chart_rows = sorted(rows, key=lambda row: (-row["weight"], -row["gap_share"], row["skill"].casefold()))[:8]
    chart_data = {
        "Skill": [row["skill"] for row in chart_rows],
        "Your level (%)": [row["candidate_percent"] for row in chart_rows],
        "Required level (%)": [row["required_percent"] for row in chart_rows],
    }
    st.bar_chart(
        chart_data,
        x="Skill",
        y=["Your level (%)", "Required level (%)"],
        horizontal=True,
        stack=False,
        sort=False,
        color=["#16a36a", "#6366f1"],
        height=max(280, len(chart_rows) * 42),
        width="stretch",
    )
    st.caption("Bars map the Basic → Intermediate → Advanced/Strong scale to 33% → 67% → 100%; they are level markers, not measured proficiency scores.")

    strong = [row["skill"] for row in rows if row["ratio"] >= 1][:4]
    developing = [row["skill"] for row in rows if 0 < row["ratio"] < 1]
    absent = [row["skill"] for row in rows if row["candidate_rank"] == 0]
    st.markdown("**A quick read:**")
    st.markdown(f"- You're building a solid foundation for **{role}**, with an estimated {readiness}% skill-level alignment.")
    if strong:
        st.markdown(f"- **Already aligned:** {', '.join(strong)}.")
    if developing:
        st.markdown(f"- **Worth strengthening:** {', '.join(developing[:4])}.")
    if absent:
        st.markdown(f"- **Not yet evidenced in your resume:** {', '.join(absent[:4])}.")
    next_skills = [row["skill"] for row in sorted(rows, key=lambda row: -row["gap_share"]) if row["gap_share"] > 0][:2]
    if next_skills:
        st.markdown(f"- I'd start with **{', '.join(next_skills)}** because they account for the largest calculated gaps in this comparison.")

    st.subheader("🚀 How far are you from your target career?")
    if remaining:
        biggest = [row for row in sorted(rows, key=lambda row: -row["gap_share"]) if row["gap_share"] > 0][:4]
        st.markdown(f"The remaining {remaining}% is driven most by:")
        for row in biggest:
            points = row["gap_share"]
            shown = f"{points:.1f}".rstrip("0").rstrip(".")
            st.markdown(f"- **{row['skill']}** — about {shown} percentage points of the calculated gap ({row['candidate_level']} vs. {row['required_level']}).")
        st.caption("Gap contributions use the same level and priority formula above; displayed values are rounded.")
    else:
        st.markdown("Your current levels meet or exceed the listed requirements. Keep your portfolio and interview examples current.")


def _render_career_flow(role, analysis_ready):
    """Show the app's career-planning sequence as responsive Streamlit cards."""
    st.subheader("🧭 Your Career Journey")
    st.caption("How the app turns your resume into a practical next-step plan.")
    rows = [
        [
            ("📄 Resume", "Uploaded and parsed"),
            ("🧠 Skills extracted", "Resume evidence"),
            ("🎯 Target career", role),
        ],
        [
            ("📈 Job readiness", "Calculated from skill levels" if analysis_ready else "After comparison"),
            ("📊 Skill comparison", "Your level vs. role requirement" if analysis_ready else "Run skill-gap analysis"),
            ("🔎 Required skills", "Role baseline or job document"),
        ],
        [
            ("⚠️ Skill gaps", "Missing and developing skills" if analysis_ready else "After comparison"),
            ("💡 Recommendations", "Career preparation and next steps"),
            ("🗺️ Learning roadmap", "A focused learning sequence"),
        ],
    ]
    for row_index, steps in enumerate(rows):
        reverse = row_index == 1
        grid = st.columns([1, 0.12, 1, 0.12, 1])
        display_steps = steps
        for slot, (title, detail) in zip((0, 2, 4), display_steps):
            with grid[slot].container(border=True):
                st.markdown(f"**{title}**")
                st.caption(detail)
        arrow = "←" if reverse else "→"
        with grid[1]:
            st.markdown(f"<div style='text-align:center;padding-top:1.1rem;font-size:1.35rem'>{arrow}</div>", unsafe_allow_html=True)
        with grid[3]:
            st.markdown(f"<div style='text-align:center;padding-top:1.1rem;font-size:1.35rem'>{arrow}</div>", unsafe_allow_html=True)
        if row_index < 2:
            end_slot = 0 if reverse else 4
            down = st.columns([1, 0.12, 1, 0.12, 1])
            with down[end_slot]:
                st.markdown("<div style='text-align:center;font-size:1.25rem'>↓</div>", unsafe_allow_html=True)


def _render_gap_entries(entries, empty_message, show_status=False):
    if not entries:
        st.caption(empty_message)
        return
    for entry in entries:
        skill = _plain_text(entry.get("skill", "Skill"))
        explanation = _plain_text(entry.get("explanation", ""))
        status = _plain_text(entry.get("status", ""))
        icon = "✅" if status.lower() == "available" else "⚠️" if status.lower() == "missing" else "📈"
        label = f"{icon} **{skill}**"
        if show_status and status:
            label += f" — {status}"
        if explanation:
            label += f": {explanation}"
        st.markdown(f"- {label}")


def rows_as_skill_entries(rows):
    results = []
    for row in rows:
        if isinstance(row, dict):
            results.append({
                "skill": _plain_text(_find_value(row, ("skill", "name", "requirement", "technology"), fallback="Unnamed skill")),
                "status": _plain_text(_find_value(row, ("status", "match_status", "level"), fallback="Required")),
                "explanation": _plain_text(_find_value(row, ("explanation", "evidence", "details", "reason"), fallback="")),
            })
        else:
            results.append({"skill": _plain_text(row), "status": "Required", "explanation": ""})
    return results


def _render_recommendations(value):
    items = _as_items(value, ("recommendations", "roles", "items"))
    if not items:
        st.caption("No role recommendations yet. Generate them to see possible next steps.")
        return
    for index, item in enumerate(items, start=1):
        if isinstance(item, dict):
            role = _plain_text(_find_value(item, ("role", "title", "job_title", "name"), fallback=f"Recommended role {index}"))
            explanation = _concise_advice(_find_value(item, ("explanation", "rationale", "reason", "summary", "fit"), fallback=None))
            matching = _find_value(item, ("matching_skills", "strengths"), fallback=None)
            gaps = _find_value(item, ("gaps", "skill_gaps", "development_areas"), fallback=None)
            with st.container(border=True):
                st.markdown(f"**{role}**")
                if explanation:
                    st.markdown(_plain_text(explanation))
                if matching:
                    st.markdown(f"- **Relevant strengths:** {_concise_advice(matching)}")
                if gaps:
                    st.markdown(f"- **Areas to develop:** {_concise_advice(gaps)}")
        else:
            text = _concise_advice(item)
            if text:
                st.markdown(f"- {text}")


def _clean_advice_text(value):
    """Remove repeated recommendation labels and normalize concise advice text."""
    text = _plain_text(value)
    text = re.sub(r"(?i)^(?:\s*recommended\s*:\s*)+", "", text)
    text = re.sub(r"(?i)\b(?:recommended\s*:\s*){2,}", "", text)
    text = re.sub(r"(?i)\b(\w+(?:[’'-]\w+)?)\s+\1\b", r"\1", text)
    text = re.sub(r"^\s*[-•]\s+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _concise_advice(value, max_sentences=2):
    """Remove duplicate sentences and keep generated advice brief."""
    text = _clean_advice_text(value)
    sentences, seen = [], set()
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        sentence = sentence.strip()
        key = re.sub(r"[^a-z0-9]+", "", re.sub(r"[\[\]]", "", sentence.casefold()))
        if sentence and key not in seen:
            seen.add(key)
            sentences.append(sentence)
        if len(sentences) >= max_sentences:
            break
    return " ".join(sentences)


def _unique_advice(items):
    """Clean an advice list and drop exact repeated items while preserving order."""
    cleaned, seen = [], set()
    for item in items:
        text = _clean_advice_text(item)
        normalized = re.sub(r"[^a-z0-9]+", "", re.sub(r"[\[\]]", "", text.casefold()))
        if text and normalized not in seen:
            seen.add(normalized)
            cleaned.append(text)
    return cleaned


def _career_preparation(profile, target_role, gaps):
    """Build clearly labeled preparation advice without inferring personal facts."""
    target = target_role.lower()
    target_skills = {
        "data analyst": ["intermediate SQL", "spreadsheet analysis", "data visualization", "business storytelling"],
        "data scientist": ["statistics", "Python data analysis", "model evaluation", "experimentation"],
        "ml engineer": ["Python engineering", "model deployment", "testing", "ML pipelines"],
        "ai engineer": ["Python", "LLM application patterns", "RAG evaluation", "API and deployment basics"],
        "python developer": ["Python fundamentals", "API development", "automated testing", "SQL and Git"],
        "business analyst": ["requirements gathering", "process mapping", "SQL or spreadsheets", "KPI communication"],
    }
    company_types = {
        "data analyst": "Explore junior analyst teams in SaaS, retail, fintech, healthcare, consulting, or small and mid-sized businesses.",
        "data scientist": "Look at junior analytics teams in technology, fintech, healthcare, retail, and consulting; internships can add applied experience.",
        "ml engineer": "Explore product teams deploying models, AI/ML startups, and data-platform teams with junior model-serving or pipeline work.",
        "ai engineer": "Consider AI product teams, SaaS companies adding AI features, application startups, and automation consultancies.",
        "python developer": "Look at SaaS companies, software consultancies, product startups, and internal automation teams hiring junior developers.",
        "business analyst": "Explore consulting, product, operations, financial services, healthcare, and business transformation teams hiring junior analysts.",
    }
    portfolio_ideas = {
        "data analyst": "Build an end-to-end analysis from a public dataset: answer a business question with SQL, clean the data, create a dashboard, and summarize the decision it supports.",
        "data scientist": "Build a reproducible project with a clear question, baseline model, sound validation, useful metrics, and a plain-language note on limitations.",
        "ml engineer": "Package a small model behind a tested API, document setup, and show a repeatable training or inference workflow.",
        "ai engineer": "Build a small grounded assistant or AI feature, then document its evaluation examples, sources, and limitations.",
        "python developer": "Publish a small Python service or automation tool with tests, sample inputs, and a clear README.",
        "business analyst": "Write a process-improvement case study with a process map, measurable KPIs, analysis, and a stakeholder-ready conclusion.",
    }
    interview_prep = {
        "data analyst": "Practice SQL joins, grouping, and window functions. Be ready to explain chart choices, data checks, and the business decision your analysis supports.",
        "data scientist": "Review statistics and experimentation, model trade-offs, metric interpretation, and how you communicate uncertainty.",
        "ml engineer": "Practice Python coding, model serving, testing, common failure cases, and deployment trade-offs.",
        "ai engineer": "Explain an AI system end to end: grounding, evaluation, privacy limits, latency, and failure handling.",
        "python developer": "Practice Python problem solving, APIs, databases, debugging, tests, and explaining design choices.",
        "business analyst": "Rehearse requirements discovery, process analysis, KPI selection, prioritization, and stakeholder communication.",
    }
    category_skills = {
        "Technology": ["role-specific tools", "testing and quality", "Git and practical project work"],
        "Finance": ["financial analysis", "Excel modeling", "forecasting"],
        "Marketing": ["campaign analytics", "market research", "content or channel strategy"],
        "Business": ["requirements analysis", "process improvement", "stakeholder communication"],
        "HR": ["HR analytics", "recruiting workflows", "HR systems and reporting"],
        "Sales": ["CRM workflows", "pipeline analysis", "prospecting and negotiation"],
        "Design / Creative": ["user or audience research", "design tools", "portfolio case studies"],
        "Supply Chain / Operations": ["operations analysis", "forecasting", "ERP or inventory workflows"],
        "Healthcare": ["healthcare workflows", "health data privacy", "health informatics tools"],
    }
    role_key = target if target in target_skills else None
    category = next(
        (name for name, choices in CAREER_CATEGORIES.items() if target in {item.casefold() for item in choices}),
        None,
    )

    gap_rows = rows_as_skill_entries(_as_items(gaps, ("skill_gaps", "skills", "items")))
    priority = []
    for row in gap_rows:
        status_key = _normal_key(row["status"])
        if any(label in status_key for label in ("missing", "improv", "partial", "developing", "intermediate")):
            skill = row["skill"]
            if skill and skill not in priority:
                priority.append(skill)
    if not priority:
        priority = target_skills[role_key] if role_key else category_skills.get(category, ["role-specific tools", "project evidence", "interview examples"])

    existing_projects = _as_items(_find_value(profile, ("projects", "portfolio_projects")), ("items", "records", "projects"))
    project_names = [
        _plain_text(_find_value(item, ("project_name", "name", "title"), fallback=""))
        for item in existing_projects if isinstance(item, dict)
    ]
    project_names = [name for name in project_names if name]
    portfolio = [f"**Portfolio project:** {portfolio_ideas[role_key] if role_key else f'Build a small {target_role} case study that shows the problem, your approach, tools, and result.'}"]
    if project_names:
        portfolio.append(f"**Build on your experience:** Polish {project_names[0]} with a clear problem statement, your contribution, evidence of results, and a recruiter-friendly README or case study.")

    strengths = _as_items(_find_value(profile, ("technical_skills", "skills", "hard_skills")), ("technical_skills", "skills", "items"))
    strength_names = [_plain_text(skill) for skill in strengths if _plain_text(skill)]
    skill_advice = [f"**Start with {', '.join(priority[:4])}.** These are the main gaps to address for {target_role}."]
    if strength_names:
        skill_advice.append(f"**Build on your strengths:** Use {', '.join(strength_names[:4])} in projects and interview examples.")

    generic_company = f"Explore junior or associate roles in {category.lower()} teams. Compare postings to see which tools and experience employers expect." if category else "Explore junior roles matching your target title and compare postings to see which skills employers expect."
    generic_interview = "Explain one project: the problem, your approach, the result, and what you would improve next. Connect the example to the role."
    return [
        ("Job-search timeline", ["Use a focused 4–8 week preparation sprint. Start applying when your resume is tailored and you have one project to show; keep learning while you apply. Adjust the pace to your schedule." ]),
        ("Company types to explore", [company_types[role_key] if role_key else generic_company]),
        ("Skills to prioritize", skill_advice),
        ("Portfolio projects", portfolio),
        ("Interview preparation", [f"**Role practice:** {interview_prep[role_key] if role_key else generic_interview}", "**STAR examples:** Prepare two short stories about a problem, your actions, and the result, using only your real project or work experience."]),
        ("LinkedIn and GitHub", ["**LinkedIn:** Align your headline and About section with your target role, and highlight measurable project outcomes.", "**GitHub:** Pin two or three polished projects with a clear README, setup steps, and results. Remove credentials and API keys before publishing."]),
    ]


def _render_career_preparation(profile, target_role, gaps):
    st.header("✅ 8. Recommended Career Preparation")
    st.caption("A few practical ideas based on your resume, target role, and current skill gaps. Adapt the timing and priorities to fit your plans.")
    seen = set()
    for title, recommendations in _career_preparation(profile, target_role, gaps):
        st.subheader(title)
        cleaned = _unique_advice(recommendations)
        fresh = []
        for item in cleaned:
            key = re.sub(r"[^a-z0-9]+", "", re.sub(r"[\[\]]", "", item.casefold()))
            if key not in seen:
                seen.add(key)
                fresh.append(item)
        _render_bullets(fresh)


def _clean_roadmap_text(value):
    """Keep roadmap Markdown readable while removing repeated labels and lines."""
    rendered, seen = [], set()
    for line in _plain_text(value).splitlines():
        match = re.match(r"^(\s*(?:#{1,6}\s*|[-*•]\s+|\d+[.)]\s+)?)(.*)$", line)
        prefix, content = match.groups() if match else ("", line)
        is_heading = prefix.lstrip().startswith("#")
        content = _clean_advice_text(content) if is_heading else _concise_advice(content)
        if not content:
            continue
        key = re.sub(r"[^a-z0-9]+", "", re.sub(r"[\[\]]", "", content.casefold()))
        if key in seen:
            continue
        seen.add(key)
        rendered.append(f"{prefix}{content}")
    return "\n".join(rendered)


def _render_roadmap(value):
    roadmap = value
    if isinstance(roadmap, str):
        candidate = roadmap.strip()
        if candidate.startswith(("{", "[")):
            try:
                roadmap = json.loads(candidate)
            except json.JSONDecodeError:
                pass
    if isinstance(roadmap, (dict, list)):
        _render_roadmap_records(roadmap)
        return

    text = _clean_roadmap_text(roadmap)
    if not text:
        st.caption("A learning roadmap will appear after skill gap analysis.")
        return
    lines = text.splitlines()
    sections = []
    current_heading = None
    current_lines = []
    for line in lines:
        clean = line.strip()
        heading_text = clean.lstrip("#* ").strip()
        is_period_heading = bool(re.match(r"(?i)^(months?|weeks?)\s*(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b", heading_text))
        is_numbered_step = bool(re.match(r"^\d+[.)]\s+", heading_text)) and any(word in heading_text.lower() for word in ("month", "week", "phase"))
        if is_period_heading or is_numbered_step:
            if current_heading is not None or current_lines:
                sections.append((current_heading, current_lines))
            current_heading, current_lines = heading_text, []
        else:
            current_lines.append(line)
    if current_heading is not None or current_lines:
        sections.append((current_heading, current_lines))

    has_period_sections = any(heading for heading, _ in sections)
    if not has_period_sections:
        st.markdown(text)
        return
    for heading, body in sections:
        if heading:
            with st.container(border=True):
                st.subheader(heading)
                body_text = "\n".join(body).strip()
                if body_text:
                    st.markdown(body_text)
                else:
                    st.caption("Learning activities and deliverable")
        elif body:
            st.markdown("\n".join(body))


def _render_roadmap_records(value):
    if isinstance(value, dict):
        period_keys = ("months", "weeks", "milestones", "roadmap", "learning_roadmap", "plan", "steps", "items")
        periods = _find_value(value, period_keys, fallback=None)
        if periods is not None and periods is not value:
            _render_roadmap_records(periods)
            return
        value = [value]
    for index, period in enumerate(value, start=1):
        if isinstance(period, dict):
            period_label = _clean_advice_text(_find_value(period, ("month", "week", "period", "phase", "timeline", "title"), fallback=f"Step {index}"))
            skill_focus = _clean_advice_text(_find_value(period, ("skill", "skill_focus", "focus", "topic"), fallback=""))
            heading = period_label
            if skill_focus and skill_focus.casefold() not in period_label.casefold():
                heading = f"{period_label}: {skill_focus}"
            with st.container(border=True):
                st.subheader(heading)
                for key, detail in period.items():
                    if _normal_key(key) in {"month", "week", "period", "phase", "timeline", "title", "skill", "skillfocus", "focus", "topic"}:
                        continue
                    label = key.replace("_", " ").capitalize()
                    details = _unique_advice(_as_items(detail, ("items", "topics", "activities")))
                    details = [_concise_advice(item) for item in details]
                    details = [item for item in details if item]
                    st.markdown(f"**{label}**")
                    _render_bullets(details)
        else:
            text = _concise_advice(period)
            if text:
                st.markdown(f"- {text}")


def clear_session():
    for key in list(st.session_state.keys()):
        del st.session_state[key]


def _queue_chat_question():
    question = str(st.session_state.get("chat_input", "")).strip()
    if question:
        st.session_state["_career_chat_pending_question"] = question
        st.session_state["chat_input"] = ""


settings = load_settings()
agent = get_agent(settings)

# Browser components and Session State can outlive a Streamlit script restart.
# Ignore component events restored from an older app process so old text/audio
# is never replayed when the user next interacts with the page.
app_instance_id = _app_instance_id()
if st.session_state.get("_career_chat_event_guard_instance") != app_instance_id:
    st.session_state["_career_chat_is_recording"] = False
    st.session_state["_career_chat_is_transcribing"] = False
    st.session_state["_career_chat_audio_input_seen"] = False
    st.session_state["_career_chat_processed_audio_id"] = ""
    for stale_key in (
        "_career_chat_transcription_future",
        "_career_chat_transcription_audio_id",
        "_career_chat_transcription_started_at",
        "_career_chat_pending_question",
        "is_recording",
        "is_transcribing",
        "transcript_pending",
    ):
        st.session_state.pop(stale_key, None)
    st.session_state["_career_chat_event_guard_instance"] = app_instance_id

# Recording lives in the browser component, so a fresh Python session is idle.
# Clear legacy activity flags restored from earlier voice implementations.
st.session_state["_career_chat_is_recording"] = False
st.session_state["_career_chat_is_transcribing"] = False
st.session_state.pop("is_recording", None)
st.session_state.pop("is_transcribing", None)
st.session_state.pop("transcript_pending", None)
st.session_state.pop("_career_chat_transcription_future", None)
st.session_state.pop("_career_chat_transcription_audio_id", None)
st.session_state.pop("_career_chat_transcription_started_at", None)

with st.sidebar:
    st.title("Career Advisor")
    st.caption("Resume analysis · role fit · learning plan")
    uploaded = st.file_uploader("Upload Resume", type=["pdf", "docx", "txt"], help="PDF, DOCX, or plain text")
    role_category = st.selectbox("Career category", list(CAREER_CATEGORIES), key="career_category")
    role_pick = st.selectbox(
        "Select Target Job Role",
        CAREER_CATEGORIES[role_category],
        key=f"target_role_{_normal_key(role_category)}",
    )
    other_role = st.text_input("Other role (optional)", placeholder="e.g. Product Analyst")
    target_role = other_role.strip() or role_pick
    if agent:
        st.success(f"Gemini configured · {settings['model']}")
    else:
        st.warning("Gemini API key is not configured.")
        st.caption("Add GEMINI_API_KEY to the .env file.")
    if st.button("Clear session", use_container_width=True):
        clear_session()
        st.rerun()

st.title("🤖 AI Career Advisor Agent")
st.caption("Resume intelligence · skill gap analysis · career planning")
st.markdown("Upload your resume and discover a practical path toward your target career.")

if uploaded:
    st.success("Resume uploaded successfully ✓")
    if "resume_text" not in st.session_state or st.session_state.get("resume_name") != uploaded.name:
        try:
            st.session_state.resume_text = parse_resume(uploaded.name, uploaded.getvalue())
            st.session_state.resume_name = uploaded.name
            st.session_state.pop("profile", None)
            st.session_state.pop("target_analysis", None)
            st.session_state.pop("recommendations", None)
            st.session_state.pop("chat_history", None)
        except Exception as exc:
            st.error(safe_error_message(exc))
            st.stop()
    st.caption(f"{uploaded.name} · {len(st.session_state.resume_text):,} characters extracted")

if st.button("Analyze Resume", type="primary", disabled=not uploaded or agent is None):
    if not uploaded:
        st.error("Upload a resume before analysis.")
    elif not agent:
        st.error("Gemini API key is not configured. Add GEMINI_API_KEY to the .env file.")
    else:
        try:
            with st.spinner("Reading your resume and preparing your profile…"):
                st.session_state.profile = agent.analyze_resume(st.session_state.resume_text)
                st.session_state.target_analysis = None
                st.session_state.recommendations = None
                st.session_state.chat_history = []
        except Exception as exc:
            st.error(safe_error_message(exc))

profile = st.session_state.get("profile")
if profile:
    st.subheader("Your resume snapshot")
    summary = get_profile_section(profile, "overall_summary", "Profile extracted from the uploaded resume.")
    if isinstance(summary, (dict, list)):
        summary = _plain_text(summary)
    st.info(summary)
    _render_profile(profile)

    st.header("🎯 5. Target career")
    st.markdown(f"Selected role: **{target_role}**")
    if st.button("Run skill gap analysis", disabled=agent is None):
        st.session_state.target_analysis = None
        current_stage = {"label": "1. Reading your profile..."}
        try:
            with st.status(current_stage["label"], expanded=True) as progress_status:
                def report_progress(message):
                    current_stage["label"] = message
                    progress_status.update(label=message, state="running")

                st.session_state.target_analysis = agent.analyze_target(
                    profile,
                    target_role,
                    progress_callback=report_progress,
                    resume_text=st.session_state.get("resume_text", ""),
                )
                progress_status.update(label="5. Analysis complete!", state="complete", expanded=True)
        except Exception as exc:
            try:
                st.session_state.target_analysis = agent.deterministic_target_analysis(
                    profile if isinstance(profile, dict) else {},
                    target_role,
                    st.session_state.get("resume_text", ""),
                )
                progress_status.update(label="5. Analysis complete!", state="complete", expanded=True)
            except Exception:
                progress_status.update(label=f"Analysis stopped during: {current_stage['label']}", state="error", expanded=True)
                st.error(f"Could not finish skill-gap analysis. {safe_error_message(exc)}")

    result = st.session_state.get("target_analysis")
    if result and result.get("role") not in (None, target_role):
        result = None
        st.info("The selected role changed. Run skill gap analysis to refresh these results.")
    if result:
        st.header("🔎 6. Skill gap analysis")
        if result.get("fallback_used"):
            st.warning(result.get("fallback_message") or "Gemini was unavailable; a local skills comparison and roadmap are shown.")
        _render_skill_gaps(result, target_role)
        with st.expander("Role requirements used", icon=":material/library_books:"):
            st.caption(result.get("context_source", "Retrieved role context"))
            st.markdown(_plain_text(_find_value(result, ("context", "sources", "requirements"), fallback="No role context retrieved.")))

    _render_career_flow(target_role, bool(result))

    st.header("📈 7. Career recommendations")
    if st.button("Recommend suitable roles", disabled=agent is None):
        try:
            with st.spinner("Matching your profile with role knowledge…"):
                st.session_state.recommendations = agent.recommend_roles(profile, ROLES)
        except Exception as exc:
            st.error(safe_error_message(exc))
    _render_recommendations(st.session_state.get("recommendations"))

    gap_evidence = _find_value(result, ("gaps", "skill_gaps", "skills"), fallback=[]) if result else []
    _render_career_preparation(profile, target_role, gap_evidence)

    st.header("📅 9. Personalized learning roadmap")
    if result:
        _render_roadmap(_find_value(result, ("roadmap", "learning_roadmap", "learning_plan"), fallback=""))
    else:
        st.caption("Run skill gap analysis to generate a personalized roadmap.")

    st.header("💬 10. AI career chat")
    st.caption("Type a question or choose the microphone. You can edit any transcript before sending.")
    voice_state = st.session_state.get("career_chat_voice", {})
    current_voice_settings = voice_settings(voice_state)
    with st.expander("Voice and speech settings", expanded=False):
        voice_toolbar(
            data={"settings": current_voice_settings},
            default=voice_defaults(),
            key="career_chat_voice",
            on_settings_change=lambda: None,
            on_transcript_event_change=lambda: None,
            height=210,
        )

    history = st.session_state.setdefault("chat_history", [])
    for message_index, message in enumerate(history):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                response_id = message.get("id") or f"response-{message_index}"
                state_key = f"career_chat_audio_{response_id}"
                voice_player(
                    data={"speech_text": clean_text_for_speech(message["content"]), "response_id": response_id, "settings": current_voice_settings},
                    key=state_key,
                    height=52,
                )
    question = st.session_state.pop("_career_chat_pending_question", "")
    if question:
        history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        try:
            resume_text = st.session_state.get("resume_text", "")
            if not isinstance(resume_text, str) or not resume_text.strip():
                raise ValueError("The uploaded resume text is unavailable. Re-upload the resume to continue career chat.")
            career_context = {
                "skill_gap": st.session_state.get("target_analysis") or {},
                "recommendations": st.session_state.get("recommendations") or [],
            }
            with st.chat_message("assistant"):
                answer = st.write_stream(
                    agent.chat_stream(
                        profile,
                        target_role,
                        question,
                        history[:-1],
                        career_context,
                        resume_text=resume_text,
                    )
                )
                response_id = uuid4().hex
                voice_player(
                    data={"speech_text": clean_text_for_speech(answer), "response_id": response_id, "settings": current_voice_settings},
                    key=f"career_chat_audio_{response_id}",
                    height=52,
                )
            history.append({"role": "assistant", "content": answer, "id": response_id})
            # Retain ten recent exchanges for follow-up resolution without growing prompts indefinitely.
            if len(history) > 20:
                del history[:-20]
        except Exception as exc:
            history.pop()
            st.session_state["_career_chat_error"] = safe_error_message(exc)

    input_column, microphone_column = st.columns([0.87, 0.13], vertical_alignment="bottom")
    with microphone_column:
        recording = st.audio_input(
            "Microphone",
            sample_rate=16000,
            key="career_chat_microphone",
            label_visibility="collapsed",
            help="Record a question to transcribe it into the chat input.",
        )

    recording_bytes = recording.getvalue() if recording is not None else b""
    recording_id = ""
    if recording_bytes:
        recording_id = getattr(recording, "file_id", None) or hashlib.sha256(recording_bytes).hexdigest()
    if not st.session_state.get("_career_chat_audio_input_seen"):
        # Ignore any browser-restored recording on first render; only a new
        # recording made after the page is ready may start transcription.
        st.session_state["_career_chat_audio_input_seen"] = True
        st.session_state["_career_chat_processed_audio_id"] = recording_id
    elif recording_id and recording_id != st.session_state.get("_career_chat_processed_audio_id", ""):
        st.session_state["_career_chat_processed_audio_id"] = recording_id
        if not question:
            st.session_state["_career_chat_is_recording"] = False
            st.session_state["_career_chat_is_transcribing"] = True
            transcription_future = None
            try:
                transcription_future = transcribe_audio_async(
                    recording_bytes,
                    getattr(recording, "type", "audio/wav") or "audio/wav",
                )
                transcript = (transcription_future.result(timeout=60) or "").strip()
                if transcript:
                    st.session_state["chat_input"] = transcript
                else:
                    raise ValueError("No speech was recognized.")
                st.session_state["_career_chat_voice_error"] = ""
            except Exception:
                st.session_state["_career_chat_voice_error"] = (
                    "No speech could be transcribed. Your input is still available; please try again or type your question."
                )
            finally:
                if transcription_future is not None and not transcription_future.done():
                    transcription_future.cancel()
                st.session_state["_career_chat_is_transcribing"] = False
                st.session_state["_career_chat_is_recording"] = False

    with input_column:
        st.text_area(
            "Ask your career question...",
            key="chat_input",
            placeholder="Ask your career question...",
            label_visibility="collapsed",
            height=84,
        )
        st.button(
            "Send",
            key="career_chat_send",
            type="primary",
            on_click=_queue_chat_question,
            disabled=not str(st.session_state.get("chat_input", "")).strip(),
            width="stretch",
        )
    if st.session_state.get("_career_chat_voice_error"):
        st.warning(st.session_state.pop("_career_chat_voice_error"))
    if st.session_state.get("_career_chat_error"):
        st.error(st.session_state.pop("_career_chat_error"))
elif uploaded:
    st.info("Select a target role, then choose Analyze Resume to start.")
else:
    st.markdown("### Start with your resume")
    st.write("Upload a PDF, DOCX, or TXT resume in the sidebar. Analysis uses your resume text, retrieved role knowledge, and the configured LLM.")

st.caption("Your uploaded resume is processed in the current app session and is not written to disk by this application.")
