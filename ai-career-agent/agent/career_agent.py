"""Orchestrates focused career tools and grounded RAG context."""
import re

from agent import tools
from agent.career_catalog import local_role_context
from agent.prompts import CHAT_PROMPT
from rag.document_loader import load_role_documents
from rag.rag_pipeline import retrieve_job_requirements


CHAT_SYSTEM_INSTRUCTION = """You are a friendly human career mentor. Answer the user's question directly and naturally. Use the candidate's resume and previous conversation when relevant. Keep responses concise and easy to scan, usually 40–120 words unless the user asks for detail. Use short paragraphs and short bullet points when multiple items are involved. Do not repeat greetings, use robotic or report-like phrases, add unnecessary disclaimers, or write huge paragraphs. Be practical, personalized, encouraging, and honest. Never invent facts that are absent from the resume or conversation."""


class CareerAgent:
    def __init__(self, client, retriever):
        self.client, self.retriever = client, retriever
        self._role_context_cache = {}

    def _role_context(self, role: str) -> str:
        """Retrieve role requirements once; use the matching source document if RAG is incomplete."""
        if role not in self._role_context_cache:
            try:
                context = retrieve_job_requirements(self.retriever, role)
            except Exception:
                context = ""
            if not tools.has_structured_role_requirements(context):
                context = self._structured_role_context(role) or context
            if not context.strip():
                raise RuntimeError(f"No role requirements are available for {role} from RAG or the local structured role documents.")
            self._role_context_cache[role] = context
        return self._role_context_cache[role]

    def _structured_role_context(self, role: str) -> str:
        """Read the selected role's existing local knowledge file as a RAG fallback."""
        docs_dir = getattr(self.retriever, "docs_dir", None)
        if not docs_dir:
            return ""
        aliases = {"ml engineer": "ml_engineer", "machine learning engineer": "ml_engineer"}
        wanted = aliases.get(role.casefold(), role.casefold().replace(" ", "_"))
        try:
            documents = load_role_documents(docs_dir)
        except Exception:
            return ""
        for document in documents:
            source = document["source"].rsplit(".", 1)[0].casefold()
            title = document["text"].splitlines()[0].strip().casefold()
            if source == wanted or title == role.casefold():
                return document["text"]
        return ""

    def analyze_resume(self, resume_text: str) -> dict:
        return tools.analyze_resume(self.client, resume_text)

    def deterministic_target_analysis(self, profile: dict, role: str, resume_text: str = "", context: str = "") -> dict:
        """Return a local skill comparison and roadmap without making Gemini calls."""
        context_source = "Local role baseline"
        if role.strip().casefold() == "data analyst":
            context = tools.data_analyst_fallback_context()
            structured = tools.calculate_data_analyst_fallback(profile or {}, resume_text)
        else:
            context = context or self._structured_role_context(role)
            if context:
                context_source = "Local job-role document"
            else:
                context = local_role_context(role)
            if not context:
                raise RuntimeError(f"Local requirements are unavailable for {role}.")
            structured = tools.calculate_skill_gap_fallback(profile or {}, context, resume_text)
        return {
            **structured,
            "role": role,
            "gaps": structured["skill_gaps"],
            "context": context,
            "roadmap": _fallback_roadmap(role, structured),
            "context_source": context_source,
            "fallback_used": False,
            "fallback_message": "",
        }

    def analyze_target(self, profile: dict, role: str, progress_callback=None, resume_text: str = "") -> dict:
        """Run skill-gap analysis locally; this path must not depend on Gemini."""
        def progress(message):
            if progress_callback:
                progress_callback(message)

        progress("1. Reading your profile...")
        if not isinstance(profile, dict) or not profile:
            raise ValueError("Analyze a resume before running skill-gap analysis.")
        progress(f"2. Comparing resume skills with {role} requirements locally...")
        result = self.deterministic_target_analysis(profile, role, resume_text)
        progress("5. Analysis complete!")
        return result

    def recommend_roles(self, profile: dict, roles: list[str]) -> list:
        results = []
        for role in roles:
            context = self._role_context(role)
            results.extend(tools.recommend_roles(self.client, profile, context))
        return results

    def chat_stream(
        self,
        profile: dict,
        role: str,
        question: str,
        history: list[dict],
        career_context: dict | None = None,
        resume_text: str = "",
    ):
        """Yield a concise mentor answer with compact resume and session memory."""
        context = _chat_role_context(role, question, career_context or {})
        compact_history = _compact_history(history)
        prompt = CHAT_PROMPT.format(
            resume_evidence=resume_text.strip() or _compact_resume_context(profile),
            context=context[:2400],
            career_notes=_compact_career_notes(career_context or {}),
            history=compact_history,
            greeting_guidance=_greeting_guidance(question, compact_history),
            question=question.strip(),
        )
        if hasattr(self.client, "generate_response_stream"):
            yield from self.client.generate_response_stream(
                prompt,
                temperature=0.35,
                system_instruction=CHAT_SYSTEM_INSTRUCTION,
            )
        else:
            # Keep compatible clients on the same instruction path if they do
            # not expose the streaming request method.
            yield self.client.generate_response(
                f"{CHAT_SYSTEM_INSTRUCTION}\n\n{prompt}",
                temperature=0.35,
            )

    def chat(
        self,
        profile: dict,
        role: str,
        question: str,
        history: list[dict],
        career_context: dict | None = None,
        resume_text: str = "",
    ) -> str:
        """Compatibility helper returning the streamed response as one string."""
        return "".join(self.chat_stream(profile, role, question, history, career_context, resume_text))


def _compact_resume_context(profile: dict) -> str:
    """Keep only career-relevant facts in repeated chat prompts; omit contact data."""
    profile = profile if isinstance(profile, dict) else {}
    lines = []
    summary = _short_text(profile.get("overall_summary"), 360)
    if summary:
        lines.append(f"Resume summary: {summary}")
    skills = list(dict.fromkeys(_profile_skills(profile)))[:16]
    if skills:
        lines.append("Skills: " + ", ".join(skills))
    for field, label in (("work_experience", "Experience"), ("projects", "Projects")):
        entries = profile.get(field, [])
        if isinstance(entries, dict):
            entries = [entries]
        if not isinstance(entries, list):
            continue
        for item in entries[:3]:
            if isinstance(item, dict):
                title = _short_text(item.get("title") or item.get("name") or item.get("project_name"), 90)
                detail = _short_text(item.get("summary") or item.get("description") or item.get("details"), 170)
                company = _short_text(item.get("company"), 60)
                heading = " — ".join(part for part in (title, company) if part)
                value = ": ".join(part for part in (heading, detail) if part)
            else:
                value = _short_text(item, 170)
            if value:
                lines.append(f"{label}: {value}")
    education = profile.get("education", [])
    if isinstance(education, dict):
        education = [education]
    if isinstance(education, list):
        degrees = []
        for item in education[:2]:
            value = item.get("degree") or item.get("qualification") or item.get("program") if isinstance(item, dict) else item
            value = _short_text(value, 100)
            if value:
                degrees.append(value)
        if degrees:
            lines.append("Education: " + "; ".join(degrees))
    return "\n".join(lines) or "No concise resume facts are available."


def _compact_career_notes(context: dict) -> str:
    analysis = context.get("skill_gap") or {}
    recommendations = context.get("recommendations") or []
    lines = []
    if isinstance(analysis, dict):
        for key, label in (("matching_skills", "Strengths"), ("skills_to_improve", "Improve"), ("missing_skills", "Gaps")):
            names = []
            for item in _items(analysis.get(key))[:7]:
                entry = _entry(item, "")
                if entry:
                    names.append(entry["skill"])
            if names:
                lines.append(f"{label}: " + ", ".join(names))
        for item in _items(analysis.get("recommendations"))[:3]:
            text = _short_text(_recommendation_text(item), 180)
            if text:
                lines.append(f"Prior recommendation: {text}")
    if isinstance(recommendations, list):
        for item in recommendations[:3]:
            text = _short_text(_recommendation_text(item), 180)
            if text:
                lines.append(f"Role recommendation: {text}")
    return "\n".join(lines) or "No skill-gap analysis has been run yet."


def _compact_history(history: list[dict]) -> str:
    rows = []
    for item in (history or [])[-8:]:
        if not isinstance(item, dict) or item.get("role") not in ("user", "assistant"):
            continue
        content = _short_text(item.get("content"), 360)
        if content:
            label = "You" if item["role"] == "user" else "Advisor"
            rows.append(f"{label}: {content}")
    return "\n".join(rows) or "No earlier messages."


def _chat_role_context(role: str, question: str, career_context: dict) -> str:
    """Use only already-computed role context; never block a chat send on retrieval."""
    if not _needs_role_retrieval(question):
        return "No role retrieval needed for this question; use the resume and recent conversation."

    analysis = career_context.get("skill_gap") or {}
    cached_context = career_context.get("retrieved_context") or (
        analysis.get("context", "") if isinstance(analysis, dict) else ""
    )
    if not isinstance(cached_context, str) or not cached_context.strip():
        return "No cached role requirements are available; answer from the resume and conversation without guessing."

    stop_words = {"what", "which", "are", "the", "for", "with", "that", "this", "from", "have", "need", "skills", "required"}
    terms = {
        term for term in re.findall(r"[a-z0-9]+", f"{role} {question}".casefold())
        if len(term) > 2 and term not in stop_words
    }
    chunks = [chunk.strip() for chunk in re.split(r"\n\s*\n", cached_context) if chunk.strip()]
    ranked = sorted(
        ((sum(term in set(re.findall(r"[a-z0-9]+", chunk.casefold())) for term in terms), index, chunk)
         for index, chunk in enumerate(chunks)),
        key=lambda item: (-item[0], item[1]),
    )
    relevant = [chunk for score, _index, chunk in ranked if score > 0][:3]
    return "\n\n".join(relevant)[:1800] if relevant else "No relevant cached role requirements are available."


def _needs_role_retrieval(question: str) -> bool:
    normalized = re.sub(r"\s+", " ", question.casefold()).strip()
    explicit_requirement = re.search(
        r"\b(requirements?|responsibilities|job descriptions?|qualifications?|tools? required)\b|"
        r"\bskills?\b.{0,35}\b(required|needed|expected)\b|"
        r"\b(required|needed|expected)\b.{0,35}\bskills?\b",
        normalized,
    )
    role_reference = re.search(
        r"\b(?:for|in|as)\b.{0,55}\b(?:target\s+|selected\s+)?(?:job|role|position)\b|"
        r"\b(data analyst|data scientist|ai engineer|ml engineer|python developer|business analyst)\b",
        normalized,
    )
    return bool(explicit_requirement or role_reference)


def _greeting_guidance(question: str, history: str) -> str:
    normalized = re.sub(r"[^a-z ]", "", question.lower()).strip()
    greeting = bool(re.match(r"^(hi|hello|hey|good morning|good afternoon|good evening)(\s|$)", normalized))
    if greeting:
        return "The user greeted you; return a natural brief greeting and answer their question."
    if history != "No earlier messages.":
        return "This is a follow-up in an ongoing conversation. Do not greet, welcome, or reintroduce yourself. Continue directly from the prior exchange."
    return "This is the first message. A brief greeting is optional; answer the career question directly."


def _short_text(value, limit: int) -> str:
    if isinstance(value, dict):
        value = value.get("name") or value.get("skill") or value.get("title") or value.get("text") or value.get("summary") or ""
    elif isinstance(value, list):
        value = ", ".join(str(item) for item in value[:8])
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:limit].rstrip()


def _profile_skills(profile: dict) -> list[str]:
    skills = []
    for field in ("technical_skills", "soft_skills", "skills"):
        values = profile.get(field, [])
        if isinstance(values, str):
            values = [values]
        if isinstance(values, list):
            skills.extend(str(value) for value in values if value)
    return skills


def _items(value):
    if value in (None, "", []):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in ("items", "skills", "results"):
            nested = value.get(key)
            if isinstance(nested, list):
                return nested
        return [value]
    return [value]


def _entry(value, status):
    if isinstance(value, dict):
        skill = value.get("skill") or value.get("name") or value.get("requirement") or value.get("technology") or ""
        explanation = value.get("explanation") or value.get("evidence") or value.get("details") or ""
        status = value.get("status") or value.get("match_status") or status
    else:
        skill, explanation = value, ""
    skill = str(skill).strip()
    return {"skill": skill, "status": status, "explanation": str(explanation).strip()} if skill else None


def _structure_skill_gaps(raw) -> dict:
    result = {"matching_skills": [], "missing_skills": [], "skills_to_improve": [], "recommendations": []}
    category_status = {
        "matching_skills": "Available",
        "missing_skills": "Missing",
        "skills_to_improve": "Needs improvement",
    }
    structured = isinstance(raw, dict) and any(key in raw for key in category_status)
    if structured:
        for category, status in category_status.items():
            result[category] = [entry for item in _items(raw.get(category)) if (entry := _entry(item, status))]
        raw_recommendations = raw.get("recommendations", [])
        result["recommendations"] = [_recommendation_text(item) for item in _items(raw_recommendations)]
        result["recommendations"] = [item for item in result["recommendations"] if item]
        # Preserve useful skill_gaps from models that return both schemas, adding
        # any entries omitted from their grouped arrays without duplicating them.
        known = {re.sub(r"[^a-z0-9]", "", row["skill"].lower()) for category in category_status for row in result[category]}
        for item in _items(raw.get("skill_gaps", [])):
            row = _entry(item, "Missing")
            if row:
                normalized = re.sub(r"[^a-z0-9]", "", row["skill"].lower())
                if normalized not in known:
                    destination = _status_category(row["status"])
                    result[destination].append(row)
                    known.add(normalized)
    else:
        rows = _items(raw.get("skill_gaps", raw.get("skills", [])) if isinstance(raw, dict) else raw)
        for item in rows:
            row = _entry(item, "Missing")
            if row:
                result[_status_category(row["status"])].append(row)

    all_rows = result["matching_skills"] + result["missing_skills"] + result["skills_to_improve"]
    if not result["recommendations"]:
        result["recommendations"] = [
            f"Recommended: build a focused practice exercise or portfolio example for {row['skill']}."
            for row in (result["missing_skills"] + result["skills_to_improve"])[:5]
        ]
    if not result["recommendations"]:
        result["recommendations"] = ["Recommended: polish a role-focused portfolio project and practice explaining the decisions and outcome."]
    result["skill_gaps"] = all_rows
    return result


def _status_category(status: str) -> str:
    normalized = re.sub(r"[^a-z]", "", status.lower())
    if any(term in normalized for term in ("available", "matched", "matching", "present", "have")):
        return "matching_skills"
    if any(term in normalized for term in ("improv", "partial", "developing", "intermediate")):
        return "skills_to_improve"
    return "missing_skills"


def _recommendation_text(value) -> str:
    if not isinstance(value, dict):
        return str(value).strip()
    title = value.get("action") or value.get("recommendation") or value.get("title") or value.get("text") or ""
    detail = value.get("explanation") or value.get("details") or value.get("why") or ""
    if title and detail:
        return f"{str(title).strip()} — {str(detail).strip()}"
    return str(title or detail).strip()


def _fallback_roadmap(role: str, analysis: dict) -> str:
    missing = analysis.get("missing_skills", [])
    improve = analysis.get("skills_to_improve", [])
    priorities = [row["skill"] for row in missing + improve if row.get("skill")]
    if not priorities:
        priorities = [row["skill"] for row in analysis.get("matching_skills", [])[:2] if row.get("skill")]
    if not priorities:
        priorities = ["role fundamentals", "a practical portfolio project"]
    high_priority = [row["skill"] for row in missing if row.get("skill")]
    medium_priority = [row["skill"] for row in improve if row.get("skill")]
    first = (high_priority + medium_priority)[:2] or priorities[:1]
    second = [skill for skill in priorities if skill not in first][:2] or first
    return (
        f"### Weeks 1–2: {', '.join(first)}\n"
        f"- Practice the highest-priority skills using exercises that match {role} requirements.\n"
        f"- **Task:** Complete one focused exercise for each skill and save the result as evidence.\n\n"
        f"### Weeks 3–4: {', '.join(second)} in a project\n"
        f"- Apply the next skills in a role-relevant project, building on what you already know.\n"
        f"- **Task:** Publish a concise case study or README showing your approach, results, and the skills demonstrated."
    )
