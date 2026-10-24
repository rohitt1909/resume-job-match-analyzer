"""
Resume-to-Job-Match Analyzer
Compares a resume to a job description using an LLM and returns structured JSON.

Two prompt versions are included so you can measure the improvement:
  v1 = naive prompt (shows the problems)
  v2 = engineered prompt (the fix)

Usage:
  export GEMINI_API_KEY="your-key"   (free key from aistudio.google.com/app/apikey)
  python analyzer.py run  --resume samples/resume.txt --job samples/job.txt --version v2
  python analyzer.py eval --resume samples/resume.txt --job samples/job.txt --runs 10
"""
import argparse
import json
import os
import re
import statistics
import sys
import time

from google import genai
from google.genai import types

MODEL = os.environ.get("MODEL", "gemini-3.5-flash")
client = genai.Client()  # reads GEMINI_API_KEY from the environment

# ---------------------------------------------------------------------------
# PROMPT V1: naive. Expect: extra text around JSON, inconsistent scores,
# invented skills.
# ---------------------------------------------------------------------------
PROMPT_V1 = """Compare this resume to the job description. Give a match score out of 100,
the matching skills, the missing skills, and suggestions. Return it as JSON.

RESUME:
{resume}

JOB DESCRIPTION:
{job}
"""

# ---------------------------------------------------------------------------
# PROMPT V2: engineered. Fixes:
#   1. Strict output schema + "JSON only"      -> parseable output
#   2. Evidence rule (quote from resume)       -> no invented skills
#   3. Sub-score rubric, total computed in code -> consistent scores
#   4. Few-shot example                         -> consistent format
#   5. Explicit handling of missing info        -> null/empty instead of guessing
# ---------------------------------------------------------------------------
SYSTEM_V2 = """You are a strict resume screening assistant.

Return ONLY a single valid JSON object. No markdown, no code fences, no text before or after.

RULES
1. A skill counts as "matching" ONLY if it is explicitly written in the resume.
   For each matching skill, include "evidence": a short exact quote (max 12 words) copied from the resume.
2. Never infer or assume skills. If the resume does not mention it, it is missing.
3. "missing_skills" lists requirements from the job description that the resume does not show.
4. Score each category as an integer using this rubric:
   - skills_score (0-50): 50 x (fraction of the job's required skills that are matched)
   - experience_score (0-30): 30 = meets or exceeds required experience, 15 = partially, 0 = none shown
   - education_score (0-20): 20 = meets the education requirement, 10 = partially, 0 = does not
5. "suggestions" must be 3 concrete, actionable items based only on the missing skills.

OUTPUT SCHEMA
{
  "matching_skills": [{"skill": string, "evidence": string}],
  "missing_skills": [string],
  "skills_score": integer,
  "experience_score": integer,
  "education_score": integer,
  "suggestions": [string, string, string]
}

EXAMPLE
Resume: "Skills: Java, SQL. Intern at ABC (6 months): built REST APIs. B.Tech CS."
Job: "Requires Java, SQL, Docker. 1+ years experience. Degree in CS."
Output:
{"matching_skills":[{"skill":"Java","evidence":"Skills: Java, SQL"},{"skill":"SQL","evidence":"Skills: Java, SQL"}],
"missing_skills":["Docker"],"skills_score":33,"experience_score":15,"education_score":20,
"suggestions":["Add a project that uses Docker","Describe the scale of the REST APIs you built","Extend experience with a longer internship or freelance work"]}
"""

USER_V2 = """RESUME:
{resume}

JOB DESCRIPTION:
{job}

Return the JSON object now."""


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------
def call_llm(version, resume, job):
    """Send the prompt for the chosen version and return the raw text reply."""
    if version == "v1":
        contents = PROMPT_V1.format(resume=resume, job=job)
        config = None
    else:
        contents = USER_V2.format(resume=resume, job=job)
        config = types.GenerateContentConfig(system_instruction=SYSTEM_V2)

    for attempt in range(4):  # free tier has rate limits, so retry politely
        try:
            resp = client.models.generate_content(model=MODEL, contents=contents, config=config)
            return resp.text or ""
        except Exception as e:
            print(f"  (API error, retrying in 20s: {str(e)[:100]})")
            time.sleep(20)
    return ""


def parse_json(text):
    """Strict parse: succeeds only if the WHOLE reply is valid JSON."""
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return None


def extract_json_lenient(text):
    """Lenient parse used for v1, to still get data out of messy replies."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def total_score(data):
    """v2 total is computed in code from sub-scores (more reliable than asking the LLM to add)."""
    if data is None:
        return None
    if all(k in data for k in ("skills_score", "experience_score", "education_score")):
        return data["skills_score"] + data["experience_score"] + data["education_score"]
    for key in ("match_score", "score", "overall_score"):  # v1 uses whatever name it likes
        if isinstance(data.get(key), (int, float)):
            return data[key]
    return None


def skill_names(data):
    """Return matching skills as plain strings (handles both v1 and v2 shapes)."""
    if not data:
        return []
    raw = data.get("matching_skills") or data.get("matched_skills") or []
    names = []
    for item in raw:
        names.append(item["skill"] if isinstance(item, dict) and "skill" in item else str(item))
    return names


def count_hallucinated(data, resume):
    """A skill is 'hallucinated' if it never appears in the resume text."""
    resume_low = resume.lower()
    return [s for s in skill_names(data) if s.lower() not in resume_low]


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def cmd_run(args):
    resume, job = open(args.resume).read(), open(args.job).read()
    raw = call_llm(args.version, resume, job)
    data = parse_json(raw) or extract_json_lenient(raw)
    if data is None:
        print("Could not parse JSON. Raw output:\n", raw)
        return
    data["total_score"] = total_score(data)
    data["hallucinated_skills"] = count_hallucinated(data, resume)
    print(json.dumps(data, indent=2))


def cmd_eval(args):
    resume, job = open(args.resume).read(), open(args.job).read()
    print(f"Model: {MODEL} | Runs per version: {args.runs}\n")
    summary = {}
    for version in ("v1", "v2"):
        valid_strict, scores, halluc, total_skills = 0, [], 0, 0
        for i in range(args.runs):
            raw = call_llm(version, resume, job)
            time.sleep(4)  # stay under free-tier rate limits
            if parse_json(raw) is not None:
                valid_strict += 1
            data = parse_json(raw) or extract_json_lenient(raw)
            score = total_score(data)
            if score is not None:
                scores.append(score)
            bad = count_hallucinated(data, resume)
            halluc += len(bad)
            total_skills += len(skill_names(data))
            print(f"  {version} run {i+1}: score={score} hallucinated={bad}")
        summary[version] = {
            "clean_json_rate": f"{valid_strict}/{args.runs}",
            "score_min_max": f"{min(scores)}-{max(scores)}" if scores else "n/a",
            "score_stdev": round(statistics.pstdev(scores), 2) if len(scores) > 1 else "n/a",
            "hallucinated_skills": f"{halluc}/{total_skills}",
        }
        print()
    print("=" * 60)
    print(f"{'Metric':<24}{'v1 (naive)':<18}{'v2 (engineered)':<18}")
    for metric in ("clean_json_rate", "score_min_max", "score_stdev", "hallucinated_skills"):
        print(f"{metric:<24}{str(summary['v1'][metric]):<18}{str(summary['v2'][metric]):<18}")


def main():
    if not os.environ.get("GEMINI_API_KEY"):
        sys.exit("Set your key first. Windows: set GEMINI_API_KEY=your-key   Mac: export GEMINI_API_KEY=your-key")
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="analyze one resume/job pair")
    r.add_argument("--resume", required=True)
    r.add_argument("--job", required=True)
    r.add_argument("--version", choices=["v1", "v2"], default="v2")
    r.set_defaults(func=cmd_run)

    e = sub.add_parser("eval", help="compare v1 vs v2 over repeated runs")
    e.add_argument("--resume", required=True)
    e.add_argument("--job", required=True)
    e.add_argument("--runs", type=int, default=10)
    e.set_defaults(func=cmd_eval)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
