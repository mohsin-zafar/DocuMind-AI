
def evaluate_resumes(job_description, resumes, ollama):
    system = (
        "You are an evidence-based resume screening assistant. "
        "You support human review and never make hiring decisions. "
        "Assess only job-related evidence. Do not infer protected traits. "
        "Never assume a skill exists if it is not stated. "
        "Use MATCHED, PARTIALLY MATCHED, NOT FOUND, or UNCLEAR. "
        "Support every assessment with resume evidence. "
        "Do not invent credentials or years of experience."
    )

    # Stage 1: Extract job requirements once.
    requirements_prompt = f"""
Analyze this job description and extract its requirements.

JOB DESCRIPTION:
{job_description[:10000]}

Return a concise, structured list containing:
1. Required technical skills
2. Required experience
3. Required education
4. Preferred skills
5. Responsibilities
6. Important soft skills

Separate required and preferred qualifications.
Do not add requirements that are not stated.
Keep the output concise.
"""

    requirements = ollama.generate(
        requirements_prompt,
        system=system,
        temperature=0.1
    )

    results = [
        "JOB REQUIREMENTS",
        str(requirements),
        "\n" + "=" * 60
    ]

    # Stage 2: Evaluate each candidate separately.
    for i, resume in enumerate(resumes, 1):
        resume_text = resume.get("text", "")[:9000]
        source = resume.get("source", f"Candidate {i}")

        candidate_prompt = f"""
Evaluate this candidate against the extracted job requirements.

JOB REQUIREMENTS:
{requirements}

CANDIDATE:
{source}

RESUME:
{resume_text}

Return these sections:

1. REQUIREMENT EVIDENCE MATRIX
For each requirement, provide:
- Requirement
- Status: MATCHED / PARTIALLY MATCHED / NOT FOUND / UNCLEAR
- Exact supporting resume evidence, or state "No evidence found"

2. STRENGTHS
List relevant skills, experience, projects and achievements.

3. GAPS
List job requirements not clearly evidenced in the resume.

4. INTERVIEW QUESTIONS
Suggest up to 5 questions to verify relevant skills or unclear claims.

5. JOB-RELATED FIT SCORE
Give a cautious score from 0-100 using:
- Required technical/job skills: 40%
- Relevant experience: 25%
- Preferred skills: 15%
- Education/certifications: 10%
- Relevant projects/achievements: 10%

Explain the score and any uncertainty.
Do not treat missing information as proof that the candidate lacks a skill.
Do not make a hiring recommendation.
Treat resume statements as claims, not independently verified facts.
"""

        try:
            analysis = ollama.generate(
                candidate_prompt,
                system=system,
                temperature=0.1
            )
            results.append(
                f"\nCANDIDATE {i}: {source}\n\n{analysis}\n"
            )

        except Exception as exc:
            results.append(
                f"\nCANDIDATE {i}: {source}\n"
                f"Analysis failed: {str(exc)}\n"
            )

        results.append("\n" + "=" * 60)

    return "\n".join(results)
