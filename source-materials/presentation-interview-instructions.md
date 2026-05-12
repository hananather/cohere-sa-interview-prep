# Cohere SA Presentation Interview Instructions

Source of truth for this repo.

- Original PDF: `source-materials/Nov 2025 - SA Presentation Interview Instructions (1).pdf`
- Desktop copy reviewed: `/Users/hananather/Desktop/Nov 2025 - SA Presentation Interview Instructions (1).pdf`
- SHA-256: `a3f1dd39e5675c8b1c9fb36958f3651ae8a5d97a738635de0e1c64875279a41a`
- Pages: 2
- Parsed date: 2026-05-10

## Why This File Matters

- This is the only direct instruction received from Cohere for the presentation.
- All strategy, demo scope, architecture choices, Q&A prep, and documentation should trace back to this brief.
- Other notes are useful only when they support this brief. They are not equal sources of truth.

## Assignment In One Sentence

Act as a Cohere Solutions Architect continuing a customer discussion, choose a scenario, and present a Cohere-based solution with a live demo, technical architecture, assumptions, and answers to business and technical stakeholder questions.

## Evaluation Signal

The interview is designed to assess:

- Technical expertise.
- Ability to apply that expertise to real customer problems.
- Solutioning judgement under ambiguity.
- Ability to present to customer stakeholders.
- Ability to answer interactive business and technical questions.
- Ability to show a working demo and explain how it works.

## Required Presentation Shape

- Allocate about 20 to 25 minutes for presentation and discussion.
- Leave the remaining time for open Q&A.
- Expect role-play with the panel acting as customer stakeholders.
- Expect interruptions and questions throughout.
- Include a live demo.
- Follow the demo with a technical implementation walkthrough.
- Prioritize how the solution works over UI polish.
- Use Cohere models where they strengthen the solution.
- Make assumptions explicit.

## Scenario Options

The PDF offers five scenario families:

- Ecommerce personalization and secure product search.
- Healthcare diagnostic support and confidential patient data.
- Financial services compliance and transparent risk assessment.
- Educational personalization and privacy-respecting curriculum adaptation.
- Public sector Defence Agent.

## Selected Scenario: Defence Agent

DefTech is a national defence technology agency that wants an AI assistant to help staff interrogate manuals, procedures, and doctrine.

### Customer Background

- Staff previously relied on internal users reading PDFs, physical manuals, and textbooks.
- DefTech has gone through a digital transformation.
- Manuals and procedures are now contained within PDFs and Docx files in a central database.

### Executive Sponsor Signal

The Chief of Staff wants AI over the new database to:

- Increase the efficiency of central planning staff.
- Increase the output of central planning staff.
- Preserve accuracy.
- Preserve traceability.

## Explicit Requirements For This Repo

Every major project choice should map to at least one of these requirements:

| Requirement from PDF | Repo implication |
| --- | --- |
| Solve a real customer problem | Start from central planning staff workflow, not from model features. |
| Cohere-based solution | Use Cohere where it directly improves retrieval, reranking, grounded answers, citations, and agent behavior. |
| Live demo required | Keep the demo reliable, scripted, and small enough to run under interview pressure. |
| Technical implementation walkthrough required | Keep architecture explainable from ingestion to answer. |
| Scalable and secure | Show the prototype pattern and the production hardening path. |
| PDFs and Docx in central database | The prototype indexes normalized PDF page artifacts. It includes one DOCX-origin source represented through the publisher's official PDF pair. Native `.docx` parsing is not implemented. |
| Improve efficiency and output | Show how the assistant reduces manual source lookup and produces cited, reviewable staff answers. |
| Accuracy is key | Show retrieval, reranking, answerability/refusal behavior, evaluation cases, and human review boundaries. |
| Traceability is key | Show citations, source metadata, retrieved pages, persona, filters, and audit-ready run metadata. |

## Strategic Interpretation

This section is project interpretation of the PDF brief. It is not additional
verbatim instruction from Cohere.

The best demo is not a generic chatbot.

The best demo is an evidence assistant:

- It searches approved source documents.
- It retrieves source pages or passages.
- It filters evidence by user access before generation.
- It ranks evidence by relevance.
- It answers only from authorized evidence.
- It cites the sources so staff can verify the answer.
- It refuses when authorized evidence is not enough.

## Current Prototype Alignment

The current Defence Agent prototype should be interpreted through this brief:

- Trust: grounded answers, citations, refusal when evidence is insufficient, evaluation cases.
- Security: persona-aware retrieval filters, no restricted source text sent to unauthorized users, tool policy.
- Privacy: local corpus, controlled data path, no public-internet dependency for customer data in the target production framing.
- Traceability: citations, source document IDs, page numbers, metadata filters, session/run state.
- Technical competency: Cohere Embed v4, Cohere Rerank v4, Cohere grounded generation/citations, Google Agent Development Kit session/tool state, manifest-driven ingestion.

## Presentation Guardrails

- Do not claim production accreditation.
- Do not claim native Docx parsing is implemented.
- If asked about Docx, say DOCX-origin sources can be normalized into the same
  page-level evidence pipeline, and the retrieval layer is format-agnostic after
  verified normalization.
- Do not claim the assistant makes decisions.
- Do not claim citations alone solve all traceability.
- Do not let bilingual retrieval, synthetic restricted docs, or UI details become the main story.
- Do not overbuild. Increase capability only when it strengthens trust, security, privacy, accuracy, traceability, or technical credibility.

## Verbatim Extract

```text
Objective: This phase is designed to assess your technical expertise and how you apply
it to solve real-world customer problems. Your task is to analyze a given scenario and
showcase your solution to a group of customer stakeholders.

Assignment: You are a Cohere Solution Architect that is continuing a previous
discussion with a customer. Examine the provided customer scenarios and prepare a
presentation that demonstrates your Cohere based solution. Your presentation should
clearly outline the problem, explain your proposed solution, and walk through the
technical architecture you are proposing. You should be prepared to answer questions
and show how your solution addresses the problem. Remember, some ambiguity is
expected in the scenario. Feel free to make and highlight necessary assumptions for
your solution.

Presentation Guidelines:

- Plan to allocate approximately 20-25 minutes for the presentation and discussion,
  using the remaining time for an open Q&A session.
- The Presentation Interview will be in a role-play format, with the interview panel
  assuming roles from the customer side. The interview panel will ask questions
  throughout the presentation. Be prepared for an interactive session with technical
  and business questions from multiple stakeholders.
- A live demo of the solution followed by a walk through of the technical
  implementation details is required. Feel free to use a jupyter notebook, streamlit,
  or any similar tools you are comfortable with.
- We are more focused on how it works, not how good it looks. While you can leverage
  public notebooks as a starting point, please ensure to modify or add elements to
  make it unique and showcase your technical ability. Using Cohere models is highly
  encouraged.

Scenario: Each of these scenarios requires an LLM/NLP-Focused solution that is
scalable, secure, and tailored to the specific needs and challenges of the industry.
Choose one that best allows you to showcase your solutioning and technical acumen.

Ecommerce: A leading e-commerce firm, GlobalMart has experienced a significant
decline in sales that has coincided with a competitor's introduction of an intelligent
personal shopper feature. Data analysis indicates that customers are leaving
GlobalMart's website early and spending less time shopping, possibly due to less
personalized and relevant product search results. The business team seeks a scalable
solution that efficiently utilizes customer data and reviews to provide highly
pertinent search results, with a critical emphasis on keeping their data secure and off
the public internet.

Healthcare Integration: Cohere is approached by HealthFirst, a large hospital network
looking to improve patient care through technology. They want to integrate AI-driven
diagnostic tools that can analyze patient data, external information, doctors notes,
and other unstructured data to assist doctors in making faster, more accurate
diagnoses. The challenge is to ensure the solution is compliant with healthcare
regulations, integrates seamlessly with their existing systems, and maintains patient
confidentiality.

Financial Services Compliance: FinTrust, a leading bank, is concerned about regulatory
compliance in its lending practices. They need an AI solution that can analyze loan
applications, assess risk, and ensure compliance with various international financial
regulations. The current system has trouble processing the text data within loan
applications. The system must be scalable, transparent, and able to adapt to changing
regulations.

Educational Personalization: EduTech, an online education platform, wants to
personalize learning for its users. They need a system that can analyze student
performance, learning styles, and preferences to tailor the content, pace, and
teaching methods for each learner. The challenge is to build a scalable, secure
solution that respects privacy and can adapt to different educational curricula and
standards.

If you are interviewing for a Public Sector opportunity on our Solutions Architect
team you may also choose this scenario:

Defence Agent: DefTech, a national defence technology agency that wants an AI
assistant to help their staff interrogate manuals, procedures and doctrine. They have
previously relied on internal users reading PDFs, physical manuals and textbooks. They
have recently gone through a digital transformation where all manuals and procedures
are contained within PDFs and Docx files in a central database. The feedback from
their Chief of Staff is that they need to leverage AI over their new database to
increase the efficiency and output of central planning staff while accuracy and
traceability are key.
```
