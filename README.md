STEP 1 — User uploads existing resume (optional, multimodal)
         OR starts fresh

STEP 2 — Chat questionnaire (LangChain + Gemini)
         Bot asks 10 questions one by one:
         Name → Contact → Summary → Experience →
         Education → Skills → Projects → Certifications →
         Target Role → Job Description

STEP 3 — Gemini generates professional resume content
         from all collected answers

STEP 4 — BiLSTM + Attention classifies generated resume
         → verifies correct category

STEP 5 — Gemini scores resume on 10 criteria
         → ATS keywords → LinkedIn summary

STEP 6 — ReportLab generates beautiful PDF
         → user downloads