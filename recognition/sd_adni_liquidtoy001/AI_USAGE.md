# AI Assistance Record

## 2026-10-02 — Project foundation

- Tool: OpenAI Codex.
- Assistance: assessment interpretation, SD/Rangpur planning, repository setup, and drafting project documentation, ignore patterns, and line-ending rules.
- Sources: supplied assessment PDF, Rangpur/GPU Lab records, and current Git repository state.
- Verification by the assistant: read the local assessment and repository documentation; inspect the staged changes and exercise ignore rules before the initial project commit.
- Student review: pending. No claim is made that the student has already reviewed or tested these artifacts.
- Experimental evidence: none. No dataset access, training, or evaluation has been performed for this project.

Add entries as work proceeds. Describe substantive AI contributions accurately and record the actual audit or empirical verification performed. Summarize this record in the README disclosure before submission.

## 2026-10-02 — Private development and submission workflow

- Codex checked the local branch and the public fork branch, confirmed the course topic commit is an ancestor, and consulted GitHub documentation on fork visibility.
- Documented local commits, private local backups, final publication, and a single PR with subsequent review updates. Updated the tentative deadline note using the student's supplied course-profile/tutor excerpts; the final submission timestamp is still unverified.
- Configured a repository-local guard against bare pushes and created a verified Git bundle backup. No project commits were pushed and no PR was opened.

## 2026-10-02 — Read-only Rangpur and dataset audit

- Codex implemented `tools/audit_adni.py`, executed it on actual course data in CPU job 620988, and recorded aggregate output and limitations.
- Checked supplied scan IDs against metadata patient IDs, counted overlap between existing splits, inspected image headers, and inventoried dependencies and disk space. No fabricated test data or training metrics were used.
- Initial shell quoting/CRLF failures are recorded in the audit report. The final audit completed with exit 0:0. Existing GPU training was not interrupted, and no packages were installed or datasets modified.
- Student review of implementation and findings is pending. Patient identifiers and source images are not included in the committed audit output.

## 2026-10-02 — Data preparation before model storage is available

- Codex drafted the pre-experiment data protocol, implemented patient-stratified manifests and image loading, and added a CPU-only real-data validation command.
- Seven meaningful local unit tests passed using explicitly synthetic fixtures; they check data/geometry contracts and rejection of corrupt manifests, not training success. Ruff checks passed. An unnecessary torchvision import was removed after local inspection showed it was absent; tensor conversion uses PyTorch directly.
- Full-data validation is queued as CPU job 623310 and is not yet accepted. No dependency installation, model download, source-data modification, or GPU allocation occurred. Student review remains pending.

## 2026-10-02 — Accepted data and queued VAE diagnostic

Codex verified completed CPU job 623310 and retrieved its real aggregate results.
After the student confirmed storage release, a read-only check found approximately
12 GiB free and no running user jobs. Codex documented the VAE protocol, pinned
the SD 1.5 VAE revision, implemented isolated package setup and a bounded paired
reconstruction diagnostic, and submitted dependent CPU/GPU jobs 623826/623828.
Local syntax and Ruff checks passed; remote setup and reconstruction remain
pending. Student review is still pending. No patient-level artifacts were pushed.
