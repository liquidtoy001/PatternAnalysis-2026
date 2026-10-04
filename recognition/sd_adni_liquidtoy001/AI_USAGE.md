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

## 2026-10-03 — Repair incompatible dependency pins

Codex inspected the actual setup failure, acknowledged the incompatible initial
Hub pin, checked PyPI metadata, changed Hub to 1.33.0 and resubmitted the guarded
CPU/GPU job chain (625832/625833). This is a dependency correction, not a successful
model experiment. Student review remains pending.

## 2026-10-04 — Real LoRA restart acceptance and website update

Codex authored the pre-run protocol, VAE clipping diagnostics and minimal LoRA
training/restart comparator; verified package constraints against PyPI metadata;
and ran CPU/GPU jobs 629842, 629851 and 629852. All completed with exit 0:0.
The restart comparator used real checkpoints and included intentional corruption
controls. Raw medical artifacts remain ignored/private; aggregate evidence was
used for the learning website. Student review is pending. This is engineering
acceptance, not a successful formal generation experiment.

## 2026-10-04 — Paired 200-step pilot

Codex read assessment Section2.4, authored the pre-run pilot protocol and adapted
an isolated copy of the accepted smoke trainer for 32 training patients and200
updates, with deterministic shuffled order and fixed DDIM sampling at0/100/200.
The original smoke code is unchanged. Local Ruff and syntax checks passed; remote
Python/shell syntax checks passed. Submitted job629919. Execution and visual
acceptance are pending; student code/protocol review remains pending.

Pilot629919 completed with exit0:0 in2m15s. Codex retrieved private artifacts,
verified actual traces/paired noise/file hashes and inspected all three grids.
Observed appearance changes and persistent structural failures are recorded in the
ledger without claiming medical validity. Student verification remains pending.

## 2026-10-04 — Queue full-split experiment

At the student's explicit request, Codex inspected live Slurm/account/QoS/storage,
recorded the long-run protocol, adapted an isolated trainer, ran local Ruff/syntax
and remote Python/shell syntax checks, and submitted gate629930 plus dependent
training629931. Preserved accepted pilot/smoke code and frozen queued source.
Full-split shuffled order and stage checkpoints are included in the resume contract.
Student review pending; queue submission is not training or quality acceptance.
