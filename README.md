# Běichén 北辰

*"He who governs by virtue is like the North Star, which remains in its place while all the other stars revolve around it."* — Confucius, Analects 論語

Běichén is an Inquiry-Based Learning AI Tutor proof-of-concept.
Built for **CCNU Project 7 - AI Tutoring System** 
*(SAgE Blended Mobility Programme, Newcastle x Faculty of AI in Education, Wuhan)*

The AI Tutor never answers a Learner's question directly. It guides them through a structured inquiry cycle — eliciting a hypothesis, investigating it through evidence and questioning, and only letting the Learner state the conclusion themselves — enforced by a pedagogical state machine rather than left to the model's own judgement.

Within the context of this program, the "North Star" is then defined as the Target Conclusion the AI Tutor guides the Learner towards.

---

## Topic & Strategy

- **Topic:** Physical Geography (clear cause-and-effect mechanisms, consistent terminology)
- **Strategy:** Inquiry-Based Learning: a fusion of Pedaste's 5-Phase model (process architecture) and Bybee's 5E model (tutor "move" taxonomy)
- **Core design principle:** the LLM does not decide the pedagogy. The architecture decides what Phase the Learner is in, whether evidence is needed, whether to hint, whether to conclude - the LLM generates language within those constraints.

Full design rationale, the five pedagogical rules, the Component/dependency-graph architecture, and the system diagrams live in [`docs/`](./docs):
- `AI_Tutor_Design_v2.md` — the design document
- `setup_pipeline_diagram.svg` — Question Analysis → Conclusion Decomposition
- `system_flow_diagram.svg` — runtime architecture / Decision Policy
- `phases_cycle_diagram.svg` — the Pedaste×5E inquiry cycle

---

## Current Status

**Layer 1 (demoable core) - built.**
- Gradio chat wired to DeepSeek V4 Flash
- Three-Phase loop (ask → investigate → conclude) enforced via system prompt
- `[STATE phase / hypothesis_recorded / resolved]` tag parsed per turn, with a consistency guard against impossible combinations
- Edge-triggered auto-export: on genuine resolution, writes a Markdown log (transcript + a second, independent LLM pass auditing the session against the five pedagogical rules) to `logs/`
- Manual JSON export available independently of the auto-export

**Layer 2 (grounding + real decomposition) - `<TODO: in progress>`**
- Wikipedia retrieval for the internal answer key + evidence
- Question Analysis → Conclusion Decomposition pipeline producing `required_components`
- Mastery / groundedness tracked in code rather than prompt-only

**Layer 3 (polish) - `<TODO: not started>`**
- Exhaustion Fallback ladder
- Inquiry-graph render

---

## Tech Stack

| Layer | Choice |
|---|---|
| Language | Python |
| LLM | DeepSeek V4 Flash (`deepseek-v4-flash`), OpenAI-compatible SDK |
| UI | Gradio |
| Grounding (Layer 2+) | Wikipedia API |

---

## Setup

```bash
git clone https://github.com/b3n-hash/Beichen-Tutor.git
cd polaris-tutor
pip install -r requirements.txt
```

Create a `.env` file in the project root:
```
DEEPSEEK_API_KEY=your_key_here
```

Test the API connection functions on its own (no UI):
```bash
python main.py
```

Run the full program:
```bash
python app.py
```


---

## Team & Roles

| Person | Background | Owns |
|---|---|---|
| Ben | CompSci | Backend, state machine, decomposition pipeline, recursive audit, repo |
| Nick | IT | Interface, conversation history, UI/integration |
| Olga | Education | Pedagogical strategy, rules, citations, learner-facing framing |
| Khalifa | Mech Eng | Conclusion→component decomposition maps, testing/validation protocols |

---

## Brief Tasks → Where They Live

| Task | Where                                                                    |
|---|--------------------------------------------------------------------------|
| 1. Tutoring Strategy Design | `docs/AI_Tutor_Design_v2.md` - Phases & Pedagogical Rules sections       |
| 2. Chatbot Implementation | `app.py`                                                                 |
| 3. Learner Testing & Iteration | `logs/` - exported sessions, each with an automated rule-violation audit |
| 4. Tutoring Quality Reflection | `<TODO: link to writeup once drafted>`                                   |

---

## Known Limitations

- Open topic scope within Physical Geography means evidence-grounding (Layer 2) is the main outstanding technical risk -vsee `docs/AI_Tutor_Design_v2.md` for the tradeoff discussion.
- The Recursive Audit is itself an LLM call and may be lenient on subtle violations (e.g. confirmations phrased as encouragement) - spot-check logs by hand, don't treat the audit as ground truth.
- `resolved` detection can silently miss a genuine resolution if the model confirms an answer without emitting the tag; a `[WATCH]` console log flags this for now (see `app.py`).

---
