# Rule Engine Documents

This folder should contain the 8 DOCX rule engine specification documents delivered during Phase 0. These documents contain the full Signal Definition Cards for all 94 rules.

## Expected Files

Copy the following DOCX files into this directory:

| File | Rules | Status |
|------|-------|--------|
| `voice-agent-rule-engine.docx` | 18 voice rules | Delivered |
| `language-agent-rule-engine.docx` | 12 language rules | Delivered |
| `facial-agent-rule-engine.docx` | 7 facial rules (FACS AU reference) | Delivered |
| `body-agent-rule-engine.docx` | 8 body language rules | Delivered |
| `gaze-agent-rule-engine.docx` | 7 gaze rules | Delivered |
| `conversation-agent-rule-engine.docx` | 7 conversation rules | Delivered |
| `fusion-agent-rule-engine.docx` | 15 pairwise cross-modal rules | Delivered |
| `compound-patterns-engine.docx` | 12 compound + 8 temporal patterns | Delivered |

## Quick Reference

For a condensed reference of all 94 rules without opening the DOCX files, see `../RULES.md`.

## Signal Definition Card Format

Every rule in these documents follows this 7-field structure:

1. **Rule ID**: Unique identifier (e.g., VOICE-STRESS-01)
2. **Signal Name**: Human-readable name
3. **Research Basis**: Citation to supporting study/paper
4. **Raw Features**: What input data the rule requires
5. **Detection Logic**: IF-THEN with specific numerical thresholds
6. **Confidence Calculation**: How certainty is computed (always 0.0-1.0)
7. **Cross-Agent Validation**: Which other agents can confirm/deny this signal
