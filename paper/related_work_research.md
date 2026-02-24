# Related Work Research — CompToolBench (Feb 2026)

## TIER 1: DIRECT COMPETITORS

### 1. ComplexFuncBench (Tsinghua, Jan 2025)
- arXiv:2501.10132
- ~1K tasks, 43 real-time APIs, long-context (128K tokens)
- Focuses on parameter constraints, not compositional topology
- No L0-L3 taxonomy, no Composition Gap metric

### 2. ToolComp / SEAL (Scale AI, Jan 2025)
- 485 prompts, 11 enterprise tools, process supervision
- Sequential chains only (no parallel/DAG)
- No Composition Gap metric

### 3. FuncBenchGen (Megagon Labs, Sep 2025, ICLR 2026)
- arXiv:2509.26553
- Contamination-free, DAG-based, controllable difficulty
- Uses SYNTHETIC functions (not real APIs)
- Closest competitor — shares DAG formulation
- GPT-5 outperforms other models significantly
- No Composition Gap or Selection Gap metrics

### 4. TPS-Bench (Nov 2025)
- arXiv:2511.01527
- 200 compounding tasks, MCP tools, planning vs scheduling
- Focuses on time optimization, not composition accuracy
- No L0-L3 taxonomy

### 5. BFCL V4 (Berkeley, ICML 2025)
- Agentic multi-turn, AST evaluation, live leaderboard
- No formal composition taxonomy or gap metric
- Primarily a leaderboard, not a research benchmark

### 6. Nexus Function Calling (Nexusflow, 2024-2025)
- 762 test cases, 3 categories (single/parallel/nested)
- Simpler taxonomy than CompToolBench's 4-level

## TIER 2: MCP-ECOSYSTEM

### 7. MCP-Bench (Accenture, Aug 2025) — arXiv:2508.20453
- 28 MCP servers, 250 tools, fuzzy instructions
### 8. MCPAgentBench (Dec 2025) — arXiv:2512.24565
- 841 tasks, 20K+ MCP tools, distractor tools
### 9. MCPToolBench++ (Aug 2025) — arXiv:2508.07575
- 4K+ MCP servers, 40+ categories, multilingual
### 10. LiveMCPBench (Aug 2025) — arXiv:2508.01780
- 95 real-world tasks, 70 servers, 527 tools, LLM-as-Judge

## TIER 3: REAL-WORLD / INTERACTIVE

### 11. WildToolBench (Oct 2025, ICLR 2026 submission)
- 57 LLMs, best <15%, real user behavior patterns
### 12. tau-bench / tau2-bench (Sierra, 2024-2025)
- arXiv:2406.12045, customer service domains
### 13. ToolSandbox (Apple, NAACL 2025) — arXiv:2408.04682
- Stateful conversational tool use
### 14. OpaqueToolsBench (Feb 2026) — arXiv:2602.15197
- Imperfect documentation, learning through interaction

## TIER 4: FOUNDATIONAL

### 15. ToolBench / StableToolBench (ICLR 2024)
- arXiv:2307.16789, 16K APIs, known quality issues
### 16. MINT (ICLR 2024) — arXiv:2309.10691
- Multi-turn with feedback
### 17. ToolTalk (2023-2024)
- 28 APIs, dialogue-focused

## TIER 5: TRAINING FRAMEWORKS

### 18. ToolACE / ToolACE-MT (ICLR 2025/2026) — arXiv:2409.00920
### 19. APIGen / APIGen-MT (Salesforce, NeurIPS 2024) — arXiv:2406.18518
### 20. ToolMind (Nanbeige, Nov 2025) — arXiv:2511.15718
- 360K samples, 20K tools

## TIER 6: ADJACENT AGENTIC

### 21. TRAIL (Patronus AI, May 2025) — arXiv:2505.08638
- Trace-level debugging, 148 traces
### 22. AgentIF (Tsinghua, NeurIPS 2025) — arXiv:2505.16944
- Instruction following in agentic scenarios
### 23. ML-Tool-Bench (Nov 2025) — arXiv:2512.00672
- ML pipeline planning
### 24. TOP-Bench (Dec 2025) — arXiv:2512.16310
- Privacy risks from tool orchestration

## CompToolBench UNIQUE EDGES

1. **4-level taxonomy (L0/L1/L2/L3)** — no one else has all 4
2. **"Composition Gap" metric** — completely novel
3. **"Selection Gap" finding** — never reported
4. **Real tools + zero cost + 18 models** — unique combination
5. **DAG-structured tasks** — only FuncBenchGen is close (synthetic)
