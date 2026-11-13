# Graph RAG vs. Vector RAG on 26 Seasons of Formula 1

**When does a knowledge graph beat vector search for question answering over structured data? Measured on 160 Formula 1 questions whose answers are computable from the data, broken down by how many facts each question chains together.**

**[Explore every question and trace →](https://dojedaro.github.io/f1-graphrag/)**

[![CI](https://github.com/dojedaro/f1-graphrag/actions/workflows/ci.yml/badge.svg)](https://github.com/dojedaro/f1-graphrag/actions/workflows/ci.yml)

## Results

Same model for every system (Claude Haiku 5.5), same data, exact scoring:

| Question type | Example | Vector RAG | Vector agent | Graph agent | Hybrid |
|---|---|---|---|---|---|
| 1-hop (36) | Who won the 2009 German Grand Prix? | 97% | 94% | 100% | 100% |
| 2-hop (36) | Who was Fisichella's teammate for the whole 2004 season? | 58% | 94% | 100% | 100% |
| 3-hop (24) | How many races did the team that won the 2010 Monaco GP win the next season? | 58% | 92% | 100% | 100% |
| Aggregation (24) | How many Grands Prix did Räikkönen win driving for Lotus? | **8%** | 88% | 100% | 100% |
| **Held-out (40)** | In how many 2016 races did the pole-sitter win? | **40%** | **100%** | **100%** | **100%** |
| **All 160** | | **88 (55%)** | **151 (94%)** | **160 (100%)** | **160 (100%)** |

| | Vector RAG | Vector agent | Graph agent | Hybrid |
|---|---|---|---|---|
| Tool calls per question | 0 | 4.7 | 3.3 | 3.6 |
| Seconds per question | 3.4 | 8.1 | 4.5 | 4.6 |
| Total cost, 160 questions | $0.10 | **$0.54** | $0.14 | $0.15 |

**Vector RAG** retrieves the 8 most similar documents once. The **vector agent** is the stronger baseline: the
same model with a `vector_search` tool (8 documents per call) it can call as often as it likes, writing its
own queries, in the same tool loop as the graph agent with a 16-turn limit (the graph agent never needed
more than 8). An earlier vector-agent run with 4 documents per call and 8 turns scored 144/160; it is kept
in `runs/*/vector-agent-k4-8turns/`.

**What the numbers say:**

- **A search loop closes almost all of the gap.** Letting vector search run in an agent loop lifts it from 55%
  to 94%, including all 40 held-out questions: it issues one query per season or race and assembles the
  pieces itself. On structured questions the graph agent's edge is **reliability and efficiency**: right on
  all 160, with fewer tool calls, in about half the time and at about a quarter of the vector agent's cost.

- **Plain vector search is fine for single facts and collapses on counting.** It gets 97% of "who won race X",
  because one retrieved race report holds the answer. It gets 8% of counting questions ("how many races
  did X win for team Y"), because the answer is spread across many race reports and the 8 most similar
  ones are not the complete set. Counting needs completeness, which similarity search doesn't promise.
- **The graph agent wins by computing instead of reading.** For "pole-to-win conversions in 2016" it
  called `race_result` for all 21 races and counted. It even noticed that in two races the pole-sitter
  finished outside the top 10 and reasoned that he therefore couldn't have won.
- **Adding vector search to the graph agent changed nothing here**, for 10% more cost. The graph covers
  everything these questions ask; the hybrid would matter for unstructured questions (race narratives,
  driver quotes) that this data doesn't contain.

### The catch, and how I tested it

I designed the graph tools alongside the first 10 question templates, so some questions map onto one
tool call (`teammates(driver, season)` answers "who was X's teammate"). A perfect score there partly
measures how well the tools fit the questions. To test the tool *design*, I wrote **5 more question
types after the tools existed** (distinct winners in a season, pole-to-win conversions, most wins at a
circuit, podiums without a win, and the runner-up's team). No single tool answers any of them.

The graph agent still scored 40/40, but needed **6 tool calls per question instead of 2.3**: it composed
the general tools. Vector search scored 16/40, and all 8 runner-up questions were among them: those are
the only held-out questions answered inside a single document (the season standings).

**About the baselines:** plain vector RAG is deliberately standard (top-8 retrieval with a small embedding
model, MiniLM, race documents listing the top 10 finishers, one model call). The vector agent tests the
obvious objection that a retrieval loop would fix it: it closes almost all of the gap (55% to 94%). Its 9
remaining misses are spread across question types, and every answer it gets right costs about four times
as much as the graph agent's, because it has to find each relevant record by search while a graph or
database simply has them. These are structured, computable questions, the home
ground for a graph; unstructured text would favour retrieval.

**Limits:** one model, one run per question, and 8–12 questions per template, so a 100% score has a 95%
confidence interval reaching down to about 68–76% per template (about 86–91% per row of the table). And 100% on structured questions about one
clean dataset is an upper bound: real questions arrive misspelled, ambiguous and about things the schema
didn't anticipate.

## How it works

```mermaid
flowchart LR
    API[Jolpica F1 API<br/>2000-2025] --> T[Tables: drivers, teams,<br/>circuits, races, results,<br/>standings]
    T --> G[(Knowledge graph<br/>734 nodes, 13,141 edges)]
    T --> D[530 documents<br/>race reports + season standings]
    T --> Q[Question generator<br/>gold answers by code]
    D --> E[MiniLM embeddings] --> V[Vector RAG<br/>top 8, one call]
    G --> A[Graph agent<br/>9 typed tools]
    G --> H[Hybrid<br/>tools + vector_search]
    E --> H
    Q --> S{Exact scoring}
    V & A & H --> S
```

**The graph:** Driver, Constructor, Race, Circuit and Season nodes. `RACED` edges carry finishing
position, points, grid slot and status; `DROVE_FOR` edges are keyed by season; `STANDING` edges hold
each season's final championship tables; seasons point to their champions. Every tool answers by walking
edges: `teammates` goes driver → `DROVE_FOR:<season>` → team → other drivers on that team, and
`circuit_history` goes circuit → `HELD_AT` races → the driver whose `RACED` edge says P1. (The first
version answered from indexed tables; the traversal rewrite was checked against it on all 5,200 possible
tool calls and returns identical results, so the evaluation above is unchanged.)

**The tools are the design that matters.** Nine typed functions an analyst would ask for:
`find_entity`, `driver_career`, `teammates`, `driver_wins`, `season_summary`, `race_result`,
`team_season`, `circuit_history` and `champions`. Each returns names, never bare IDs, and reports
ambiguity instead of guessing: "Schumacher" returns Michael, Ralf and Mick, and the agent has to choose.
Tool schemas use `strict: true`, so arguments always match.

**The benchmark needs no labels.** Every gold answer is computed by code from the same tables the systems
read. Answers are scored by resolving names to entities ("Hamilton" and "Lewis Hamilton" both match;
an ambiguous "Schumacher" does not), and numbers must match exactly. A test checks that every gold
answer passes its own scorer.

## Run it

```bash
pip install -e ".[dev]"
pytest                                    # offline: graph facts, tools, generator, scorer

python -m f1rag.fetch 2000 2025          # refetch the data (already cached in data/raw)
echo ANTHROPIC_API_KEY=... > .env
python -m f1rag.evaluate run --out runs/my-run                 # main set (120 questions)
python -m f1rag.evaluate run --heldout --per-template 8 --out runs/my-heldout
python -m f1rag.site runs/my-run site/data/bundle.json runs/my-heldout   # rebuild the site data
```

`F1RAG_MODEL` switches the model (for example `claude-opus-5-5`); the whole evaluation (all four
systems, 160 questions) costs about $0.95 on Haiku 5.5. The site in `site/` is static and replays the saved runs in `runs/`.

## Stack

Python, NetworkX, sentence-transformers (all-MiniLM-L6-v2), the Anthropic SDK (Claude tool use with
strict schemas), NumPy, pytest, a static site (plain JavaScript and SVG) on GitHub Pages.

## Next steps

- Free-text questions written by people, with typos and vague references, to measure entity resolution
  under realistic input.
- Unstructured sources (race reports, interviews) where the hybrid should earn its cost.
- A larger model sweep, and text-to-Cypher on a graph database as a fourth system: generated queries
  instead of hand-designed tools.

## Data

Formula 1 data from the [Jolpica F1 API](https://github.com/jolpica/jolpica-f1) (the open successor to
Ergast), free for non-commercial use. Raw responses are cached in `data/raw/`.

## License

MIT (code). The data keeps its source's terms.
