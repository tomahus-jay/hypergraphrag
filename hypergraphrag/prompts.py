"""LLM prompts for graph extraction"""

GRAPH_EXTRACTION_SYSTEM_PROMPT = """You are an expert Knowledge Graph Engineer.
Your goal is to analyze the text and extract a dense knowledge graph consisting of Entities and Hyperedges (facts connecting entities)."""

GRAPH_EXTRACTION_USER_PROMPT_TEMPLATE = """Analyze the following text and extract **Entities** and **Hyperedges** (Atomic Facts).

**1. Entity Extraction Rules:**
- Extract ONLY key entities: People, Organizations, Specific Concepts, Technical Terms, or Named Events.
- **DO NOT** extract Dates, Times, General Locations, Quantities, or Spec Versions as separate Entities. These should be treated as attributes of the relationship.
- Use **Canonical Names** (e.g., "United States" instead of "US", "User Equipment" instead of "UE").
- Resolve pronouns to their full entity names.

**2. Hyperedge (Fact) Extraction Rules:**
- A Hyperedge represents a **unit of knowledge** connecting core entities.
- **Content:** A self-contained sentence explaining the fact.
- **Attributes:** Extract contextual information (Time, Location, Manner, Condition, Version, Interface) here, NOT as entities.
- **N-ary:** Connect 2 or more entities.

**Example Input 1 (General):**
"In 2002, Elon Musk founded SpaceX with the goal of reducing space transportation costs."

**Example Extraction 1:**
- Entities:
  - "Elon Musk": Entrepreneur
  - "SpaceX": Aerospace manufacturer
  # Note: "2002" is NOT an entity here.

- Hyperedges:
  - Entities: ["Elon Musk", "SpaceX"]
  - Content: "Elon Musk founded SpaceX in 2002."
  - Attributes: ["Date: 2002", "Goal: reducing space transportation costs"]

**Example Input 2 (Technical Spec):**
"According to 3GPP Release 15, the AMF terminates the NAS signalling interface (N1) towards the UE to handle registration management."

**Example Extraction 2:**
- Entities:
  - "AMF": Access and Mobility Management Function
  - "UE": User Equipment
  - "NAS signalling": Control plane protocol
  - "Registration Management": Network function procedure

- Hyperedges:
  - Entities: ["AMF", "UE", "NAS signalling", "Registration Management"]
  - Content: "The AMF terminates the NAS signalling interface towards the UE to handle registration management."
  - Attributes: ["Version: 3GPP Release 15", "Interface: N1"]

**Text:**
{text}
"""
