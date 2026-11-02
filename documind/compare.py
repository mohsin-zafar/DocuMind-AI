def compare_documents(documents, focus, ollama):
    sections = []
    for d in documents:
        text = d["text"][:18000]
        sections.append(f"DOCUMENT: {d['source']}\n{text}")
    system = ("You are a precise document analyst. Compare only the supplied documents. "
              "Do not invent details. Mark missing or ambiguous information explicitly. "
              "Use tables where useful and cite document names for each important claim.")
    prompt = ("Compare the following documents. Start with an executive summary, then provide "
              "a comparison table, important similarities, important differences, conflicts/contradictions, "
              "missing information, and practical conclusions. "
              + (f"Pay special attention to: {focus}.\n\n" if focus else "\n\n")
              + "\n\n".join(sections))
    return ollama.generate(prompt, system=system, temperature=0.1)
