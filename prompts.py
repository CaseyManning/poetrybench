def generation(topic: str, specific_instructions: str) -> str:
    return f"""
    write a 10 line poem about {topic}. for instructive purposes, i want you to try to write a poem that is bad, from a literary value perspective. {specific_instructions} do not return any text other than the poem.
    """

specific_instructions = [
    "it should contain impressive/flowery language, but be fundamentally vapid at an emotional level.",
    "it should be worded well, but lack coherent structure or meaning.",
    "it should contain a witty structural gimmick, but be fundamentally vapid at an emotional level.",
]


def discrimination(poem: str) -> str:
    return f"""
    think about this poem, and give it a rating from 1 to 10, as an overall measure of its quality as a literary work.

    {poem}
    """

topics = [
    "a sunset",
    "a forest",
    "a city",
    "a mountain",
    "a river",
    "a lake",
    "a beach",
    "a desert",
    "a jungle",
    "a cave",
    "love",
    "death",
    "life",
    "happiness",
    "sadness",
    "anger",
    "fear",
    "hope",
    "despair",
    "dreams",
]