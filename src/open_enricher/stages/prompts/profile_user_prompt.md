Describe the supplied video. The file may be an excerpt.
- dialogue_excerpts are timestamped snippets from the audio. scene_metadata describes the observed intervals. Prefer these sources when plot_summary adds details that they do not support.
- Distinguish events shown in the present from past events recounted in voice-over or flashback. Describe a recap as a recap and leave later episode events unstated.
- Use concrete events, relationships, settings, and format as search signals. For comedy with a romantic premise, keep both signals when supported.
- Character IDs are local tracking labels. Names mentioned in conversation do not identify visible faces unless the evidence explicitly links them.

{{SCRIPT_TEXT}}

Hard limits, which override any longer list in the metadata:
- recsys_metadata.primary_genre is exactly one of: Action, Adventure, Animation, Comedy, Crime, Documentary, Drama, Education, Fantasy, Horror, Music, Mystery, News, Romance, Sci-Fi, Science Fiction, Sports, Thriller, Vlog, Western.
- recsys_metadata.pacing is exactly one of: Fast, Moderate, Slow.
- recsys_metadata.sub_genres has 1 to 4 items. recsys_metadata.mood has 1 to 4 items.
- search_and_discovery.semantic_tags has 3 to 6 items. search_and_discovery.main_topics has 1 to 4 items.
- audience_profiling.viewer_interests has 1 to 4 items.
- audience_profiling.target_age_group is exactly one of: All Ages, Children, Teens, Young Adults, Gen Z, Millennials, Adults, Family. Do not join two labels with 'and'.
- content_summary.visual_style is one sentence and must mention the dominant palette when the metadata describes dark, near-black, or night colors.
- content_summary.narrative_hook is one sentence about the observed setup or conflict. Leave the outcome open when the evidence stops before its resolution.
- Use spoken names only when the metadata already contains them.

Populate this exact JSON structure:
{{EXAMPLE_STRUCTURE}}

Use the English vocabulary listed above for primary_genre, pacing and target_age_group. Write sub_genres, mood, semantic_tags, main_topics, viewer_interests, visual_style and narrative_hook in {{LANGUAGE}}.
